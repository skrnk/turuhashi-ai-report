import os
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from tradingview_screener import Query, Column
from google import genai
import io
import time

# --- 1. 初期設定 ---
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-3-flash-preview"

# --- 2. 指標計算ロジック (pandas-ta不要版) ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_filtered_candidates():
    print("🔭 ステップ1: スクリーナーで広域スキャン中...")
    try:
        q = (Query().set_markets('japan')
             .select('name', 'description', 'close', 'RSI', 'SMA5', 'SMA20', 'SMA50', 'high', 'low')
             .where(Column('type').isin(['stock', 'etf']))
             .limit(100).get_scanner_data())
        
        df = pd.DataFrame(q[1])
        if df.empty: return df

        df['candle_q'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['dist_25'] = (df['close'] / df['SMA20'] - 1) * 100
        
        active_df = df[(df['RSI'] > 53) & (df['SMA5'] > df['SMA20']) & (df['candle_q'] > 0.8)].copy()
        active_df['strategy'] = 'Active'
        
        sniper_df = df[(df['dist_25'] < -12) & (df['candle_q'] > 0.65)].copy()
        sniper_df['strategy'] = 'Sniper'
        
        return pd.concat([active_df, sniper_df]).drop_duplicates(subset=['name']).head(5)
    except Exception as e:
        print(f"❌ スキャンエラー: {e}")
        return pd.DataFrame()

def create_chart_bytes(symbol, strategy):
    """yfinanceを使用してチャート画像を生成"""
    try:
        # 日本株コード(例: 7203)を yfinance形式(7203.T)に変換
        yf_symbol = f"{symbol.split(':')[-1]}.T" if symbol.isdigit() or symbol.startswith('TSE:') else symbol
        
        # 指数などのマッピング
        mapping = {"NI225": "^N225", "DJI": "^DJI", "SPX": "^GSPC"}
        yf_symbol = mapping.get(symbol.split(':')[-1], yf_symbol)

        data = yf.download(yf_symbol, period="6mo", interval="1d", progress=False)
        if data.empty: return None

        # 指標計算
        data['SMA25'] = data['Close'].rolling(window=25).mean()
        data['SMA75'] = data['Close'].rolling(window=75).mean()
        
        # 最新60日分を描画
        plot_data = data.tail(60)
        buf = io.BytesIO()
        ap = [
            mpf.make_addplot(plot_data['SMA25'], color='orange', width=1),
            mpf.make_addplot(plot_data['SMA75'], color='blue', width=1)
        ]
        mpf.plot(plot_data, type='candle', style='charles', addplot=ap, savefig=buf, volume=True)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"⚠️ チャート生成失敗({symbol}): {e}")
        return None

def main():
    final_candidates = get_filtered_candidates()
    if final_candidates.empty:
        print("📭 基準クリア銘柄なし")
        return

    print(f"🎯 精鋭 {len(final_candidates)} 銘柄をマルチモーダル分析中...")
    for _, row in final_candidates.iterrows():
        name = row['name']
        print(f"📈 分析: {name}")
        
        chart_img = create_chart_bytes(name, row['strategy'])
        
        prompt = f"""
        銘柄: {row.get('description', name)}
        戦略: {row['strategy']}
        憲章3.4に基づき、添付のチャート画像を分析してください。
        最後に EXECUTE または WAIT の判断を下してください。
        """
        
        try:
            contents = [prompt]
            if chart_img:
                contents.append(genai.types.Part.from_bytes(data=chart_img.read(), mime_type="image/png"))
            
            response = client.models.generate_content(model=MODEL_NAME, contents=contents)
            print(f"🤖 Geminiの結論:\n{response.text}\n" + "="*50)
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ 分析エラー: {e}")

if __name__ == "__main__":
    main()
