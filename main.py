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

def get_filtered_candidates():
    print("🔭 ステップ1: スクリーナーで広域スキャン中...")
    try:
        q = (Query().set_markets('japan')
             .select('name', 'description', 'close', 'RSI', 'SMA5', 'SMA20', 'SMA50', 'high', 'low')
             .where(Column('type').isin(['stock', 'etf']))
             .limit(100).get_scanner_data())
        
        df = pd.DataFrame(q[1])
        if df.empty: return df

        # 判定用指標
        df['candle_q'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['dist_25'] = (df['close'] / df['SMA20'] - 1) * 100
        
        # 憲章3.4 精密足切り
        active_df = df[(df['RSI'] > 53) & (df['SMA5'] > df['SMA20']) & (df['candle_q'] > 0.8)].copy()
        active_df['strategy'] = 'Active (順張り)'
        
        sniper_df = df[(df['dist_25'] < -12) & (df['candle_q'] > 0.65)].copy()
        sniper_df['strategy'] = 'Sniper (逆張り)'
        
        return pd.concat([active_df, sniper_df]).drop_duplicates(subset=['name']).head(5)
    except Exception as e:
        print(f"❌ スキャンエラー: {e}")
        return pd.DataFrame()

def create_chart_bytes(symbol):
    """yfinanceの型エラーとマルチインデックスを解消してチャート生成"""
    try:
        ticker = symbol.split(':')[-1]
        yf_symbol = f"{ticker}.T" if ticker.isdigit() else ticker
        mapping = {"NI225": "^N225", "DJI": "^DJI", "SPX": "^GSPC"}
        yf_symbol = mapping.get(ticker, yf_symbol)

        data = yf.download(yf_symbol, period="1y", interval="1d", progress=False)
        if data.empty: return None

        # yfinance 0.2.x 対策: マルチインデックスの解除
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # 全データをfloatに強制変換（エラー: must be ALL float or int への対策）
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[cols].astype(float)
        data = data.dropna()

        # 指標
        data['SMA25'] = data['Close'].rolling(window=25).mean()
        data['SMA75'] = data['Close'].rolling(window=75).mean()
        
        plot_data = data.tail(60).copy()
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

    print(f"🎯 精鋭 {len(final_candidates)} 銘柄をAIマルチモーダル分析開始")
    for _, row in final_candidates.iterrows():
        name = row['name']
        strategy_label = row['strategy'] # "Active (順張り)" or "Sniper (逆張り)"
        
        print(f"\n{'='*20} {strategy_label} 判定中: {name} {'='*20}")
        
        chart_img = create_chart_bytes(name)
        
        # プロンプトの指示を「分析タイプの明記」に特化
        prompt = f"""
        銘柄: {row.get('description', name)} ({name})
        分析タイプ: {strategy_label}

        【最優先指示】
        回答の冒頭に必ず 『【分析タイプ: {strategy_label}】』 と一行目に記載すること。
        
        【分析指示】
        提供されたチャート（25MA:橙, 75MA:青）を読み取り、『運用憲章3.4』に基づき
        「なぜこのタイプ（順張りまたは逆張り）として選出されたか」を考慮して分析せよ。
        
        最後に、EXECUTE（実行） または WAIT（待機） を理由と共に示せ。
        """
        
        try:
            contents = [prompt]
            if chart_img:
                contents.append(genai.types.Part.from_bytes(data=chart_img.read(), mime_type="image/png"))
                print("📸 チャート画像の添付に成功しました。")
            else:
                print("⚠️ 画像なしでテキスト分析のみ実行します。")
            
            response = client.models.generate_content(model=MODEL_NAME, contents=contents)
            print(f"🤖 Geminiの結論:\n{response.text}\n")
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ 分析エラー: {e}")

if __name__ == "__main__":
    main()
