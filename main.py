import os
import pandas as pd
import io
import time

# --- ライブラリの読み込みガード ---
try:
    import pandas_ta as ta
    import mplfinance as mpf
    from tradingview_screener import Query, Column
    from google import genai
    from tvdatafeed import TvDatafeed, Interval
except ImportError as e:
    print(f"❌ ライブラリのインポートに失敗しました: {e}")
    # 致命的なエラーとして終了
    exit(1)

# --- 1. 初期設定 ---
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-3-flash-preview"
TV_SESS_ID = os.environ.get('TV_SESSION_ID')

# Session IDを使用してインスタンス化
tv = TvDatafeed(username=None, password=None, sessionid=TV_SESS_ID)

def get_filtered_candidates():
    print("🔭 ステップ1: スクリーナーで広域スキャン中...")
    headers = {"Cookie": f"sessionid={TV_SESS_ID}"} if TV_SESS_ID else None
    
    try:
        q = (Query().set_markets('japan')
             .select('name', 'description', 'close', 'RSI', 'SMA5', 'SMA20', 'SMA50', 'high', 'low')
             .where(Column('type').isin(['stock', 'etf']))
             .limit(100).get_scanner_data(headers=headers))
        
        df = pd.DataFrame(q[1])
        if df.empty: return df

        # 2次フィルター
        df['candle_q'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['dist_25'] = (df['close'] / df['SMA20'] - 1) * 100
        
        # 順張り/逆張りの判定
        active_df = df[(df['RSI'] > 53) & (df['SMA5'] > df['SMA20']) & (df['close'] > df['SMA50']) & (df['candle_q'] > 0.8)].copy()
        active_df['strategy'] = 'Active'
        
        sniper_df = df[(df['dist_25'] < -12) & (df['candle_q'] > 0.65)].copy()
        sniper_df['strategy'] = 'Sniper'
        
        return pd.concat([active_df, sniper_df]).drop_duplicates(subset=['name']).head(5)
    except Exception as e:
        print(f"❌ スキャンエラー: {e}")
        return pd.DataFrame()

def create_chart_bytes(symbol, interval_type):
    try:
        interval = Interval.in_daily if interval_type == 'daily' else Interval.in_weekly
        tv_symbol = symbol.split(':')[-1]
        
        hist = tv.get_hist(symbol=tv_symbol, exchange='TSE', interval=interval, n_bars=70)
        if hist is None or hist.empty: return None
        
        # テクニカル指標の付与
        hist.ta.sma(length=25, append=True)
        hist.ta.sma(length=75, append=True)
        
        # カラム名特定 (SMA_25 等に対応)
        sma25 = [c for c in hist.columns if 'SMA_25' in c or 'sma25' in c][0]
        sma75 = [c for c in hist.columns if 'SMA_75' in c or 'sma75' in c][0]
        
        buf = io.BytesIO()
        ap = [
            mpf.make_addplot(hist[sma25], color='orange', width=0.8),
            mpf.make_addplot(hist[sma75], color='blue', width=0.8)
        ]
        mpf.plot(hist, type='candle', style='charles', addplot=ap, savefig=buf, volume=True, tight_layout=True)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"⚠️ チャート生成失敗({symbol}): {e}")
        return None

def main():
    final_candidates = get_filtered_candidates()
    
    if final_candidates.empty:
        print("📭 本日、基準をクリアした銘柄はありませんでした。")
        return

    print(f"🎯 精鋭 {len(final_candidates)} 銘柄をAIマルチモーダル分析します...")
    
    for _, row in final_candidates.iterrows():
        name = row['name']
        strategy = row['strategy']
        print(f"📈 【{strategy}】分析中: {name}")
        
        img_daily = create_chart_bytes(name, 'daily')
        img_weekly = create_chart_bytes(name, 'weekly')
        
        prompt = f"""
        あなたは『運用憲章3.4』のマスター・AIストラテジストです。
        銘柄: {row.get('description', name)} ({name})
        
        添付の日足・週足チャートをマルチタイムフレーム分析し、
        EXECUTE または WAIT の最終判断を理由と共に示せ。
        """
        
        try:
            contents = [prompt]
            if img_daily:
                contents.append(genai.types.Part.from_bytes(data=img_daily.read(), mime_type="image/png"))
            if img_weekly:
                contents.append(genai.types.Part.from_bytes(data=img_weekly.read(), mime_type="image/png"))
            
            response = client.models.generate_content(model=MODEL_NAME, contents=contents)
            print(f"🤖 Geminiの結論:\n{response.text}\n" + "="*50)
            time.sleep(5)
        except Exception as e:
            print(f"⚠️ Gemini分析エラー: {e}")

if __name__ == "__main__":
    main()
