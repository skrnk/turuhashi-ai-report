import os
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
from tradingview_screener import Query, Column
from google import genai
from tvDatafeed import TvDatafeed, Interval
import io
import time

# --- 1. 初期設定 ---
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-3-flash-preview"

# Session IDを使用してインスタンス化
tv = TvDatafeed(username=None, password=None, sessionid=os.environ.get('TV_SESSION_ID'))

def get_filtered_candidates():
    print("🔭 ステップ1: スクリーナーで広域スキャン中...")
    headers = {"Cookie": f"sessionid={os.environ.get('TV_SESSION_ID')}"}
    
    try:
        q = (Query().set_markets('japan')
             .select('name', 'description', 'close', 'RSI', 'SMA5', 'SMA20', 'SMA50', 'high', 'low')
             .where(Column('type').isin(['stock', 'etf']))
             .limit(150).get_scanner_data(headers=headers))
        
        df = pd.DataFrame(q[1])
        if df.empty: return df

        # 精密判定
        df['candle_q'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['dist_25'] = (df['close'] / df['SMA20'] - 1) * 100
        
        # 順張り: 基準を少しだけ最適化
        active_df = df[(df['RSI'] > 52) & (df['SMA5'] > df['SMA20']) & (df['close'] > df['SMA50']) & (df['candle_q'] > 0.75)].copy()
        active_df['strategy'] = 'Active'
        
        # 逆張り: 25MAから10%以上乖離
        sniper_df = df[(df['dist_25'] < -10) & (df['candle_q'] > 0.6)].copy()
        sniper_df['strategy'] = 'Sniper'
        
        # 上位5銘柄に厳選
        return pd.concat([active_df, sniper_df]).head(5)
    except Exception as e:
        print(f"❌ スキャンエラー: {e}")
        return pd.DataFrame()

def create_chart_bytes(symbol, interval_str, n_bars=60):
    """チャート画像を生成"""
    try:
        # tvDatafeedのインターバル設定
        interval_map = {
            'daily': Interval.in_daily,
            'weekly': Interval.in_weekly
        }
        
        # 銘柄名がTSEコードのみの場合、TSE:を補完
        tv_symbol = symbol if ":" in symbol else symbol
        
        # データ取得
        hist = tv.get_hist(symbol=tv_symbol, exchange='TSE', interval=interval_map[interval_str], n_bars=n_bars)
        if hist is None or hist.empty: return None
        
        # テクニカル追加
        hist['sma25'] = ta.sma(hist['close'], length=25)
        hist['sma75'] = ta.sma(hist['close'], length=75)
        
        buf = io.BytesIO()
        # プロット
        ap = [
            mpf.make_addplot(hist['sma25'], color='orange', width=0.7),
            mpf.make_addplot(hist['sma75'], color='blue', width=0.7)
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
        print("📭 基準をクリアする銘柄（Active/Sniper）はありませんでした。")
        return

    print(f"🎯 精鋭 {len(final_candidates)} 銘柄のマルチタイムフレーム分析を開始します...")
    
    for _, row in final_candidates.iterrows():
        name = row['name']
        strategy = row['strategy']
        print(f"📈 【{strategy}】分析実行: {name}")
        
        # 日足・週足画像の生成
        img_daily = create_chart_bytes(name, 'daily')
        img_weekly = create_chart_bytes(name, 'weekly')
        
        prompt = f"""
        あなたは『運用憲章3.4』のマスター・ストラテジストです。
        銘柄: {row.get('description', name)}
        戦略モード: {strategy}
        
        【提供データ】
        - RSI: {row['RSI']:.1f}
        - SMA20からの乖離: {row['dist_25']:.1f}%
        - キャンドル品質: {row['candle_q']:.2f}
        
        【依頼】
        添付された2枚の画像（1枚目: 日足チャート、2枚目: 週足チャート）を読み取り、
        マルチタイムフレーム分析を行ってください。
        特に「週足のトレンド」が、今回の日足のエントリーを裏付けているかを重視してください。
        
        最後に、日本株なら「楽天証券」、CFDなら「GMO」の観点を含め、
        EXECUTE または WAIT の最終判断を下してください。
        """
        
        try:
            contents = [prompt]
            if img_daily:
                contents.append(genai.types.Part.from_bytes(data=img_daily.read(), mime_type="image/png"))
            if img_weekly:
                contents.append(genai.types.Part.from_bytes(data=img_weekly.read(), mime_type="image/png"))
            
            response = client.models.generate_content(model=MODEL_NAME, contents=contents)
            print(f"🤖 Gemini 3 Flashの結論:\n{response.text}\n{'-'*50}")
            time.sleep(3) # レート制限対策
        except Exception as e:
            print(f"⚠️ Gemini分析エラー: {e}")

if __name__ == "__main__":
    main()
