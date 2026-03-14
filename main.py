import os
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
from tradingview_screener import Query, Column
from google import genai
import io
import time

# tvDatafeedのインポートエラーを防ぐためのガード
try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    print("❌ tvDatafeedの読み込みに失敗しました。インストールを確認してください。")

# --- 1. 初期設定 ---
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-3-flash-preview"
TV_SESS_ID = os.environ.get('TV_SESSION_ID')

# Session IDを使用してインスタンス化
try:
    tv = TvDatafeed(username=None, password=None, sessionid=TV_SESS_ID)
except:
    tv = None

def get_filtered_candidates():
    print("🔭 ステップ1: スクリーナーで広域スキャン中...")
    headers = {"Cookie": f"sessionid={TV_SESS_ID}"}
    
    try:
        q = (Query().set_markets('japan')
             .select('name', 'description', 'close', 'RSI', 'SMA5', 'SMA20', 'SMA50', 'high', 'low')
             .where(Column('type').isin(['stock', 'etf']))
             .limit(100).get_scanner_data(headers=headers))
        
        df = pd.DataFrame(q[1])
        if df.empty: return df

        # 精密判定 (2次フィルター)
        df['candle_q'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['dist_25'] = (df['close'] / df['SMA20'] - 1) * 100
        
        # 判定基準を最新の憲章3.4に準拠
        active_df = df[(df['RSI'] > 53) & (df['SMA5'] > df['SMA20']) & (df['close'] > df['SMA50']) & (df['candle_q'] > 0.8)].copy()
        active_df['strategy'] = 'Active'
        
        sniper_df = df[(df['dist_25'] < -12) & (df['candle_q'] > 0.65)].copy()
        sniper_df['strategy'] = 'Sniper'
        
        # 上位5銘柄に厳選してGeminiの負担（時間）を減らす
        return pd.concat([active_df, sniper_df]).drop_duplicates(subset=['name']).head(5)
    except Exception as e:
        print(f"❌ スキャンエラー: {e}")
        return pd.DataFrame()

def create_chart_bytes(symbol, interval_type, n_bars=70):
    """チャート画像を生成してバイナリで返す"""
    if tv is None: return None
    try:
        # インターバルの設定
        interval = Interval.in_daily if interval_type == 'daily' else Interval.in_weekly
        
        # データ取得 (TSEコードを自動判別)
        tv_symbol = symbol.split(':')[-1]
        hist = tv.get_hist(symbol=tv_symbol, exchange='TSE', interval=interval, n_bars=n_bars)
        
        if hist is None or hist.empty: return None
        
        # テクニカル指標の計算
        hist['sma25'] = ta.sma(hist['close'], length=25)
        hist['sma75'] = ta.sma(hist['close'], length=75)
        
        buf = io.BytesIO()
        ap = [
            mpf.make_addplot(hist['sma25'], color='orange', width=0.8),
            mpf.make_addplot(hist['sma75'], color='blue', width=0.8)
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
        print("📭 本日、憲章3.4の精密基準をクリアする銘柄はありませんでした。")
        return

    print(f"🎯 精鋭 {len(final_candidates)} 銘柄のマルチタイムフレーム分析を開始...")
    
    for _, row in final_candidates.iterrows():
        name = row['name']
        strategy = row['strategy']
        print(f"📈 【{strategy}判定】分析実行中: {name}")
        
        img_daily = create_chart_bytes(name, 'daily')
        img_weekly = create_chart_bytes(name, 'weekly')
        
        prompt = f"""
        あなたは『運用憲章3.4』のマスター・AIストラテジストです。
        銘柄: {row.get('description', name)}
        戦略モード: {strategy}
        
        【提供データ】
        - RSI: {row['RSI']:.1f}
        - 25MA乖離率: {row['dist_25']:.1f}%
        - キャンドル品質: {row['candle_q']:.2f}
        
        【分析依頼】
        添付された「日足（1枚目）」と「週足（2枚目）」のチャートを読み取り、以下のマルチタイムフレーム分析を行ってください。
        
        1. 週足の長期トレンドが日足のシグナルを肯定しているか？
        2. 直近の視覚的なレジスタンス/サポートラインはどこか？
        3. 日本株なら楽天証券（トレスタ設定）、CFDならGMO（税制面）を考慮した提言をせよ。
        
        最後に、EXECUTE または WAIT の最終判断を下してください。
        """
        
        try:
            contents = [prompt]
            if img_daily:
                contents.append(genai.types.Part.from_bytes(data=img_daily.read(), mime_type="image/png"))
            if img_weekly:
                contents.append(genai.types.Part.from_bytes(data=img_weekly.read(), mime_type="image/png"))
            
            response = client.models.generate_content(model=MODEL_NAME, contents=contents)
            print(f"🤖 Gemini 3 Flash の回答:\n{response.text}\n" + "="*50)
            time.sleep(5) # APIレート制限への配慮
        except Exception as e:
            print(f"⚠️ Gemini分析エラー: {e}")

if __name__ == "__main__":
    main()
