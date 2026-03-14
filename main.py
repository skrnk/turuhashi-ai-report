import os
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
from tradingview_screener import Query, Column
from google import genai
from tvdatafeed import TvDatafeed, Interval
import io
import time

# --- 1. 初期設定 ---
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-3-flash-preview"
tv = TvDatafeed(sessionid=os.environ.get('TV_SESSION_ID'))

def get_filtered_candidates():
    print("🔭 ステップ1: スクリーナーで広域スキャン中...")
    headers = {"Cookie": f"sessionid={os.environ.get('TV_SESSION_ID')}"}
    
    # 憲章3.4の基本条件で1次抽出
    q = (Query().set_markets('japan')
         .select('name', 'description', 'close', 'RSI', 'SMA5', 'SMA20', 'SMA50', 'high', 'low')
         .where(Column('type').isin(['stock', 'etf']))
         .limit(150).get_scanner_data(headers=headers))
    
    df = pd.DataFrame(q[1])
    
    # --- 精密判定（ここで3〜5銘柄まで絞り込む） ---
    df['candle_q'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['dist_25'] = (df['close'] / df['SMA20'] - 1) * 100
    
    # Active: 5MA > 25MA かつ 価格 > 50MA かつ 強気の足
    active_df = df[(df['RSI'] > 53) & (df['SMA5'] > df['SMA20']) & (df['close'] > df['SMA50']) & (df['candle_q'] > 0.8)].copy()
    active_df['strategy'] = 'Active'
    
    # Sniper: 25MA乖離が激しく、下げ止まりの兆し
    sniper_df = df[(df['dist_25'] < -12) & (df['candle_q'] > 0.6)].copy()
    sniper_df['strategy'] = 'Sniper'
    
    return pd.concat([active_df, sniper_df]).sort_values('RSI', ascending=False).head(5)

def create_chart_bytes(symbol, interval, n_bars=60):
    """TradingViewからデータを取得してチャート画像を生成"""
    try:
        # 市場コードの補完（簡易版）
        tv_symbol = symbol if ":" in symbol else f"TSE:{symbol}"
        hist = tv.get_hist(symbol=tv_symbol, exchange='TSE', interval=interval, n_bars=n_bars)
        if hist is None: return None
        
        # 移動平均線の追加
        hist['sma25'] = ta.sma(hist['close'], length=25)
        hist['sma75'] = ta.sma(hist['close'], length=75)
        
        buf = io.BytesIO()
        ap = [mpf.make_addplot(hist['sma25'], color='orange'), mpf.make_addplot(hist['sma75'], color='blue')]
        mpf.plot(hist, type='candle', style='charles', addplot=ap, savefig=buf, volume=True)
        buf.seek(0)
        return buf
    except:
        return None

def main():
    final_candidates = get_filtered_candidates()
    
    if final_candidates.empty:
        print("📭 基準をクリアする銘柄はありませんでした。")
        return

    print(f"🎯 精鋭 {len(final_candidates)} 銘柄のマルチタイムフレーム分析を開始します...")
    
    for _, row in final_candidates.iterrows():
        name = row['name']
        strategy = row['strategy']
        print(f"📈 分析実行: {name} ({strategy})")
        
        # 日足と週足のチャート生成
        img_daily = create_chart_bytes(name, Interval.in_daily)
        img_weekly = create_chart_bytes(name, Interval.in_weekly)
        
        prompt = f"""
        あなたは『運用憲章3.4』のマスター・ストラテジストです。
        銘柄: {row.get('description', name)}
        戦略: {strategy}
        
        提供した2枚の画像（1枚目: 日足、2枚目: 週足）を分析してください。
        
        【分析要件】
        1. 週足のトレンド（森）は、日足のエントリー（木）を肯定しているか？
        2. 直近のレジスタンスラインを視覚的に特定せよ。
        3. 日本株なら楽天（トレスタ）、CFDならGMOの観点で結論を出せ。
        
        最終判断: EXECUTE または WAIT
        """
        
        try:
            # 画像とテキストをセットで送信
            contents = [prompt]
            if img_daily: contents.append(genai.types.Part.from_bytes(data=img_daily.read(), mime_type="image/png"))
            if img_weekly: contents.append(genai.types.Part.from_bytes(data=img_weekly.read(), mime_type="image/png"))
            
            response = client.models.generate_content(model=MODEL_NAME, contents=contents)
            print(f"🤖 Gemini 3 Flashの結論:\n{response.text}\n{'-'*50}")
            time.sleep(5) # 画像処理負荷を考慮して少し長めに待機
        except Exception as e:
            print(f"⚠️ 解析エラー: {e}")

if __name__ == "__main__":
    main()
