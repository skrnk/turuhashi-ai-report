import os
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
import google.generativeai as genai
from tradingview_screener import Query, Column
import io
import time

# --- 1. 初期設定 ---
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-3-flash-preview') # 2026年最新モデル

# GMO銘柄リストの読み込み
def load_gmo_symbols():
    try:
        return pd.read_csv('gmo_symbols.csv')
    except:
        return pd.DataFrame(columns=['Name', 'Symbol', 'Category'])

# --- 2. 憲章3.4判定ロジック (Python移植版) ---
def check_charter_logic(df_daily, df_weekly):
    if len(df_daily) < 75 or len(df_weekly) < 26: return False, "データ不足"
    
    # 日足テクニカル
    df_daily['rsi'] = ta.rsi(df_daily['close'], length=14)
    df_daily['sma5'] = ta.sma(df_daily['close'], length=5)
    df_daily['sma25'] = ta.sma(df_daily['close'], length=25)
    df_daily['sma75'] = ta.sma(df_daily['close'], length=75)
    
    # 判定要素
    curr = df_daily.iloc[-1]
    prev3 = df_daily.iloc[-4]
    slope25 = curr['sma25'] > prev3['sma25']
    candle_q = (curr['close'] - curr['low']) / (curr['high'] - curr['low']) if (curr['high'] - curr['low']) > 0 else 0
    
    # 週足トレンド
    df_weekly['sma26'] = ta.sma(df_weekly['close'], length=26)
    weekly_trend_up = df_weekly['close'].iloc[-1] > df_weekly['sma26'].iloc[-1]

    # Active(順張り)基準
    is_active = (curr['rsi'] > 50 and curr['sma5'] > curr['sma25'] and 
                 slope25 and curr['close'] > curr['sma75'] and candle_q > 0.7)
    
    if is_active and weekly_trend_up:
        return True, "EXECUTE (Active-MTF適合)"
    return False, "WAIT"

# --- 3. チャート画像生成 (メモリ上) ---
def create_chart_image(df, title, periods=60):
    plot_df = df.suffix(periods)
    buf = io.BytesIO()
    apds = [
        mpf.make_addplot(plot_df['sma25'], color='orange'),
        mpf.make_addplot(plot_df['sma75'], color='blue')
    ]
    mpf.plot(plot_df, type='candle', style='charles', addplot=apds, 
             savefig=buf, volume=True, title=title)
    buf.seek(0)
    return buf

# --- 4. メイン処理 ---
def main():
    print("🚀 スキャニング開始...")
    gmo_list = load_gmo_symbols()
    
    # 日本株スクリーニング (TradingView API経由で1次フィルター)
    # 実際の運用ではここでスクリーナーを叩き、上位10件ほどに絞ります
    target_symbols = ["TSE:7203", "TSE:6758", "TSE:8035"] # 例としてトヨタ、ソニー、東エレク
    
    # GMOリストも統合
    all_targets = target_symbols + gmo_list['Symbol'].tolist()
    
    results = []
    
    for symbol in all_targets:
        try:
            # ここで本来は価格データを取得 (tvDatafeed等)
            # 簡易化のため判定ロジックが通ったと仮定してGeminiへ投げるフローを記述
            
            # --- Gemini へのマルチモーダル分析依頼 ---
            print(f"🧐 分析中: {symbol}")
            
            # 本来はここで画像を生成
            # img_daily = create_chart_image(df_daily, f"{symbol} Daily")
            # img_weekly = create_chart_image(df_weekly, f"{symbol} Weekly")
            
            prompt = f"""
            あなたは運用憲章3.4を遵守するストラテジストです。
            銘柄: {symbol}
            現在、日足と週足のマルチタイムフレーム分析において「EXECUTE」のシグナルが出ています。
            
            提供された画像（日足・週足）を読み取り、以下の点を確認してください：
            1. 週足の長期トレンドが日足の買いをサポートしているか？
            2. 直近の抵抗帯（レジスタンス）はどこか？
            3. 【重要】日本株なら「楽天証券：トレーリングストップ」、CFDなら「GMO：節税効率」の観点から助言せよ。
            
            最終結論を EXECUTE または WAIT で示し、その理由を述べなさい。
            """
            
            # response = model.generate_content([prompt, img_daily, img_weekly])
            # results.append(response.text)
            
            time.sleep(2) # レート制限回避
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")

    # 最後にレポートを出力 (GitHub Step Summary)
    print("✅ レポート作成完了")

if __name__ == "__main__":
    main()
