import os
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from tradingview_screener import Query, Column
from google import genai
import io
import time
import requests
import json
from datetime import datetime, timedelta, timezone

# --- タイムゾーン設定 (JST) ---
JST = timezone(timedelta(hours=9), 'JST')

# --- 初期設定 ---
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-3-flash-preview" # ユーザー元の設定を維持
NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
NOTION_DB_ID = os.environ.get("NOTION_DATABASE_ID")
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL")

# --- 1. シンボル翻訳辞書 ---
def get_yf_symbol(symbol):
    s = str(symbol).split(':')[-1].strip()
    mapping = {
        "NI225": "^N225", "DJI": "^DJI", "SPX": "^GSPC", "NAS100": "^NDX",
        "RTY": "^RUT", "UK100": "^FTSE", "GER40": "^GDAXI", "EUSTX50": "^STOXX50E",
        "FRA40": "^FCHI", "HSI": "^HSI", "XIN9": "000001.SS", "NIFTY": "^NSEI",
        "XAUUSD": "GC=F", "XAGUSD": "SI=F", "XPTUSD": "PL=F", "XPDUSD": "PA=F",
        "WTI": "CL=F", "BRENT": "BZ=F", "NATGAS": "NG=F",
        "SOYBNUSD": "ZS=F", "CORNUSD": "ZC=F", "WHEATUSD": "ZW=F", "COPPER": "HG=F"
    }
    return mapping.get(s, s)

# --- 【新規追加】TradingView実測RVol取得 ---
def get_tv_rvol_actual(symbol):
    """TradingViewから直接RVolを取得。取得不能時はNoneを返す"""
    exchange = "TSE"
    tv_ticker = symbol
    if ".T" in symbol:
        exchange = "TSE"
        tv_ticker = symbol.replace(".T", "")
    elif symbol == "ZS=F": exchange = "CME"; tv_ticker = "ZS1!"
    elif symbol == "ZW=F": exchange = "CME"; tv_ticker = "ZW1!"
    elif symbol == "ZC=F": exchange = "CME"; tv_ticker = "ZC1!"
    else: return None
    try:
        q = Query().set_tickers(f"{exchange}:{tv_ticker}").select('relative_volume')
        _, data = q.get_scanner_data()
        if data: return float(data[0]['relative_volume'])
    except: pass
    return None

# --- 2. 憲章3.4 判定ロジック (RVol箇所のみ改善) ---
def calculate_charter_logic(data, symbol=""):
    # yfinance最新版のマルチインデックス対策（これがないと現在のyfinanceでは動作しません）
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if len(data) < 75: return None, "データ本数不足"
    
    close = data['Close']
    high = data['High']
    low = data['Low']
    vol = data['Volume']

    # RSI (Wilder's RMA)
    delta = close.diff()
    avg_gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rsi_series = (100 - (100 / (1 + (avg_gain / avg_loss))))
    rsi = rsi_series.iloc[-1]

    sma5 = close.rolling(5).mean()
    sma25 = close.rolling(25).mean()
    sma75 = close.rolling(75).mean()

    # --- RVol改善点：TradingView API優先 + VIX連動型 ---
    tv_rvol = get_tv_rvol_actual(symbol)
    rvol = tv_rvol if tv_rvol is not None else (vol / vol.rolling(20).mean()).iloc[-1]
    
    # VIXチェック (現在のVIX 27を考慮し、25超なら1.5xを要求)
    try:
        vix_val = yf.download("^VIX", period="1d", progress=False)['Close'].iloc[-1]
    except:
        vix_val = 20
    rvol_threshold = 1.5 if vix_val > 25 else 1.2
    # ---------------------------------------------

    slope_ok = sma25.iloc[-1] > sma25.iloc[-4]
    c_qual = ((close - low) / (high - low)).iloc[-1]

    cond_a = {
        "RSI > 50": rsi > 50, 
        "5MA > 25MA": sma5.iloc[-1] > sma25.iloc[-1], 
        "Slope OK": slope_ok, 
        f"RVol > {rvol_threshold}x": rvol > rvol_threshold, 
        "Price > 75MA": close.iloc[-1] > sma75.iloc[-1], 
        "Candle > 0.7": c_qual > 0.7
    }

    cross_over = (close.iloc[-1] > sma5.iloc[-1]) and (close.iloc[-2] <= sma5.iloc[-2])
    cond_s = {
        "RSI < 40": rsi < 40, 
        "Dist < -10%": close.iloc[-1] < sma25.iloc[-1] * 0.90, 
        "5MA Crossover": cross_over, 
        "Candle > 0.7": c_qual > 0.7
    }

    chk_str = "\n".join([f"{'●' if v else '○'} {k}" for k, v in {**cond_a, **cond_s}.items()])

    if all(cond_a.values()): return "Active (順張り)", chk_str
    if all(cond_s.values()): return "Sniper (逆張り)", chk_str
    
    fails = [k for k, v in {**cond_a, **cond_s}.items() if not v]
    return None, f"不適合: {', '.join(fails[:3])}"

# --- 3. メイン実行ループ (Discord/Gemini部はユーザー元のロジックを完全維持) ---
def run_t_strategy():
    target_symbols = ["4063.T", "9412.T", "6356.T", "7186.T", "ZS=F", "ZW=F", "ZC=F"]
    
    for s in target_symbols:
        try:
            print(f"Checking {s}...")
            df_d = yf.download(s, start=(datetime.now(JST) - timedelta(days=120)).strftime('%Y-%m-%d'), progress=False)
            df_w = yf.download(s, start=(datetime.now(JST) - timedelta(days=600)).strftime('%Y-%m-%d'), interval="1wk", progress=False)
            
            if df_d.empty or df_w.empty: continue
            
            mode, chk_str = calculate_charter_logic(df_d, s) # 引数sを追加
            
            if mode:
                # 日足チャート
                img_d = io.BytesIO()
                mpf.plot(df_d.tail(60), type='candle', style='charles', volume=True, mav=(5,25,75), savefig=img_d)
                img_d.seek(0)
                
                # 週足チャート
                img_w = io.BytesIO()
                mpf.plot(df_w.tail(40), type='candle', style='charles', volume=True, mav=(13,26), savefig=img_w)
                img_w.seek(0)
                
                curr_time_jst = datetime.now(JST).strftime('%Y-%m-%d %H:%M')
                # ユーザー元のプロンプト構造を完全維持
                prompt = f"銘柄: {s}\n戦略モード: {mode}\n【システムチェック値】\n{chk_str}\n\n【分析指令】\n1. ビジネスモデル解析: 業界トレンドと優位性を述べよ。\n2. テクニカル検証: 数値が視覚的にも本物のトレンドか。\n3. 総合評価: Tier S, Tier A, Tier B以下を決定せよ。\n※コモディティは構造的トレンド明白な場合のみTier Sとせよ。\n\n■Symbol: {s}\n■報告日次: {curr_time_jst}\n■結論: [Tier S / Tier A / Tier B]\n■売買執行: [EXECUTE または WAIT]\n---\n【憲章チェック】\n{chk_str}\n【業界分析】\n(記述)\n【総合分析】\n(記述)"
                
                # 画像2枚をGeminiに送る元のロジックを維持
                response = client.models.generate_content(
                    model=MODEL_NAME, 
                    contents=[
                        prompt, 
                        genai.types.Part.from_bytes(data=img_d.read(), mime_type="image/png"), 
                        genai.types.Part.from_bytes(data=img_w.read(), mime_type="image/png")
                    ]
                )
                
                res_text = response.text
                tier = "Tier B"
                if "Tier S" in res_text.split('---')[0]: tier = "Tier S"
                elif "Tier A" in res_text.split('---')[0]: tier = "Tier A"
                
                if tier in ["Tier S", "Tier A"]:
                    # Discord通報 (画像付き)
                    if DISCORD_WEBHOOK:
                        img_d.seek(0)
                        files = {'file1': ('chart_d.png', img_d, 'image/png')}
                        payload = {"content": f"🚀 **{s} {mode} {tier} シグナル発生**\n{res_text}"}
                        requests.post(DISCORD_WEBHOOK, data=payload, files=files)
                        print(f"Signal sent for {s}")
                    
            time.sleep(1)
        except Exception as e:
            print(f"Error checking {s}: {e}")

if __name__ == "__main__":
    run_t_strategy()
