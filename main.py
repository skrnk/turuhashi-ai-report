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
# モデル名を環境に合わせて修正
MODEL_NAME = "gemini-3-flash-preview" 
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

# --- 2. TradingView API 直接参照ロジック ---
def get_tv_actual_rvol(yf_symbol):
    exchange = "TSE"
    tv_ticker = yf_symbol
    
    if ".T" in yf_symbol:
        exchange = "TSE"
        tv_ticker = yf_symbol.replace(".T", "")
    elif yf_symbol == "ZS=F": exchange = "CME"; tv_ticker = "ZS1!"
    elif yf_symbol == "ZW=F": exchange = "CME"; tv_ticker = "ZW1!"
    elif yf_symbol == "ZC=F": exchange = "CME"; tv_ticker = "ZC1!"
    elif yf_symbol == "GC=F": exchange = "COMEX"; tv_ticker = "GC1!"
    elif yf_symbol == "CL=F": exchange = "NYMEX"; tv_ticker = "CL1!"
    else: return None

    try:
        q = Query().set_tickers(f"{exchange}:{tv_ticker}").select('relative_volume')
        _, data = q.get_scanner_data()
        if data:
            val = data[0]['relative_volume']
            return float(val) if val is not None else None
    except:
        pass
    return None

# --- 3. 憲章3.4 v3.2 判定ロジック ---
def calculate_charter_logic(data, symbol, current_vix=20):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    active_data = data[data['Volume'] > 0].copy()
    if len(active_data) < 75: return None, "有効データ不足"
    
    close = active_data['Close'].squeeze()
    high = active_data['High'].squeeze()
    low = active_data['Low'].squeeze()
    vol = active_data['Volume'].squeeze()

    delta = close.diff()
    avg_gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rsi_series = (100 - (100 / (1 + (avg_gain / avg_loss))))
    rsi = float(rsi_series.iloc[-1])

    sma5 = close.rolling(5).mean()
    sma25 = close.rolling(25).mean()
    sma75 = close.rolling(75).mean()

    tv_rvol = get_tv_actual_rvol(symbol)
    rvol = float(tv_rvol) if tv_rvol is not None else float((vol / vol.rolling(20).mean()).iloc[-1])
    
    vix_val = float(current_vix.iloc[-1]) if hasattr(current_vix, 'iloc') else float(current_vix)
    rvol_threshold = 1.5 if vix_val > 25 else 1.2
    
    slope_ok = bool(sma25.iloc[-1] > sma25.iloc[-4])
    candle_range = float(high.iloc[-1] - low.iloc[-1])
    c_qual = float((close.iloc[-1] - low.iloc[-1]) / candle_range) if candle_range > 0 else 0

    cond_a = {
        "RSI > 50": rsi > 50,
        "5MA > 25MA": bool(sma5.iloc[-1] > sma25.iloc[-1]),
        "Slope OK": slope_ok,
        f"RVol > {rvol_threshold}x": rvol > rvol_threshold,
        "Price > 75MA": bool(close.iloc[-1] > sma75.iloc[-1]),
        "Candle > 0.7": c_qual > 0.7
    }

    cross_over = bool((close.iloc[-1] > sma5.iloc[-1]) and (close.iloc[-2] <= sma5.iloc[-2]))
    cond_s = {
        "RSI < 40": rsi < 40,
        "Dist < -10%": bool(close.iloc[-1] < sma25.iloc[-1] * 0.90),
        "5MA Crossover": cross_over,
        "Candle > 0.7": c_qual > 0.7
    }

    chk_str = "\n".join([f"{'●' if v else '○'} {k}" for k, v in {**cond_a, **cond_s}.items()])

    if all(cond_a.values()): return "Active (順張り)", chk_str
    if all(cond_s.values()): return "Sniper (逆張り)", chk_str
    
    fails = [k for k, v in {**cond_a, **cond_s}.items() if not v]
    return None, f"不適合: {', '.join(fails[:3])}"

# --- 4. メインループ ---
def run_t_strategy():
    try:
        vix_df = yf.download("^VIX", period="5d", progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        current_vix = vix_df['Close'].iloc[-1]
    except:
        current_vix = 20

    target_symbols = ["4063.T", "9412.T", "6356.T", "7186.T", "ZS=F", "ZW=F", "ZC=F"]
    
    for s in target_symbols:
        try:
            print(f"Checking {s}...")
            df = yf.download(s, start=(datetime.now(JST) - timedelta(days=120)).strftime('%Y-%m-%d'), progress=False)
            if df.empty: continue
            
            mode, chk_str = calculate_charter_logic(df, s, current_vix)
            
            if mode:
                img_d = io.BytesIO()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                mpf.plot(df.tail(60), type='candle', style='charles', volume=True, mav=(5,25,75), savefig=img_d)
                img_d.seek(0)
                
                curr_time_jst = datetime.now(JST).strftime('%Y-%m-%d %H:%M')
                prompt = f"銘柄: {s}\n戦略モード: {mode}\n【システムチェック値】\n{chk_str}\n\n■Symbol: {s}\n■報告日次: {curr_time_jst}\n■結論: [Tier S / Tier A / Tier B]\n■売買執行: [EXECUTE または WAIT]\n---\n【憲章チェック】\n{chk_str}"
                
                # API呼び出しの修正
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[
                        prompt, 
                        genai.types.Part.from_bytes(data=img_d.read(), mime_type="image/png")
                    ]
                )
                
                if DISCORD_WEBHOOK and ("Tier S" in response.text or "Tier A" in response.text):
                    payload = {"content": f"🚀 **{s} {mode} シグナル発生**\n{response.text}"}
                    requests.post(DISCORD_WEBHOOK, json=payload)
                    print(f"Signal sent for {s}")
            else:
                print(f"{s}: {chk_str}")
                    
            time.sleep(1)
        except Exception as e:
            print(f"Error checking {s}: {e}")

if __name__ == "__main__":
    run_t_strategy()
