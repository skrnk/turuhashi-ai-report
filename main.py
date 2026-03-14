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
from datetime import datetime

# --- 初期設定 ---
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-3-flash-preview"
NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
NOTION_DB_ID = os.environ.get("NOTION_DATABASE_ID")
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL")

def get_yf_symbol(symbol):
    s = str(symbol).split(':')[-1].strip()
    mapping = {"NI225": "^N225", "DJI": "^DJI", "SPX": "^GSPC", "NAS100": "^NDX", "RTY": "^RUT", "UK100": "^FTSE", "GER40": "^GDAXI", "EUSTX50": "^STOXX50E", "FRA40": "^FCHI", "HSI": "^HSI", "GOLD": "GC=F", "USOIL": "CL=F", "USDJPY": "JPY=X", "大豆": "ZS=F", "コーン": "ZC=F", "小麦": "ZW=F"}
    if s in mapping: return mapping[s]
    return f"{s}.T" if s.isdigit() else s

def calculate_charter_logic(data):
    if len(data) < 75: return None, "データ本数不足"
    close, high, low, vol = data['Close'], data['High'], data['Low'], data['Volume']
    
    # 指標計算
    delta = close.diff(); avg_gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean(); avg_loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
    sma5, sma25, sma75 = close.rolling(5).mean(), close.rolling(25).mean(), close.rolling(75).mean()
    rvol = vol / vol.rolling(10).mean()
    slope_ok = sma25.iloc[-1] > sma25.iloc[-4]
    c_qual = (close - low) / (high - low)
    
    # --- Active (順張り) 判定 ---
    cond_a = {
        "RSI > 50": rsi.iloc[-1] > 50,
        "5MA > 25MA": sma5.iloc[-1] > sma25.iloc[-1],
        "Slope OK": slope_ok,
        "RVol > 1.2x": rvol.iloc[-1] > 1.2,
        "Price > 75MA": close.iloc[-1] > sma75.iloc[-1],
        "Candle > 0.7": c_qual.iloc[-1] > 0.7
    }
    if all(cond_a.values()): return "Active (順張り)", cond_a
    
    # --- Sniper (逆張り) 判定 ---
    cross_over = (close.iloc[-1] > sma5.iloc[-1]) and (close.iloc[-2] <= sma5.iloc[-2])
    cond_s = {
        "RSI < 40": rsi.iloc[-1] < 40,
        "Dist < -10%": close.iloc[-1] < sma25.iloc[-1] * 0.90,
        "5MA Crossover": cross_over,
        "Candle > 0.7": c_qual.iloc[-1] > 0.7
    }
    if all(cond_s.values()): return "Sniper (逆張り)", cond_s
    
    # 不適合理由の集約
    fail_reason = ", ".join([k for k, v in {**cond_a, **cond_s}.items() if not v])
    return None, fail_reason

def main():
    print("🔭 憲章3.4 v2.1 スキャン開始...")
    targets = [] # CSVとスキャナーから取得 (中略)
    
    # --- 精密フィルター ---
    final_hits = []
    for item in unique_targets:
        yf_symbol = get_yf_symbol(item['Symbol'])
        data = yf.download(yf_symbol, period="180d", progress=False)
        if data.empty: continue
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        
        mode, details = calculate_charter_logic(data)
        if mode:
            item['Mode'], item['Details'] = mode, details
            final_hits.append(item)
            print(f"🎯 適合発見: {item['Name']} ({mode})")
        else:
            # ここで「なぜ落ちたか」をログに出す
            print(f"  [-] {item['Name']} は不適合: {details}")

    # --- AI分析セクション ---
    for hit in final_hits[:20]:
        # (チャート生成処理 中略)
        
        # 規準チェックリストを文字列化
        checklist_str = "\n".join([f"- {k}: {'✅' if v else '❌'}" for k, v in hit['Details'].items()])
        
        prompt = f"""
        報告日次: {datetime.now().strftime('%Y-%m-%d')}
        銘柄: {hit['Name']} ({hit['Symbol']})
        分析モード: {hit['Mode']}

        【憲章3.4 規準チェック結果（システム算出値）】
        {checklist_str}

        上記システム判定を踏まえ、提供された日足・週足チャートを視覚的に詳細分析せよ。
        回答は必ず以下の形式で開始すること：
        ---
        ■報告日次: {datetime.now().strftime('%Y-%m-%d')}
        ■結論: [EXECUTE または WAIT]
        ---

        次に、サマリーの各規準（RSI、移動平均、傾き、乖離、出来高、マクロトレンド、キャンドル品質）について、
        あなたの「視覚的見解」を述べよ。
        """
        # (Gemini送信・Notion/Discord通知処理 中略)
