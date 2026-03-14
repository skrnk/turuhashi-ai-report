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

# --- 1. テクニカル指標計算 (PineScript v2.1 完全同期) ---
def calculate_charter_logic(data):
    if len(data) < 75: return None
    close, high, low, vol = data['Close'], data['High'], data['Low'], data['Volume']
    
    # RSI (Wilder法 / RMA)
    delta = close.diff()
    avg_gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
    
    # SMA
    sma5 = close.rolling(5).mean()
    sma25 = close.rolling(25).mean()
    sma75 = close.rolling(75).mean()
    rvol = vol / vol.rolling(10).mean()
    
    # 25MA傾き (3日前比較)
    slope_ok = sma25.iloc[-1] > sma25.iloc[-4]
    
    # キャンドル品質
    f_candle = ((close - low) / (high - low)).iloc[-1] > 0.7
    
    # --- Active (順張り) 判定 ---
    is_active = (rsi.iloc[-1] > 50) and \
                (sma5.iloc[-1] > sma25.iloc[-1]) and \
                slope_ok and \
                (rvol.iloc[-1] > 1.2) and \
                (close.iloc[-1] > sma75.iloc[-1]) and \
                f_candle
    
    # --- Sniper (逆張り) 判定 ---
    cross_over = (close.iloc[-1] > sma5.iloc[-1]) and (close.iloc[-2] <= sma5.iloc[-2])
    is_sniper = (rsi.iloc[-1] < 40) and \
                (close.iloc[-1] < sma25.iloc[-1] * 0.90) and \
                cross_over and \
                f_candle
    
    if is_active: return "Active (順張り)"
    if is_sniper: return "Sniper (逆張り)"
    return None

def get_gmo_list():
    if not os.path.exists('gmo_symbols.csv'): return []
    try:
        df = pd.read_csv('gmo_symbols.csv', sep=None, engine='python')
        df.columns = df.columns.str.strip()
        return df.to_dict('records')
    except: return []

def get_yf_symbol(ticker):
    mapping = {"NI225": "^N225", "DJI": "^DJI", "SPX": "^GSPC", "NAS100": "^IXIC", "GOLD": "GC=F", "OIL": "CL=F", "USDJPY": "JPY=X"}
    ticker = str(ticker).strip()
    return mapping.get(ticker, f"{ticker}.T" if ticker.isdigit() else ticker)

def create_chart_bytes(symbol, interval="1d", n_bars=70):
    try:
        yf_symbol = get_yf_symbol(symbol)
        data = yf.download(yf_symbol, period="2y", interval=interval, progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data['SMA25'] = data['Close'].rolling(25).mean()
        data['SMA75'] = data['Close'].rolling(75).mean()
        plot_data = data.tail(n_bars).copy()
        buf = io.BytesIO()
        ap = [mpf.make_addplot(plot_data['SMA25'], color='orange'), mpf.make_addplot(plot_data['SMA75'], color='blue')]
        mpf.plot(plot_data, type='candle', style='charles', addplot=ap, volume=True, returnfig=True, datetime_format='%y-%m-%d', tight_layout=True, savefig=buf)
        buf.seek(0)
        return buf
    except: return None

def post_to_notion(name, strategy, judge, analysis):
    if not NOTION_TOKEN or not NOTION_DB_ID: return
    headers = {"Authorization": f"Bearer {NOTION_TOKEN}", "Content-Type": "application/json", "Notion-Version": "2022-06-28"}
    data = {"parent": {"database_id": NOTION_DB_ID}, "properties": {"Name": {"title": [{"text": {"content": name}}]}, "Strategy": {"select": {"name": strategy}}, "Judge": {"select": {"name": judge}}, "Date": {"date": {"start": datetime.now().isoformat()}}, "Analysis": {"rich_text": [{"text": {"content": analysis[:1900]}}]}}}
    requests.post("https://api.notion.com/v1/pages", headers=headers, json=data)

def post_to_discord(name, strategy, judge, analysis, img_d, img_w):
    if not DISCORD_WEBHOOK: return
    color = 0x00ff00 if judge == "EXECUTE" else 0x808080
    payload = {"embeds": [{"title": f"🚀 【{strategy}】: {name}", "description": analysis[:1800], "color": color, "image": {"url": "attachment://daily.png"}}, {"image": {"url": "attachment://weekly.png"}}]}
    files = {"file1": ("daily.png", img_d, "image/png"), "file2": ("weekly.png", img_w, "image/png")}
    requests.post(DISCORD_WEBHOOK, data={"payload_json": json.dumps(payload)}, files=files)

def main():
    print("🔭 憲章3.4 v2.1 真の完全同期スキャン開始...")
    targets = get_gmo_list()
    try:
        q = (Query().set_markets('japan').select('name', 'description', 'close').where(Column('close') <= 12000, Column('type').isin(['stock', 'etf'])).limit(150).get_scanner_data())
        targets += [{'Name': r['description'], 'Symbol': r['name']} for r in q[1]]
    except: pass
    
    seen = set()
    final_hits = []
    for t in targets:
        if t['Symbol'] not in seen:
            try:
                data = yf.download(get_yf_symbol(t['Symbol']), period="150d", progress=False)
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                mode = calculate_charter_logic(data)
                if mode:
                    t['Mode'] = mode
                    final_hits.append(t)
                    print(f"🎯 適合: {t['Name']} ({mode})")
            except: pass
            seen.add(t['Symbol'])

    for hit in final_hits[:25]:
        name, symbol, mode = hit['Name'], hit['Symbol'], hit['Mode']
        img_d = create_chart_bytes(symbol, interval="1d")
        img_w = create_chart_bytes(symbol, interval="1wk", n_bars=40)
        if not img_d or not img_w: continue

        prompt = f"銘柄: {name} ({symbol})\n戦略: {mode}\n憲章3.4 v2.1に基づき分析せよ。最後に EXECUTE または WAIT を示せ。"
        try:
            res = client.models.generate_content(model=MODEL_NAME, contents=[prompt, genai.types.Part.from_bytes(data=img_d.read(), mime_type="image/png"), genai.types.Part.from_bytes(data=img_w.read(), mime_type="image/png")])
            judge = "EXECUTE" if "EXECUTE" in res.text.upper() else "WAIT"
            post_to_notion(f"{name} ({symbol})", mode, judge, res.text)
            img_d.seek(0); img_w.seek(0)
            post_to_discord(f"{name} ({symbol})", mode, judge, res.text, img_d, img_w)
            time.sleep(2)
        except: pass

if __name__ == "__main__":
    main()
