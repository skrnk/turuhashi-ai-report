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

# --- 1. 翻訳辞書：Yahoo Finance 互換形式への変換 ---
def get_yf_symbol(symbol):
    ticker = str(symbol).split(':')[-1].strip() # NASDAQ:AAPL -> AAPL
    mapping = {
        "NI225": "^N225", "DJI": "^DJI", "SPX": "^GSPC", "NAS100": "^NDX",
        "RTY": "^RUT", "UK100": "^FTSE", "GER40": "^GDAXI", "FRA40": "^FCHI",
        "HSI": "^HSI", "XIN9": "000001.SS", "NIFTY": "^NSEI", "GOLD": "GC=F",
        "XAUUSD": "GC=F", "XAGUSD": "SI=F", "XPTUSD": "PL=F", "XPDUSD": "PA=F",
        "USOIL": "CL=F", "UKOIL": "BZ=F", "NATGAS": "NG=F", "VIX": "^VIX",
        "USDJPY": "JPY=X"
    }
    if ticker in mapping: return mapping[ticker]
    return f"{ticker}.T" if ticker.isdigit() else ticker

# --- 2. 憲章3.4 v2.1 ロジック (原本完全同期) ---
def calculate_charter_logic(data):
    if len(data) < 75: return None
    close, high, low, vol = data['Close'], data['High'], data['Low'], data['Volume']
    
    # RSI (Wilder法 / RMA)
    delta = close.diff()
    avg_gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
    
    # 指標
    sma5, sma25, sma75 = close.rolling(5).mean(), close.rolling(25).mean(), close.rolling(75).mean()
    rvol = vol / vol.rolling(10).mean()
    slope_ok = sma25.iloc[-1] > sma25.iloc[-4]
    f_candle = ((close - low) / (high - low)).iloc[-1] > 0.7
    
    # Active判定
    is_active = (rsi.iloc[-1] > 50) and (sma5.iloc[-1] > sma25.iloc[-1]) and slope_ok and (rvol.iloc[-1] > 1.2) and (close.iloc[-1] > sma75.iloc[-1]) and f_candle
    # Sniper判定 (5MAクロスオーバー)
    cross_over = (close.iloc[-1] > sma5.iloc[-1]) and (close.iloc[-2] <= sma5.iloc[-2])
    is_sniper = (rsi.iloc[-1] < 40) and (close.iloc[-1] < sma25.iloc[-1] * 0.90) and cross_over and f_candle
    
    if is_active: return "Active (順張り)"
    if is_sniper: return "Sniper (逆張り)"
    return None

def create_chart_bytes(symbol, interval="1d", n_bars=70):
    try:
        yf_symbol = get_yf_symbol(symbol)
        data = yf.download(yf_symbol, period="2y", interval=interval, progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data['SMA25'] = data['Close'].rolling(25).mean()
        data['SMA75'] = data['Close'].rolling(75).mean()
        buf = io.BytesIO()
        ap = [mpf.make_addplot(data.tail(n_bars)['SMA25'], color='orange'), mpf.make_addplot(data.tail(n_bars)['SMA75'], color='blue')]
        mpf.plot(data.tail(n_bars), type='candle', style='charles', addplot=ap, volume=True, savefig=buf, datetime_format='%y-%m-%d')
        buf.seek(0)
        return buf
    except: return None

# --- 通知・保存関数 ---
def post_to_notion(name, strategy, judge, analysis):
    if not NOTION_TOKEN or not NOTION_DB_ID: return
    requests.post("https://api.notion.com/v1/pages", headers={"Authorization": f"Bearer {NOTION_TOKEN}", "Content-Type": "application/json", "Notion-Version": "2022-06-28"}, json={"parent": {"database_id": NOTION_DB_ID}, "properties": {"Name": {"title": [{"text": {"content": name}}]}, "Strategy": {"select": {"name": strategy}}, "Judge": {"select": {"name": judge}}, "Date": {"date": {"start": datetime.now().isoformat()}}, "Analysis": {"rich_text": [{"text": {"content": analysis[:1900]}}]}}})

def post_to_discord(name, strategy, judge, analysis, img_d, img_w):
    if not DISCORD_WEBHOOK: return
    color = 0x00ff00 if judge == "EXECUTE" else 0x808080
    payload = {"embeds": [{"title": f"🚀 【{strategy}】: {name}", "description": analysis[:1800], "color": color, "image": {"url": "attachment://d.png"}}, {"image": {"url": "attachment://w.png"}}]}
    requests.post(DISCORD_WEBHOOK, data={"payload_json": json.dumps(payload)}, files={"f1": ("d.png", img_d, "image/png"), "f2": ("w.png", img_w, "image/png")})

# --- メインロジック ---
def main():
    print("🔭 憲章3.4 v2.1 全域ハイブリッド・スキャン開始...")
    targets = []
    # 1. CSV読み込み
    if os.path.exists('gmo_symbols.csv'):
        csv_df = pd.read_csv('gmo_symbols.csv', sep=None, engine='python')
        targets = [{'Name': r.get('Name', r['Symbol']), 'Symbol': r['Symbol']} for _, r in csv_df.iterrows()]
    
    # 2. 東証スキャナー (文法修正済)
    try:
        q = (Query().set_markets('japan').select('name', 'description', 'close')
             .where(Column('close') <= 12000, Column('type').isin(['stock', 'etf']))
             .limit(100).get_scanner_data())
        targets += [{'Name': r['description'], 'Symbol': r['name']} for r in q[1]]
    except Exception as e: print(f"⚠️ スキャナーエラー: {e}")

    # 重複排除
    seen = set()
    unique_targets = []
    for t in targets:
        if t['Symbol'] not in seen:
            unique_targets.append(t); seen.add(t['Symbol'])

    print(f"📊 検証母数: {len(unique_targets)} 銘柄。間引き開始...")

    final_hits = []
    for item in unique_targets:
        try:
            yf_symbol = get_yf_symbol(item['Symbol'])
            data = yf.download(yf_symbol, period="180d", progress=False)
            if data.empty: continue
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            
            mode = calculate_charter_logic(data)
            if mode:
                item['Mode'] = mode
                final_hits.append(item)
                print(f"🎯 適合: {item['Name']} ({mode})")
        except: continue

    print(f"🏁 最終候補: {len(final_hits)} 銘柄。AI分析実行...")
    for hit in final_hits[:20]: # 負荷考慮
        try:
            name, symbol, mode = hit['Name'], hit['Symbol'], hit['Mode']
            img_d, img_w = create_chart_bytes(symbol, "1d"), create_chart_bytes(symbol, "1wk", 40)
            if not img_d or not img_w: continue
            
            prompt = f"銘柄: {name} ({symbol})\n戦略: {mode}\n憲章3.4 v2.1に基づき分析せよ。最後に EXECUTE または WAIT を示せ。"
            response = client.models.generate_content(model=MODEL_NAME, contents=[prompt, genai.types.Part.from_bytes(data=img_d.read(), mime_type="image/png"), genai.types.Part.from_bytes(data=img_w.read(), mime_type="image/png")])
            
            judge = "EXECUTE" if "EXECUTE" in response.text.upper() else "WAIT"
            post_to_notion(f"{name} ({symbol})", mode, judge, response.text)
            img_d.seek(0); img_w.seek(0)
            post_to_discord(f"{name} ({symbol})", mode, judge, response.text, img_d, img_w)
            print(f"✅ 通知完了: {name}")
            time.sleep(2)
        except Exception as e: print(f"⚠️ 分析エラー({hit['Symbol']}): {e}")

if __name__ == "__main__":
    main()
