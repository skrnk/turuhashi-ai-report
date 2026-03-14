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

# --- 1. 翻訳辞書：Yahoo Finance 形式への完全対応 ---
def get_yf_symbol(symbol):
    s = str(symbol).split(':')[-1].strip() # NASDAQ:AAPL -> AAPL
    mapping = {
        "NI225": "^N225", "DJI": "^DJI", "SPX": "^GSPC", "NAS100": "^NDX",
        "RTY": "^RUT", "UK100": "^FTSE", "GER40": "^GDAXI", "EUSTX50": "^STOXX50E",
        "FRA40": "^FCHI", "HSI": "^HSI", "XIN9": "000001.SS", "NIFTY": "^NSEI",
        "XAUUSD": "GC=F", "XAGUSD": "SI=F", "XPTUSD": "PL=F", "XPDUSD": "PA=F",
        "USOIL": "CL=F", "UKOIL": "BZ=F", "NATGAS": "NG=F", "VIX": "^VIX",
        "SOYBNUSD": "ZS=F", "CORNUSD": "ZC=F", "WHEATUSD": "ZW=F", "HG1!": "HG=F",
        "BRK.B": "BRK-B" # Yahooはドットではなくハイフン
    }
    if s in mapping: return mapping[s]
    return f"{s}.T" if s.isdigit() else s

# --- 2. 憲章3.4 v2.1 ロジック ---
def calculate_charter_logic(data, name):
    if len(data) < 75: return None, "データ不足(75本未満)"
    close, high, low, vol = data['Close'], data['High'], data['Low'], data['Volume']
    
    # RSI (RMA)
    delta = close.diff()
    avg_gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rsi_val = (100 - (100 / (1 + (avg_gain / avg_loss)))).iloc[-1]
    
    # 指標
    sma5, sma25, sma75 = close.rolling(5).mean(), close.rolling(25).mean(), close.rolling(75).mean()
    rvol = (vol / vol.rolling(10).mean()).iloc[-1]
    slope_ok = sma25.iloc[-1] > sma25.iloc[-4]
    c_qual = ((close - low) / (high - low)).iloc[-1]
    
    # Active判定
    cond_a = [rsi_val > 50, sma5.iloc[-1] > sma25.iloc[-1], slope_ok, rvol > 1.2, close.iloc[-1] > sma75.iloc[-1], c_qual > 0.7]
    if all(cond_a): return "Active (順張り)", "OK"
    
    # Sniper判定
    cross_over = (close.iloc[-1] > sma5.iloc[-1]) and (close.iloc[-2] <= sma5.iloc[-2])
    cond_s = [rsi_val < 40, close.iloc[-1] < sma25.iloc[-1] * 0.90, cross_over, c_qual > 0.7]
    if all(cond_s): return "Sniper (逆張り)", "OK"
    
    # なぜダメだったかの理由（デバッグ用）
    reasons = []
    if not slope_ok: reasons.append("25MA傾きNG")
    if rsi_val <= 50 and rsi_val >= 40: reasons.append(f"RSI中立({rsi_val:.1f})")
    if c_qual <= 0.7: reasons.append(f"ヒゲ/品質NG({c_qual:.2f})")
    return None, ", ".join(reasons)

def create_chart_bytes(symbol, interval="1d", n_bars=70):
    try:
        yf_symbol = get_yf_symbol(symbol)
        data = yf.download(yf_symbol, period="1y", interval=interval, progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        buf = io.BytesIO()
        ap = [mpf.make_addplot(data.tail(n_bars)['Close'].rolling(25).mean(), color='orange')]
        mpf.plot(data.tail(n_bars), type='candle', style='charles', addplot=ap, savefig=buf)
        buf.seek(0)
        return buf
    except: return None

# --- 保存・通知関数 (以前の正常版を維持) ---
def post_to_notion(name, strategy, judge, analysis):
    if not NOTION_TOKEN or not NOTION_DB_ID: return
    try:
        requests.post("https://api.notion.com/v1/pages", headers={"Authorization": f"Bearer {NOTION_TOKEN}", "Content-Type": "application/json", "Notion-Version": "2022-06-28"}, json={"parent": {"database_id": NOTION_DB_ID}, "properties": {"Name": {"title": [{"text": {"content": name}}]}, "Strategy": {"select": {"name": strategy}}, "Judge": {"select": {"name": judge}}, "Date": {"date": {"start": datetime.now().isoformat()}}, "Analysis": {"rich_text": [{"text": {"content": analysis[:1900]}}]}}})
    except: pass

def post_to_discord(name, strategy, judge, analysis, img_d, img_w):
    if not DISCORD_WEBHOOK: return
    color = 0x00ff00 if judge == "EXECUTE" else 0x808080
    payload = {"embeds": [{"title": f"🚀 【{strategy}】: {name}", "description": analysis[:1800], "color": color, "image": {"url": "attachment://d.png"}}, {"image": {"url": "attachment://w.png"}}]}
    requests.post(DISCORD_WEBHOOK, data={"payload_json": json.dumps(payload)}, files={"f1": ("d.png", img_d, "image/png"), "f2": ("w.png", img_w, "image/png")})

# --- メインロジック ---
def main():
    print("🔭 憲章3.4 v2.1 監査モード・スキャン開始...")
    targets = []
    
    # 1. CSV読み込み
    if os.path.exists('gmo_symbols.csv'):
        csv_df = pd.read_csv('gmo_symbols.csv', sep=None, engine='python')
        csv_df.columns = csv_df.columns.str.strip()
        targets = [{'Name': str(r.get('Name', r['Symbol'])), 'Symbol': str(r['Symbol'])} for _, r in csv_df.iterrows()]
    
    # 2. 東証スキャナー (修正版)
    try:
        q = Query().set_markets('japan').select('name', 'description').where(Column('close') <= 12000, Column('type').isin(['stock', 'etf'])).limit(100).get_scanner_data()
        # q[1] が DataFrame の場合とリストの場合があるため安全に処理
        df_scan = q[1] if isinstance(q[1], pd.DataFrame) else pd.DataFrame(q[1])
        for _, r in df_scan.iterrows():
            targets.append({'Name': r.get('description', r['name']), 'Symbol': r['name']})
    except Exception as e: print(f"⚠️ スキャナー警告: {e}")

    seen = set()
    final_hits = []
    print(f"📊 検証母数: {len(targets)} 銘柄。憲章フィルター適用...")

    for item in targets:
        if item['Symbol'] in seen: continue
        seen.add(item['Symbol'])
        
        yf_symbol = get_yf_symbol(item['Symbol'])
        try:
            data = yf.download(yf_symbol, period="180d", progress=False)
            if data.empty:
                # print(f"  [-] {item['Symbol']}: データなし") # ログが汚れるので抑制
                continue
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            
            mode, reason = calculate_charter_logic(data, item['Name'])
            if mode:
                item['Mode'] = mode
                final_hits.append(item)
                print(f"🎯 適合発見: {item['Name']} ({mode})")
            else:
                pass # 静かにスキップ。見たい場合は print(f"  [x] {item['Name']}: {reason}")
        except: continue

    print(f"🏁 最終候補: {len(final_hits)} 銘柄。")
    for hit in final_hits[:20]:
        try:
            img_d, img_w = create_chart_bytes(hit['Symbol'], "1d"), create_chart_bytes(hit['Symbol'], "1wk", 40)
            if not img_d or not img_w: continue
            
            prompt = f"銘柄: {hit['Name']} ({hit['Symbol']})\n戦略: {hit['Mode']}\n憲章3.4 v2.1に基づき分析せよ。最後に EXECUTE または WAIT を示せ。"
            response = client.models.generate_content(model=MODEL_NAME, contents=[prompt, genai.types.Part.from_bytes(data=img_d.read(), mime_type="image/png"), genai.types.Part.from_bytes(data=img_w.read(), mime_type="image/png")])
            
            judge = "EXECUTE" if "EXECUTE" in response.text.upper() else "WAIT"
            post_to_notion(f"{hit['Name']} ({hit['Symbol']})", hit['Mode'], judge, response.text)
            img_d.seek(0); img_w.seek(0)
            post_to_discord(hit['Name'], hit['Mode'], judge, response.text, img_d, img_w)
            print(f"✅ 通知完了: {hit['Name']}")
            time.sleep(2)
        except Exception as e: print(f"⚠️ 分析エラー: {e}")

if __name__ == "__main__":
    main()
