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

# --- 1. シンボル翻訳辞書 ---
def get_yf_symbol(symbol):
    s = str(symbol).split(':')[-1].strip()
    mapping = {
        "NI225": "^N225", "DJI": "^DJI", "SPX": "^GSPC", "NAS100": "^NDX",
        "RTY": "^RUT", "UK100": "^FTSE", "GER40": "^GDAXI", "EUSTX50": "^STOXX50E",
        "FRA40": "^FCHI", "HSI": "^HSI", "XIN9": "000001.SS", "NIFTY": "^NSEI",
        "XAUUSD": "GC=F", "XAGUSD": "SI=F", "XPTUSD": "PL=F", "XPDUSD": "PA=F",
        "USOIL": "CL=F", "UKOIL": "BZ=F", "NATGAS": "NG=F", "VIX": "^VIX",
        "USDJPY": "JPY=X", "大豆": "ZS=F", "コーン": "ZC=F", "小麦": "ZW=F",
        "SOYBNUSD": "ZS=F", "CORNUSD": "ZC=F", "WHEATUSD": "ZW=F", "HG1!": "HG=F",
        "GOLD": "GC=F"
    }
    if s in mapping: return mapping[s]
    return f"{s}.T" if s.isdigit() else s

# --- 2. 憲章3.4 v2.1 判定ロジック ---
def calculate_charter_logic(data):
    if len(data) < 75: return None, "データ本数不足"
    close, high, low, vol = data['Close'], data['High'], data['Low'], data['Volume']
    
    delta = close.diff()
    avg_gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rsi = (100 - (100 / (1 + (avg_gain / avg_loss)))).iloc[-1]
    
    sma5, sma25, sma75 = close.rolling(5).mean(), close.rolling(25).mean(), close.rolling(75).mean()
    rvol = (vol / vol.rolling(10).mean()).iloc[-1]
    slope_ok = sma25.iloc[-1] > sma25.iloc[-4]
    c_qual = ((close - low) / (high - low)).iloc[-1]
    
    cond_a = {
        "RSI > 50": rsi > 50,
        "5MA > 25MA": sma5.iloc[-1] > sma25.iloc[-1],
        "Slope OK": slope_ok,
        "RVol > 1.2x": rvol > 1.2,
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

    if all(cond_a.values()): return "Active (順張り)", cond_a
    if all(cond_s.values()): return "Sniper (逆張り)", cond_s
    
    fails = [k for k, v in {**cond_a, **cond_s}.items() if not v]
    return None, f"不適合: {', '.join(fails[:3])}"

# --- 3. チャート生成 ---
def create_chart_bytes(symbol, interval="1d", n_bars=70):
    try:
        yf_symbol = get_yf_symbol(symbol)
        data = yf.download(yf_symbol, period="2y", interval=interval, progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        plot_data = data.tail(n_bars).copy()
        plot_data['SMA25'] = data['Close'].rolling(25).mean().tail(n_bars)
        plot_data['SMA75'] = data['Close'].rolling(75).mean().tail(n_bars)
        buf = io.BytesIO()
        ap = [mpf.make_addplot(plot_data['SMA25'], color='orange'), mpf.make_addplot(plot_data['SMA75'], color='blue')]
        mpf.plot(plot_data, type='candle', style='charles', addplot=ap, volume=True, savefig=buf, datetime_format='%y-%m-%d', tight_layout=True)
        buf.seek(0)
        return buf
    except: return None

# --- 4. 通知処理 ---
def post_to_notion(name, strategy, judge, analysis):
    if not NOTION_TOKEN or not NOTION_DB_ID: return
    headers = {"Authorization": f"Bearer {NOTION_TOKEN}", "Content-Type": "application/json", "Notion-Version": "2022-06-28"}
    requests.post("https://api.notion.com/v1/pages", headers=headers, json={"parent": {"database_id": NOTION_DB_ID}, "properties": {"Name": {"title": [{"text": {"content": name}}]}, "Strategy": {"select": {"name": strategy}}, "Judge": {"select": {"name": judge}}, "Date": {"date": {"start": datetime.now().isoformat()}}, "Analysis": {"rich_text": [{"text": {"content": analysis[:1900]}}]}}})

def post_to_discord(name, strategy, judge, analysis, img_d, img_w):
    if not DISCORD_WEBHOOK: return
    color = 0x00ff00 if judge == "EXECUTE" else 0x808080
    payload = {"embeds": [{"title": f"🚀 {name}", "description": analysis[:1800], "color": color, "image": {"url": "attachment://d.png"}}, {"image": {"url": "attachment://w.png"}}]}
    requests.post(DISCORD_WEBHOOK, data={"payload_json": json.dumps(payload)}, files={"f1": ("d.png", img_d, "image/png"), "f2": ("w.png", img_w, "image/png")})

# --- 5. メイン実行 ---
def main():
    print("🔭 憲章3.4 v2.1 監査モード開始...")
    targets = []
    if os.path.exists('gmo_symbols.csv'):
        df_csv = pd.read_csv('gmo_symbols.csv', sep=None, engine='python')
        targets = [{'Name': str(r.get('Name', r['Symbol'])), 'Symbol': str(r['Symbol'])} for _, r in df_csv.iterrows()]
    
    try:
        q = Query().set_markets('japan').select('name', 'description').where(Column('close') <= 12000, Column('type').isin(['stock', 'etf'])).limit(250).get_scanner_data()
        df_scan = q[1] if isinstance(q[1], pd.DataFrame) else pd.DataFrame(q[1])
        for _, r in df_scan.iterrows():
            targets.append({'Name': r.get('description', r['name']), 'Symbol': r['name']})
    except: pass

    seen = set()
    final_hits = []
    for t in targets:
        if t['Symbol'] in seen: continue
        seen.add(t['Symbol'])
        try:
            data = yf.download(get_yf_symbol(t['Symbol']), period="150d", progress=False)
            if data.empty: continue
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            mode, details = calculate_charter_logic(data)
            if mode:
                t['Mode'], t['Details'] = mode, details
                final_hits.append(t)
                print(f"🎯 適合発見: {t['Name']} ({mode})")
            else:
                print(f"  [-] {t['Name']} ({t['Symbol']}) は不適合: {details}")
        except: continue

    print(f"🏁 最終候補: {len(final_hits)} 銘柄。AI分析を開始...")

    for hit in final_hits:
        try:
            name, symbol, mode = hit['Name'], hit['Symbol'], hit['Mode']
            img_d = create_chart_bytes(symbol, "1d"), create_chart_bytes(symbol, "1wk", 40)
            if not img_d[0] or not img_d[1]: continue
            
            # チェックリストの構築 (●×形式)
            chk_str = "\n".join([f"{'●' if v else '×'} {k}" for k, v in hit['Details'].items()])
            curr_time = datetime.now().strftime('%Y-%m-%d %H:%M')
            
            prompt = f"""
            分析依頼: {name} ({symbol})

            【システムによる憲章規準チェック】
            {chk_str}

            上記を踏まえ、添付の日足・週足チャートを分析せよ。
            回答は必ず以下のフォーマットを厳守すること：

            ■報告日次: {curr_time}
            ■結論: [EXECUTE または WAIT]
            ■判定モード: {mode}
            ---
            【憲章規準チェック結果】
            {chk_str}

            【分析コメント】
            (ここに詳細な分析を記述)
            """
            
            response = client.models.generate_content(model=MODEL_NAME, contents=[prompt, genai.types.Part.from_bytes(data=img_d[0].read(), mime_type="image/png"), genai.types.Part.from_bytes(data=img_d[1].read(), mime_type="image/png")])
            res_text = response.text
            judge = "EXECUTE" if "EXECUTE" in res_text.upper() else "WAIT"
            
            post_to_notion(f"{name} ({symbol})", mode, judge, res_text)
            img_d[0].seek(0); img_d[1].seek(0)
            post_to_discord(f"{name} ({symbol})", mode, judge, res_text, img_d[0], img_d[1])
            print(f"✅ 通知完了: {name}")
            time.sleep(2)
        except Exception as e: print(f"⚠️ 分析エラー: {e}")

if __name__ == "__main__":
    main()
