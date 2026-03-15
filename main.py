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
        "SOYBNUSD": "ZS=F", "CORNUSD": "ZC=F", "WHEATUSD": "ZW=F", "HG1!": "HG=F", "GOLD": "GC=F"
    }
    if s in mapping: return mapping[s]
    return f"{s}.T" if s.isdigit() else s

# --- 2. 憲章3.4 v2.1 判定ロジック (修正版) ---
def calculate_charter_logic(data):
    """
    RVolロジックをTradingViewの標準(元祖PineScript/Screener仕様)に修正。
    - RSI: Wilder's RMA (Matches TV)
    - RVol: 10日ではなく20日平均を採用し、当日分を母数から除外（当日/過去20日平均）
    - エラーハンドリング: 出来高0や価格変化なし時のゼロ除算を回避
    """
    if len(data) < 75: return None, "データ本数不足"
    
    close = data['Close']
    high = data['High']
    low = data['Low']
    vol = data['Volume']

    # 1. RSI (14, Wilder's RMA) - TVの標準RSIと一致
    delta = close.diff()
    avg_gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rsi_series = (100 - (100 / (1 + (avg_gain / avg_loss))))
    rsi = rsi_series.iloc[-1]

    # 2. 移動平均線 (SMA)
    sma5 = close.rolling(5).mean()
    sma25 = close.rolling(25).mean()
    sma75 = close.rolling(75).mean()

    # 3. RVol (Relative Volume) - TradingView基準への修正
    # TVスクリーナーは10日ですが、より信頼性の高い20日平均を採用。
    # 「当日の出来高 / 直前20日の平均出来高(shift1)」とすることでTVの実測値に合わせます。
    rvol_period = 20 
    avg_vol_prev = vol.shift(1).rolling(rvol_period).mean().iloc[-1]
    rvol = vol.iloc[-1] / avg_vol_prev if avg_vol_prev > 0 else 0

    # 4. Slope (SMA25)
    slope_ok = sma25.iloc[-1] > sma25.iloc[-4]

    # 5. Candle Quality (実体率/高値圏引け)
    candle_range = (high.iloc[-1] - low.iloc[-1])
    c_qual = ((close.iloc[-1] - low.iloc[-1]) / candle_range) if candle_range > 0 else 0

    # 条件判定 (Active / 順張り)
    cond_a = {
        "RSI > 50": rsi > 50, 
        "5MA > 25MA": sma5.iloc[-1] > sma25.iloc[-1], 
        "Slope OK": slope_ok, 
        "RVol > 1.2x": rvol > 1.2, 
        "Price > 75MA": close.iloc[-1] > sma75.iloc[-1], 
        "Candle > 0.7": c_qual > 0.7
    }

    # 条件判定 (Sniper / 逆張り)
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
        if data.empty: return None, 0
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        plot_data = data.tail(n_bars).copy()
        plot_data['SMA25'] = data['Close'].rolling(25).mean().tail(n_bars)
        plot_data['SMA75'] = data['Close'].rolling(75).mean().tail(n_bars)
        buf = io.BytesIO()
        ap = [mpf.make_addplot(plot_data['SMA25'], color='orange'), mpf.make_addplot(plot_data['SMA75'], color='blue')]
        mpf.plot(plot_data, type='candle', style='charles', addplot=ap, volume=True, savefig=buf, datetime_format='%y-%m-%d', tight_layout=True)
        buf.seek(0)
        return buf, data['Close'].iloc[-1]
    except: return None, 0

# --- 4. 通知処理 ---
def post_to_notion(name, strategy, judge, tier, entry_price, analysis):
    if not NOTION_TOKEN or not NOTION_DB_ID: return
    headers = {"Authorization": f"Bearer {NOTION_TOKEN}", "Content-Type": "application/json", "Notion-Version": "2022-06-28"}
    jst_now = datetime.now(JST).isoformat()
    data = {"parent": {"database_id": NOTION_DB_ID}, "properties": {"Name": {"title": [{"text": {"content": name}}]}, "Strategy": {"select": {"name": strategy}}, "Judge": {"select": {"name": judge}}, "Tier": {"select": {"name": tier}}, "Entry Price": {"number": entry_price}, "Date": {"date": {"start": jst_now}}, "Analysis": {"rich_text": [{"text": {"content": analysis[:1900]}}]}}}
    requests.post("https://api.notion.com/v1/pages", headers=headers, json=data)

def post_to_discord(name, strategy, judge, tier, analysis, img_d, img_w):
    if not DISCORD_WEBHOOK: return
    
    # 分析本文をEmbedの外に出すことで、PCで横幅いっぱいに表示させる
    # 分析内容をコードブロックで囲むことで、等幅フォントになり読みやすさも向上
    main_content = f"🚀 **{name} 【{tier}】**\n\n{analysis[:1900]}"
    
    payload = {
        "content": main_content,
        "embeds": [
            {
                "title": "テクニカルチャート監査 (日足/週足)",
                "color": 0x00ff00, 
                "image": {"url": "attachment://d.png"}
            },
            {
                "image": {"url": "attachment://w.png"}
            }
        ]
    }
    
    files = {
        "f1": ("d.png", img_d, "image/png"),
        "f2": ("w.png", img_w, "image/png")
    }
    
    requests.post(DISCORD_WEBHOOK, data={"payload_json": json.dumps(payload)}, files=files)

# --- 5. メイン実行 ---
def main():
    jst_now_str = datetime.now(JST).strftime('%Y-%m-%d %H:%M')
    print(f"🔭 憲章3.4 v2.1 監査・Tier選別開始 (JST: {jst_now_str})...")
    targets = []
    if os.path.exists('gmo_symbols.csv'):
        df_csv = pd.read_csv('gmo_symbols.csv', sep=None, engine='python')
        targets = [{'Name': str(r.get('Name', r['Symbol'])), 'Symbol': str(r['Symbol'])} for _, r in df_csv.iterrows()]
    try:
        q = Query().set_markets('japan').select('name', 'description').where(Column('close') <= 12000, Column('type').isin(['stock', 'etf'])).limit(250).get_scanner_data()
        df_scan = q[1] if isinstance(q[1], pd.DataFrame) else pd.DataFrame(q[1])
        for _, r in df_scan.iterrows(): targets.append({'Name': r.get('description', r['name']), 'Symbol': r['name']})
    except: pass
    seen, final_hits = set(), []
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
            else: print(f"  [-] {t['Name']} ({t['Symbol']}) は不適合: {details}")
        except: continue
    print(f"🏁 最終候補: {len(final_hits)} 銘柄。AI深度分析を開始...")
    for hit in final_hits:
        try:
            name, symbol, mode = hit['Name'], hit['Symbol'], hit['Mode']
            img_d_tuple = create_chart_bytes(symbol, "1d")
            img_w_tuple = create_chart_bytes(symbol, "1wk", 40)
            img_d, entry_price, img_w = img_d_tuple[0], img_d_tuple[1], img_w_tuple[0]
            if not img_d or not img_w: continue
            chk_str = "\n".join([f"{'●' if v else '×'} {k}" for k, v in hit['Details'].items()])
            curr_time_jst = datetime.now(JST).strftime('%Y-%m-%d %H:%M')
            
            # AIへのプロンプトにSymbol項目を追加
            prompt = f"銘柄: {name} ({symbol})\n戦略モード: {mode}\n【システムチェック値】\n{chk_str}\n\n【分析指令】\n1. ビジネスモデル解析: 業界トレンドと優位性を述べよ。\n2. テクニカル検証: 数値が視覚的にも本物のトレンドか。\n3. 総合評価: Tier S, Tier A, Tier B以下を決定せよ。\n※コモディティは構造的トレンド明白な場合のみTier Sとせよ。\n\n■Symbol: {symbol}\n■報告日次: {curr_time_jst}\n■結論: [Tier S / Tier A / Tier B]\n■売買執行: [EXECUTE または WAIT]\n---\n【憲章チェック】\n{chk_str}\n【業界分析】\n(記述)\n【総合分析】\n(記述)"
            
            response = client.models.generate_content(model=MODEL_NAME, contents=[prompt, genai.types.Part.from_bytes(data=img_d.read(), mime_type="image/png"), genai.types.Part.from_bytes(data=img_w.read(), mime_type="image/png")])
            res_text = response.text
            tier = "Tier B"
            if "Tier S" in res_text.split('---')[0]: tier = "Tier S"
            elif "Tier A" in res_text.split('---')[0]: tier = "Tier A"
            if tier in ["Tier S", "Tier A"]:
                judge = "EXECUTE" if "EXECUTE" in res_text.upper() else "WAIT"
                post_to_notion(f"{name} ({symbol})", mode, judge, tier, entry_price, res_text)
                print(f"📌 Notion保存[{tier}]: {name}")
                if tier == "Tier S":
                    img_d.seek(0); img_w.seek(0)
                    post_to_discord(name, mode, judge, tier, res_text, img_d, img_w)
                    print(f"✅ Discord通知完了: {name}")
            else: print(f"  [x] {name} は {tier} のため通知スキップ。")
            time.sleep(2)
        except Exception as e: print(f"⚠️ 分析エラー: {e}")

if __name__ == "__main__": main()
