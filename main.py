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

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_gmo_list():
    """gmo_symbols.csvから監視銘柄を取得"""
    file_name = 'gmo_symbols.csv'
    if os.path.exists(file_name):
        try:
            df = pd.read_csv(file_name)
            df = df.drop_duplicates(subset=['Symbol'])
            return df.to_dict('records')
        except Exception as e:
            print(f"⚠️ CSV読み込みエラー: {e}")
    return []

def get_yf_symbol(ticker):
    """yfinance形式変換 (GMO指数のマッピング)"""
    mapping = {
        "NI225": "^N225", "DJI": "^DJI", "SPX": "^GSPC", 
        "NAS100": "^IXIC", "GOLD": "GC=F", "OIL": "CL=F", "USDJPY": "JPY=X"
    }
    if ticker in mapping: return mapping[ticker]
    return f"{ticker}.T" if ticker.isdigit() else ticker

def create_chart_bytes(symbol, interval="1d", n_bars=70):
    """改良版チャート: RSI14, 25/75SMA凡例, yy-mm-dd形式"""
    try:
        yf_symbol = get_yf_symbol(symbol)
        data = yf.download(yf_symbol, period="2y", interval=interval, progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float).dropna()
        data['SMA25'] = data['Close'].rolling(25).mean()
        data['SMA75'] = data['Close'].rolling(75).mean()
        data['RSI'] = calculate_rsi(data['Close'])
        
        plot_data = data.tail(n_bars).copy()
        buf = io.BytesIO()
        
        ap = [
            mpf.make_addplot(plot_data['SMA25'], color='orange', width=1.5),
            mpf.make_addplot(plot_data['SMA75'], color='blue', width=1.5),
            mpf.make_addplot(plot_data['RSI'], panel=2, color='purple', ylabel='RSI')
        ]
        
        fig, axlist = mpf.plot(
            plot_data, type='candle', style='charles', addplot=ap,
            volume=True, returnfig=True, datetime_format='%y-%m-%d',
            tight_layout=True, panel_ratios=(6,2,2)
        )
        axlist[0].legend(['SMA25 (Org)', 'SMA75 (Blu)'], loc='upper left', fontsize='8')
        
        fig.savefig(buf, format='png')
        buf.seek(0)
        return buf
    except: return None

def post_to_notion(display_name, strategy, judge, analysis):
    if not NOTION_TOKEN or not NOTION_DB_ID: return
    headers = {"Authorization": f"Bearer {NOTION_TOKEN}", "Content-Type": "application/json", "Notion-Version": "2022-06-28"}
    summary = (analysis[:1900] + '...') if len(analysis) > 1900 else analysis
    data = {
        "parent": {"database_id": NOTION_DB_ID},
        "properties": {
            "Name": {"title": [{"text": {"content": display_name}}]},
            "Strategy": {"select": {"name": strategy}},
            "Judge": {"select": {"name": judge}},
            "Date": {"date": {"start": datetime.now().isoformat()}},
            "Analysis": {"rich_text": [{"text": {"content": summary}}]}
        }
    }
    requests.post("https://api.notion.com/v1/pages", headers=headers, json=data)

def post_to_discord(display_name, strategy, judge, analysis, img_daily, img_weekly):
    if not DISCORD_WEBHOOK: return
    color = 0x00ff00 if judge == "EXECUTE" else 0x808080
    payload = {
        "embeds": [
            {
                "title": f"🚀 【{strategy}】判定: {display_name}",
                "description": analysis[:1800],
                "color": color,
                "fields": [{"name": "最終判断", "value": f"**{judge}**", "inline": True}],
                "image": {"url": "attachment://daily.png"}
            },
            {"image": {"url": "attachment://weekly.png"}}
        ]
    }
    files = {"file1": ("daily.png", img_daily, "image/png"), "file2": ("weekly.png", img_weekly, "image/png")}
    requests.post(DISCORD_WEBHOOK, data={"payload_json": json.dumps(payload)}, files=files)

def main():
    print("🔭 憲章3.5 ハイブリッド・スキャン開始...")
    
    # 1. 東証スキャン (12,000円制限)
    try:
        q = (Query().set_markets('japan').select('name', 'description', 'close', 'RSI')
             .where(Column('close').le(12000), Column('type').isin(['stock', 'etf'])).limit(40).get_scanner_data())
        df_tse = pd.DataFrame(q[1])
        scan_candidates = [{'Name': r['description'], 'Symbol': r['name'], 'Category': 'TSE_Stock'} for _, r in df_tse.iterrows()]
    except: scan_candidates = []

    # 2. GMOリスト (gmo_symbols.csv) 読み込み
    gmo_list = get_gmo_list()
    
    # 3. リスト統合
    all_targets = gmo_list + scan_candidates
    seen = set()
    final_targets = []
    for t in all_targets:
        if t['Symbol'] not in seen:
            final_targets.append(t); seen.add(t['Symbol'])

    print(f"🎯 分析対象: {len(final_targets)} 銘柄をセットしました。")

    for target in final_targets[:10]: # API負荷分散のため上位10件
        symbol = target['Symbol']
        display_name = f"{target['Name']} ({symbol})"
        category = target['Category']
        print(f"🚀 精密検証中: {display_name}")
        
        img_daily = create_chart_bytes(symbol, interval="1d")
        img_weekly = create_chart_bytes(symbol, interval="1wk", n_bars=40)
        
        if not img_daily: continue

        prompt = f"""
        回答冒頭に必ず 『【分析タイプ: {category}】』 と記載せよ。
        銘柄: {display_name}
        
        【憲章3.5 厳格チェック項目】
        提供された2枚のチャート（1枚目:日足, 2枚目:週足）から以下に言及せよ：
        1. RSI14(下段)の数値と勢い
        2. SMA25(橙)/75(青)の序列・傾き（PineScript基準：三位一体か）
        3. 週足チャートから見た長期トレンド整合性
        4. キャンドルの形状（品質）から見たエントリー優位性

        最後に、EXECUTE または WAIT を理由と共に示せ。
        """
        
        try:
            response = client.models.generate_content(
                model=MODEL_NAME, 
                contents=[prompt, genai.types.Part.from_bytes(data=img_daily.read(), mime_type="image/png"),
                          genai.types.Part.from_bytes(data=img_weekly.read(), mime_type="image/png")]
            )
            res_text = response.text
            judge = "EXECUTE" if "EXECUTE" in res_text.upper() else "WAIT"
            
            # Notion保存とDiscord通知
            post_to_notion(display_name, category, judge, res_text)
            img_daily.seek(0); img_weekly.seek(0)
            post_to_discord(display_name, category, judge, res_text, img_daily, img_weekly)
            
            print(f"🤖 {symbol}: 判断完了")
            time.sleep(2)
        except Exception as e: print(f"⚠️ エラー({symbol}): {e}")

if __name__ == "__main__":
    main()
