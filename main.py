import os
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from tradingview_screener import Query, Column
from google import genai
import io
import time
import requests
from datetime import datetime

# --- 初期設定 ---
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-3-flash-preview"
NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
NOTION_DB_ID = os.environ.get("NOTION_DATABASE_ID")
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL")

def post_to_notion(symbol, strategy, judge, analysis):
    """分析結果をNotionデータベースにテキスト保存"""
    if not NOTION_TOKEN or not NOTION_DB_ID: return
    url = "https://api.notion.com/v1/pages"
    headers = {"Authorization": f"Bearer {NOTION_TOKEN}", "Content-Type": "application/json", "Notion-Version": "2022-06-28"}
    summary = (analysis[:1900] + '...') if len(analysis) > 1900 else analysis
    data = {
        "parent": {"database_id": NOTION_DB_ID},
        "properties": {
            "Name": {"title": [{"text": {"content": f"{symbol} ({datetime.now().strftime('%m/%d')})"}}]},
            "Strategy": {"select": {"name": strategy}},
            "Judge": {"select": {"name": judge}},
            "Date": {"date": {"start": datetime.now().isoformat()}},
            "Analysis": {"rich_text": [{"text": {"content": summary}}]}
        }
    }
    requests.post(url, headers=headers, json=data)

def post_to_discord(symbol, strategy, judge, analysis, image_buf):
    """分析結果とチャート画像をDiscordにプッシュ通知"""
    if not DISCORD_WEBHOOK: return
    
    color = 0x00ff00 if judge == "EXECUTE" else 0x808080 # 緑 or 灰
    payload = {
        "embeds": [{
            "title": f"🚀 【{strategy}】判定: {symbol}",
            "description": analysis[:2000],
            "color": color,
            "fields": [
                {"name": "最終判断", "value": f"**{judge}**", "inline": True},
                {"name": "分析日時", "value": datetime.now().strftime('%Y-%m-%d %H:%M'), "inline": True}
            ],
            "image": {"url": "attachment://chart.png"}
        }]
    }
    
    files = {"file": ("chart.png", image_buf, "image/png")}
    try:
        requests.post(DISCORD_WEBHOOK, data={"payload_json": pd.io.json.dumps(payload)}, files=files)
        print(f"📡 Discord通知完了: {symbol}")
    except Exception as e:
        print(f"⚠️ Discord送信失敗: {e}")

def get_filtered_candidates():
    print("🔭 スキャニング開始...")
    try:
        q = (Query().set_markets('japan').select('name', 'description', 'close', 'RSI', 'SMA5', 'SMA20', 'SMA50', 'high', 'low')
             .where(Column('type').isin(['stock', 'etf'])).limit(100).get_scanner_data())
        df = pd.DataFrame(q[1])
        if df.empty: return df
        df['candle_q'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['dist_25'] = (df['close'] / df['SMA20'] - 1) * 100
        active_df = df[(df['RSI'] > 53) & (df['SMA5'] > df['SMA20']) & (df['candle_q'] > 0.8)].copy()
        active_df['strategy'] = 'Active (順張り)'
        sniper_df = df[(df['dist_25'] < -12) & (df['candle_q'] > 0.65)].copy()
        sniper_df['strategy'] = 'Sniper (逆張り)'
        return pd.concat([active_df, sniper_df]).drop_duplicates(subset=['name']).head(5)
    except Exception as e:
        print(f"❌ スキャンエラー: {e}")
        return pd.DataFrame()

def create_chart_bytes(symbol):
    try:
        ticker = symbol.split(':')[-1]
        yf_symbol = f"{ticker}.T" if ticker.isdigit() else ticker
        mapping = {"NI225": "^N225", "DJI": "^DJI", "SPX": "^GSPC"}
        yf_symbol = mapping.get(ticker, yf_symbol)
        data = yf.download(yf_symbol, period="1y", interval="1d", progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float).dropna()
        data['SMA25'] = data['Close'].rolling(window=25).mean()
        data['SMA75'] = data['Close'].rolling(window=75).mean()
        plot_data = data.tail(60).copy()
        buf = io.BytesIO()
        ap = [mpf.make_addplot(plot_data['SMA25'], color='orange'), mpf.make_addplot(plot_data['SMA75'], color='blue')]
        mpf.plot(plot_data, type='candle', style='charles', addplot=ap, savefig=buf, volume=True)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"⚠️ チャート失敗: {e}"); return None

def main():
    final_candidates = get_filtered_candidates()
    if final_candidates.empty:
        print("📭 候補なし"); return

    for _, row in final_candidates.iterrows():
        name = row['name']
        strategy = row['strategy']
        print(f"🚀 分析中: {name} ({strategy})")
        
        chart_img = create_chart_bytes(name)
        prompt = f"""
        回答冒頭に必ず 『【分析タイプ: {strategy}】』 と記載せよ。
        銘柄: {row.get('description', name)} ({name})
        判定基準: 運用憲章3.4
        チャート（橙:25MA, 青:75MA）を分析し、最後に EXECUTE または WAIT を示せ。
        """
        
        try:
            contents = [prompt]
            if chart_img:
                img_data = chart_img.read()
                contents.append(genai.types.Part.from_bytes(data=img_data, mime_type="image/png"))
                chart_img.seek(0) # Discord用に戻す
            
            response = client.models.generate_content(model=MODEL_NAME, contents=contents)
            res_text = response.text
            judge = "EXECUTE" if "EXECUTE" in res_text.upper() else "WAIT"
            
            # --- Notion保存 & Discord通知 ---
            post_to_notion(name, strategy, judge, res_text)
            if chart_img:
                post_to_discord(name, strategy, judge, res_text, chart_img)
            
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ 分析エラー: {e}")

if __name__ == "__main__":
    main()
