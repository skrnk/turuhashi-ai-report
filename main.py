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

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_chart_bytes(symbol, interval="1d", period="1y", n_bars=60):
    try:
        ticker = symbol.split(':')[-1]
        yf_symbol = f"{ticker}.T" if ticker.isdigit() else ticker
        data = yf.download(yf_symbol, period=period, interval=interval, progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float).dropna()
        
        # 指標計算
        data['SMA25'] = data['Close'].rolling(window=25).mean()
        data['SMA75'] = data['Close'].rolling(window=75).mean()
        data['RSI'] = calculate_rsi(data['Close'])
        
        plot_data = data.tail(n_bars).copy()
        buf = io.BytesIO()

        # 追加プロット設定 (SMAとRSI)
        ap = [
            mpf.make_addplot(plot_data['SMA25'], color='orange', width=1.5), # 凡例ラベルはコードで後述
            mpf.make_addplot(plot_data['SMA75'], color='blue', width=1.5),
            mpf.make_addplot(plot_data['RSI'], panel=2, color='purple', ylabel='RSI(14)', secondary_y=False)
        ]

        # チャート描画 (凡例と日付フォーマット)
        fig, axlist = mpf.plot(
            plot_data, type='candle', style='charles', addplot=ap,
            volume=True, returnfig=True,
            datetime_format='%y-%m-%d', # 26-03-14 形式
            tight_layout=True,
            panel_ratios=(6,2,2) # メイン, 出来高, RSI の比率
        )
        
        # 凡例の追加
        axlist[0].legend(['25MA (Org)', '75MA (Blu)'], loc='upper left', fontsize='small')
        
        fig.savefig(buf, format='png')
        plt_close = True # メモリ解放
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"⚠️ チャート失敗: {e}"); return None

# get_filtered_candidates(), post_to_notion(), post_to_discord() は前回の正常動作版を維持...
# main() 内で憲章3.5をGeminiに伝えるプロンプトに更新
