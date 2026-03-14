import os
import pandas as pd
from tradingview_screener import Query, Column
from google import genai
import time

# --- 1. 初期設定 (最新SDK 2026準拠) ---
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-3-flash-preview"

def get_candidates():
    print("🔭 東証全体 & GMO銘柄をスキャン中...")
    
    # TV_SESSION_ID をヘッダーにセット
    headers = {"Cookie": f"sessionid={os.environ.get('TV_SESSION_ID')}"}
    
    # 日本株(東証)の野心的スキャン
    # 注意：TradingViewのフィールド名は 'RSI', 'SMA20', 'SMA50' など大文字が基本です
    try:
        tse_query = (Query().set_markets('japan')
                     .select('name', 'description', 'close', 'volume', 'RSI', 'SMA20', 'SMA50', 'high', 'low')
                     .where(
                         Column('RSI') > 50,
                         Column('close') > Column('SMA20'),
                         Column('type') == 'stock'
                     )
                     .limit(100).get_scanner_data(headers=headers))
        
        # ETFのスキャン
        etf_query = (Query().set_markets('japan')
                     .select('name', 'description', 'close', 'volume', 'RSI', 'SMA20', 'SMA50', 'high', 'low')
                     .where(Column('type') == 'etf')
                     .limit(50).get_scanner_data(headers=headers))

        df = pd.concat([pd.DataFrame(tse_query[1]), pd.DataFrame(etf_query[1])])
        return df
    except Exception as e:
        print(f"❌ TradingViewデータ取得失敗: {e}")
        return pd.DataFrame()

# --- 2. 2次フィルター: 憲章3.4の精密判定 ---
def filter_by_charter(df):
    if df.empty: return df
    
    # キャンドル品質判定 (0.7以上)
    df['candle_q'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # 75MAの代わりにSMA50で1次足切り(スクリーナー仕様)を行い、
    # 詳細はGeminiに画像で判定させるのが最も合理的です
    final_candidates = df[
        (df['candle_q'] > 0.7) & 
        (df['close'] > df['SMA50'])
    ].copy()
    
    return final_candidates

def main():
    raw_candidates = get_candidates()
    final_list = filter_by_charter(raw_candidates)
    
    if final_list.empty:
        print("📭 本日、基準に合致する銘柄はありませんでした。")
        return

    print(f"🎯 厳選された {len(final_list)} 銘柄をGeminiに送信します...")
    
    for _, row in final_list.iterrows():
        name = row['name']
        desc = row.get('description', name)
        
        print(f"🧐 最終分析中: {desc} ({name})")
        
        prompt = f"""
        あなたは『運用憲章3.4』をマスターしたシニア・ストラテジストです。
        銘柄: {desc} ({name})
        
        【テクニカル状況】
        - RSI: {row['RSI']:.1f}
        - 現在値: {row['close']}
        - SMA20: {row['SMA20']}
        
        【指示】
        1. この銘柄が日本株なら「楽天証券」、CFDなら「GMO」を考慮した助言をせよ。
        2. 最終判断を EXECUTE または WAIT で示せ。
        """
        
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            print(f"🤖 Geminiの判断:\n{response.text}\n")
            time.sleep(2) # レート制限対策
        except Exception as e:
            print(f"⚠️ Gemini分析エラー: {e}")

if __name__ == "__main__":
    main()
