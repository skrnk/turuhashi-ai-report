import os
import pandas as pd
from tradingview_screener import Query, Column
import google.generativeai as genai
import time

# --- 1. 初期設定 ---
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-3-flash-preview')

def get_candidates():
    print("🔭 東証全体 & GMO銘柄をスキャン中...")
    
    # 日本株(東証)の野心的スキャン
    # 憲章3.4の基本条件(RSI > 50, 価格 > 25MA)に合うものをサーバー側で抽出
    tse_query = (Query().set_markets('japan')
                 .select('name', 'description', 'close', 'volume', 'rsi', 'sma25', 'sma75', 'high', 'low')
                 .where(
                     Column('rsi') > 50,
                     Column('close') > Column('sma25'),
                     Column('type') == 'stock'
                 )
                 .limit(100).get_scanner_data())
    
    # ETF/指数のスキャン
    etf_query = (Query().set_markets('japan')
                 .select('name', 'description', 'close', 'volume', 'rsi', 'sma25', 'sma75', 'high', 'low')
                 .where(
                     Column('type') == 'etf'
                 )
                 .limit(50).get_scanner_data())

    # 取得データをPandasで統合
    df = pd.concat([pd.DataFrame(tse_query[1]), pd.DataFrame(etf_query[1])])
    return df

# --- 2. 2次フィルター: Python側で憲章3.4の精密判定 ---
def filter_by_charter(df):
    if df.empty: return df
    
    # キャンドル品質判定 (0.7以上)
    # (終値 - 安値) / (高値 - 安値)
    df['candle_q'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # 精密判定
    final_candidates = df[
        (df['candle_q'] > 0.7) & 
        (df['close'] > df['sma75']) # 75MAより上でマクロ順張り
    ].copy()
    
    return final_candidates

def main():
    raw_candidates = get_candidates()
    final_list = filter_by_charter(raw_candidates)
    
    print(f"🎯 厳選された {len(final_list)} 銘柄をGeminiに送信します...")
    
    for _, row in final_list.iterrows():
        name = row['name']
        desc = row.get('description', name)
        
        print(f"🧐 最終分析中: {desc} ({name})")
        
        prompt = f"""
        あなたは『運用憲章3.4』をマスターしたシニア・ストラテジストです。
        銘柄: {desc} ({name})
        
        【テクニカル状況】
        - RSI: {row['rsi']:.1f}
        - 現在値: {row['close']}
        - 25MA: {row['sma25']}
        
        【指示】
        1. この銘柄が日本株なら「楽天証券のトレーリングストップ」、CFDなら「GMOの節税」を考慮した助言をせよ。
        2. 最終判断を EXECUTE または WAIT で示せ。
        """
        
        try:
            response = model.generate_content(prompt)
            print(f"🤖 Geminiの判断:\n{response.text}\n")
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ エラー: {e}")

if __name__ == "__main__":
    main()
