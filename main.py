import os
import pandas as pd
# pandas-ta のインポートを確実に
try:
    import pandas_ta as ta
except ImportError:
    print("⚠️ pandas_ta の読み込みに失敗しました。")

import mplfinance as mpf
from tradingview_screener import Query, Column
from google import genai
import io
import time

# tvdatafeed のインポート（大文字小文字の両方に対応）
try:
    from tvdatafeed import TvDatafeed, Interval
except ImportError:
    from tvDatafeed import TvDatafeed, Interval

# --- 1. 初期設定 ---
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-3-flash-preview"
TV_SESS_ID = os.environ.get('TV_SESSION_ID')

# Session IDを使用してインスタンス化
# Session IDがない場合は警告を出して続行
if not TV_SESS_ID:
    print("⚠️ TV_SESSION_ID が設定されていません。パブリックデータのみ使用します。")
tv = TvDatafeed(username=None, password=None, sessionid=TV_SESS_ID)

def get_filtered_candidates():
    print("🔭 ステップ1: スクリーナーで広域スキャン中...")
    headers = {"Cookie": f"sessionid={TV_SESS_ID}"} if TV_SESS_ID else None
    
    try:
        # フィールド名は TV API の仕様に合わせ大文字
        q = (Query().set_markets('japan')
             .select('name', 'description', 'close', 'RSI', 'SMA5', 'SMA20', 'SMA50', 'high', 'low')
             .where(Column('type').isin(['stock', 'etf']))
             .limit(100).get_scanner_data(headers=headers))
        
        df = pd.DataFrame(q[1])
        if df.empty: return df

        # 2次フィルター: キャンドル品質と乖離率
        df['candle_q'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['dist_25'] = (df['close'] / df['SMA20'] - 1) * 100
        
        # Active(順張り): 強いトレンドと高キャンドル品質
        active_df = df[(df['RSI'] > 53) & (df['SMA5'] > df['SMA20']) & (df['close'] > df['SMA50']) & (df['candle_q'] > 0.8)].copy()
        active_df['strategy'] = 'Active'
        
        # Sniper(逆張り): 大きな乖離と反発の兆し
        sniper_df = df[(df['dist_25'] < -12) & (df['candle_q'] > 0.65)].copy()
        sniper_df['strategy'] = 'Sniper'
        
        # 重複を排除し、最大5件に絞る（Geminiの解析時間を最適化）
        combined = pd.concat([active_df, sniper_df]).drop_duplicates(subset=['name'])
        return combined.head(5)
    except Exception as e:
        print(f"❌ スキャンエラー: {e}")
        return pd.DataFrame()

def create_chart_bytes(symbol, interval_type, n_bars=70):
    try:
        # インターバル設定
        interval = Interval.in_daily if interval_type == 'daily' else Interval.in_weekly
        # シンボルから市場プレフィックスを除去 (例: TSE:7203 -> 7203)
        pure_symbol = symbol.split(':')[-1]
        
        hist = tv.get_hist(symbol=pure_symbol, exchange='TSE', interval=interval, n_bars=n_bars)
        if hist is None or hist.empty: return None
        
        # テクニカル指標の付与 (pandas-ta を使用)
        hist.ta.sma(length=25, append=True)
        hist.ta.sma(length=75, append=True)
        
        # カラム名の正規化 (pandas-ta は SMA_25 のような名前で作成することがある)
        sma25_col = [c for c in hist.columns if 'SMA_25' in c or 'sma25' in c][0]
        sma75_col = [c for c in hist.columns if 'SMA_75' in c or 'sma75' in c][0]
        
        buf = io.BytesIO()
        ap = [
            mpf.make_addplot(hist[sma25_col], color='orange', width=0.8),
            mpf.make_addplot(hist[sma75_col], color='blue', width=0.8)
        ]
        mpf.plot(hist, type='candle', style='charles', addplot=ap, savefig=buf, volume=True, tight_layout=True)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"⚠️ チャート生成失敗({symbol}): {e}")
        return None

def main():
    final_candidates = get_filtered_candidates()
    
    if final_candidates.empty:
        print("📭 本日、精密判定をクリアした銘柄はありませんでした。")
        return

    print(f"🎯 精鋭 {len(final_candidates)} 銘柄をマルチタイムフレーム解析します...")
    
    for _, row in final_candidates.iterrows():
        name = row['name']
        strategy = row['strategy']
        print(f"📈 【{strategy}モード】分析中: {name}")
        
        img_daily = create_chart_bytes(name, 'daily')
        img_weekly = create_chart_bytes(name, 'weekly')
        
        prompt = f"""
        あなたは『運用憲章3.4』のマスター・AIストラテジストです。
        銘柄: {row.get('description', name)} ({name})
        戦略: {strategy}
        
        【指示】
        添付された1枚目（日足）と2枚目（週足）のチャートを読み取り、以下のマルチタイムフレーム分析を行ってください。
        1. 週足が上昇トレンドで、日足のActiveシグナルを肯定しているか？
        2. 直近の価格抵抗帯はどこか？
        3. 最後に、EXECUTE または WAIT で最終判断を示せ。
        """
        
        try:
            contents = [prompt]
            if img_daily:
                contents.append(genai.types.Part.from_bytes(data=img_daily.read(), mime_type="image/png"))
            if img_weekly:
                contents.append(genai.types.Part.from_bytes(data=img_weekly.read(), mime_type="image/png"))
            
            response = client.models.generate_content(model=MODEL_NAME, contents=contents)
            print(f"🤖 Geminiの結論:\n{response.text}\n" + "="*50)
            time.sleep(5)
        except Exception as e:
            print(f"⚠️ Gemini解析エラー: {e}")

if __name__ == "__main__":
    main()
