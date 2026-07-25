from fastapi import FastAPI, Request, Form, HTTPException, status, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import APIKeyCookie
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import secrets

warnings.filterwarnings('ignore')

app = FastAPI(title="股佳寶", version="2.9")

COOKIE_NAME = "stock_session"
MY_SECRET_PASSWORD = "ChiaPaoKU1688940318skrskr"
cookie_sec = APIKeyCookie(name=COOKIE_NAME, auto_error=False)

# 載入機器學習模型
reg_high = joblib.load("stock_high_regressor.pkl")
reg_low = joblib.load("stock_low_regressor.pkl")

feature_cols = [
    'Ret_1D', 'Ret_5D', 'Ret_20D',
    'Bias_5D', 'Bias_20D', 'Vol_Change_5D', 'RSI_14', 'ATR_14',
    'VIX_Level', 'VIX_Change_5D', 'SP500_Ret_5D', 'TWII_Ret_5D',
    'Oil_Price', 'Oil_Change_5D',
    'ES_Ret_1D', 'NQ_Ret_1D'
]

def download_stock_data(ticker: str, period="3mo", interval="1d"):
    """自動處理上市 (.TW) 與上櫃 (.TWO) 容錯下載，並完整標準化 OHLCV 欄位的核心輔助函式"""
    ticker = ticker.strip().upper()
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    
    if df.empty:
        if ticker.endswith(".TW"):
            alt_ticker = ticker[:-3] + ".TWO"
            df = yf.download(alt_ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        elif ticker.endswith(".TWO"):
            alt_ticker = ticker[:-4] + ".TW"
            df = yf.download(alt_ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        elif "." not in ticker:
            df = yf.download(ticker + ".TW", period=period, interval=interval, auto_adjust=True, progress=False)
            if df.empty:
                df = yf.download(ticker + ".TWO", period=period, interval=interval, auto_adjust=True, progress=False)
                
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).capitalize() for c in df.columns]
        
    return df

@app.get("/login", response_class=HTMLResponse)
def login_page():
    return """
    <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <title>登入 - 股佳寶 GoodJob</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                .gold-text { color: #d4af37; }
                .gold-border { border-color: #d4af37; }
                .gold-bg { background-color: #d4af37; }
                .gold-bg:hover { background-color: #aa8c2c; }
            </style>
        </head>
        <body class="bg-black text-[#d4af37] flex items-center justify-center min-h-screen font-sans p-4">
            <div class="bg-zinc-950 p-6 sm:p-8 rounded-2xl shadow-2xl w-full max-w-sm border gold-border">
                <div class="text-center mb-6">
                    <h1 class="text-2xl sm:text-3xl font-bold tracking-wider gold-text">⚜️ 股佳寶 ⚜️</h1>
                    <p class="text-sm font-semibold text-zinc-300 mt-1 tracking-widest">GoodJob</p>
                    <p class="text-xs text-zinc-400 mt-1">頂級股市多空預測系統</p>
                </div>
                <form action="/login" method="post" class="space-y-4">
                    <div>
                        <label class="block text-sm text-zinc-400 mb-2">請輸入存取密碼</label>
                        <input type="password" name="password" required class="w-full px-4 py-2.5 rounded bg-black border gold-border text-[#d4af37] focus:outline-none focus:ring-1 focus:ring-[#d4af37] text-sm">
                    </div>
                    <button type="submit" class="w-full gold-bg text-black font-bold py-2.5 rounded-lg transition shadow-lg cursor-pointer">解鎖系統</button>
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/login")
def login(password: str = Form(...)):
    if secrets.compare_digest(password, MY_SECRET_PASSWORD):
        response = RedirectResponse(url="/disclaimer", status_code=status.HTTP_303_SEE_OTHER)
        response.set_cookie(key=COOKIE_NAME, value="authenticated", httponly=True)
        return response
    return HTMLResponse("<body style='background:black; color:#d4af37; font-family:sans-serif; text-align:center; padding-top:100px;'><h3>密碼錯誤！<a href='/login' style='color:#d4af37;'>點此重試</a></h3></body>", status_code=401)

def verify_session(cookie: str = Depends(cookie_sec)):
    if cookie != "authenticated":
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login"}
        )

# ==================== 修復：補上 K 線圖所需的 API 端點 ====================
@app.get("/chart-data/{ticker}")
def get_chart_data(ticker: str, user: str = Depends(verify_session)):
    try:
        df = download_stock_data(ticker, period="3mo", interval="1d")
        if df.empty:
            return {"error": "No data found"}
        
        df = df.reset_index()
        date_col = 'Date' if 'Date' in df.columns else df.columns[0]
        
        candles = []
        for _, row in df.iterrows():
            try:
                dt = pd.to_datetime(row[date_col]).strftime('%Y-%m-%d')
                open_p = float(row['Open'])
                high_p = float(row['High'])
                low_p = float(row['Low'])
                close_p = float(row['Close'])
                
                if pd.notna(open_p) and pd.notna(high_p) and pd.notna(low_p) and pd.notna(close_p):
                    candles.append({
                        "time": dt,
                        "open": open_p,
                        "high": high_p,
                        "low": low_p,
                        "close": close_p
                    })
            except Exception:
                continue
                
        return {"candles": candles}
    except Exception as e:
        return {"error": str(e)}

@app.get("/prices/{tickers}")
def get_prices(tickers: str, user: str = Depends(verify_session)):
    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
    prices_res = {}
    for t in ticker_list:
        try:
            df = download_stock_data(t, period="5d", interval="1d")
            if not df.empty and len(df) >= 1:
                close_series = df['Close'].dropna()
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.iloc[:, 0]
                
                curr_price = float(close_series.iloc[-1])
                prev_price = float(close_series.iloc[-2]) if len(close_series) >= 2 else curr_price
                is_up = curr_price >= prev_price
                
                prices_res[t] = {
                    "price": round(curr_price, 2),
                    "is_up": is_up
                }
            else:
                prices_res[t] = {"price": "查無報價", "is_up": True}
        except Exception:
            prices_res[t] = {"price": "查無報價", "is_up": True}
    return {"prices": prices_res}
# =======================================================================

@app.get("/disclaimer", response_class=HTMLResponse)
def disclaimer_page(user: str = Depends(verify_session)):
    return """
    <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <title>免責聲明 - 股佳寶 GoodJob</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                .gold-text { color: #d4af37; }
                .gold-border { border-color: #d4af37; }
                .gold-bg { background-color: #d4af37; }
            </style>
        </head>
        <body class="bg-black text-[#d4af37] flex items-center justify-center min-h-screen font-sans p-4">
            <div class="bg-zinc-950 p-6 sm:p-8 rounded-2xl shadow-2xl w-full max-w-lg border gold-border flex flex-col h-[520px]">
                <h2 class="text-xl sm:text-2xl font-bold text-center mb-4 gold-text">📜 使用者免責聲明</h2>
                
                <div id="termsBox" onscroll="checkScroll()" class="flex-1 bg-black border border-zinc-800 p-4 rounded-lg overflow-y-auto text-xs sm:text-sm text-zinc-300 space-y-3 mb-4">
                    <p class="font-bold gold-text">歡迎使用「股佳寶 (GoodJob)」系統。在您開始使用本系統提供的所有預測數據與分析工具前，請務必詳細閱讀以下條款：</p>
                    <p>1. <strong>參考性質</strong>：本系統所產出之所有多空預測結果、趨勢分析及數據指標，僅供學術研究與內部參考之用，不構成任何形式的投資建議、買賣邀約或保證獲利承諾。</p>
                    <p>2. <strong>投資風險</strong>：金融市場瞬息萬變，歷史數據與機器學習模型無法完全預測未來突發事件。使用者須自行評估市場風險，並對自身的投資決策負全責。</p>
                    <p>3. <strong>免責範圍</strong>：開發者與本系統營運團隊不對因使用本系統數據而導致的任何直接或間接財務損失承擔法律責任。</p>
                    <p>4. <strong>資料正確性</strong>：系統透過公開渠道（如 Yahoo Finance 等）抓取即時與歷史行情，若遇網路延遲或數據源異常，系統將不負預期中斷之責。</p>
                    <p class="text-zinc-500 italic">【請完整捲動至最底部，方可解鎖同意按鈕】</p>
                </div>

                <div class="space-y-3">
                    <label class="flex items-center space-x-2 text-xs sm:text-sm text-zinc-400 select-none cursor-not-allowed" id="agreeLabel">
                        <input type="checkbox" id="agreeCheck" disabled onchange="toggleBtn()" class="w-4 h-4 accent-[#d4af37] cursor-not-allowed">
                        <span>我已詳細閱讀並同意上述所有免責聲明條款</span>
                    </label>
                    <button id="enterBtn" onclick="acceptDisclaimer()" disabled class="w-full bg-zinc-800 text-zinc-500 font-bold py-2.5 rounded-lg transition cursor-not-allowed">進入系統</button>
                </div>
            </div>

            <script>
                const box = document.getElementById('termsBox');
                const checkbox = document.getElementById('agreeCheck');
                const agreeLabel = document.getElementById('agreeLabel');
                const enterBtn = document.getElementById('enterBtn');
                let scrolledToBottom = false;

                function checkScroll() {
                    if (box.scrollTop + box.clientHeight >= box.scrollHeight - 10) {
                        scrolledToBottom = true;
                        checkbox.disabled = false;
                        checkbox.parentElement.classList.remove('cursor-not-allowed');
                        checkbox.classList.remove('cursor-not-allowed');
                        agreeLabel.classList.remove('cursor-not-allowed');
                        agreeLabel.classList.add('cursor-pointer', 'text-[#d4af37]');
                    }
                }

                function toggleBtn() {
                    if (checkbox.checked && scrolledToBottom) {
                        enterBtn.disabled = false;
                        enterBtn.classList.remove('bg-zinc-800', 'text-zinc-500', 'cursor-not-allowed');
                        enterBtn.classList.add('gold-bg', 'text-black', 'cursor-pointer', 'shadow-lg');
                    } else {
                        enterBtn.disabled = true;
                        enterBtn.classList.add('bg-zinc-800', 'text-zinc-500', 'cursor-not-allowed');
                        enterBtn.classList.remove('gold-bg', 'text-black', 'cursor-pointer', 'shadow-lg');
                    }
                }

                function acceptDisclaimer() {
                    document.cookie = "disclaimer=agreed; path=/";
                    window.location.href = "/";
                }
            </script>
        </body>
    </html>
    """

@app.get("/", response_class=HTMLResponse)
def home(request: Request, user: str = Depends(verify_session)):
    disclaimer_cookie = request.cookies.get("disclaimer")
    if disclaimer_cookie != "agreed":
        return RedirectResponse(url="/disclaimer", status_code=303)

    return """
    <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <title>股佳寶 GoodJob - 頂級股市多空預測儀表板</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <!-- 引入 TradingView Lightweight Charts -->
            <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
            <style>
                .gold-text { color: #d4af37; }
                .gold-border { border-color: #d4af37; }
                .gold-bg { background-color: #d4af37; }
                .gold-bg:hover { background-color: #aa8c2c; }
                .tab-active { border-bottom: 2px solid #d4af37; color: #d4af37; }
            </style>
        </head>
        <body class="bg-black text-[#d4af37] min-h-screen flex flex-col justify-between font-sans selection:bg-[#d4af37] selection:text-black">
            
            <div class="max-w-4xl mx-auto w-full p-3 sm:p-6">
                <header class="flex justify-between items-center mb-6 border-b border-zinc-800 pb-4">
                    <div>
                        <h1 class="text-2xl sm:text-3xl font-bold tracking-wider gold-text">⚜️ 股佳寶 ⚜️</h1>
                        <p class="text-xs text-zinc-400 mt-0.5">GoodJob | 頂級多空預測引擎</p>
                    </div>
                    <div>
                        <a href="/login" class="text-xs text-zinc-500 hover:text-[#d4af37] transition">重新登入</a>
                    </div>
                </header>

                <!-- 分頁按鈕列 -->
                <div class="flex border-b border-zinc-800 mb-6 gap-3 sm:gap-6 overflow-x-auto pb-2 scrollbar-none">
                    <button onclick="switchTab(1)" id="tabBtn1" class="pb-2 font-semibold text-sm sm:text-lg tab-active transition cursor-pointer whitespace-nowrap">自選股 1</button>
                    <button onclick="switchTab(2)" id="tabBtn2" class="pb-2 font-semibold text-sm sm:text-lg text-zinc-500 transition cursor-pointer whitespace-nowrap">自選股 2</button>
                    <button onclick="switchTab(3)" id="tabBtn3" class="pb-2 font-semibold text-sm sm:text-lg text-zinc-500 transition cursor-pointer whitespace-nowrap">自選股 3</button>
                    <button onclick="switchTab(4)" id="tabBtn4" class="pb-2 font-semibold text-sm sm:text-lg text-zinc-500 transition cursor-pointer whitespace-nowrap">自選股 4</button>
                    <button onclick="switchTab('selector')" id="tabBtnselector" class="pb-2 font-semibold text-sm sm:text-lg text-zinc-500 transition cursor-pointer whitespace-nowrap flex items-center gap-1">🌟 專業選股專區</button>
                </div>

                <!-- 分頁內容區 -->
                <div id="contentArea"></div>
            </div>

            <!-- AI 預測彈出視窗 Modal -->
            <div id="stockModal" class="fixed inset-0 bg-black/80 z-50 flex items-center justify-center hidden p-4">
                <div class="bg-zinc-950 border gold-border w-full max-w-lg rounded-2xl p-6 relative max-h-[90vh] overflow-y-auto shadow-2xl">
                    <button onclick="closeModal()" class="absolute top-4 right-4 text-zinc-400 hover:text-white text-2xl font-bold cursor-pointer">&times;</button>
                    <div id="modalContent"></div>
                </div>
            </div>

            <!-- 底部名片角落 -->
            <footer class="w-full border-t border-zinc-900 py-4 px-4 sm:px-6 text-right text-xs text-zinc-500 bg-black tracking-wider mt-12">
                開發者: 顧家寶 | 開發者信箱: <a href="mailto:jgu9410@gmail.com" class="hover:text-[#d4af37] underline">jgu9410@gmail.com</a>
            </footer>

            <script>
                let currentTab = 1;

                const categories = [
                    {
                        name: "🇹🇼 台股 - 半導體與電子零組件 (核心權值)",
                        stocks: [
                            { symbol: "2330.TW", name: "台積電" }, { symbol: "2454.TW", name: "聯發科" }, { symbol: "2317.TW", name: "鴻海" },
                            { symbol: "2308.TW", name: "台達電" }, { symbol: "2303.TW", name: "聯電" }, { symbol: "3711.TW", name: "日月光投控" },
                            { symbol: "2382.TW", name: "廣達" }, { symbol: "3231.TW", name: "緯創" }, { symbol: "2357.TW", name: "華碩" },
                            { symbol: "2353.TW", name: "宏碁" }, { symbol: "6669.TW", name: "緯穎" }, { symbol: "3017.TW", name: "奇鋐" },
                            { symbol: "2421.TW", name: "建準" }, { symbol: "3034.TW", name: "聯詠" }, { symbol: "2408.TW", name: "南亞科" },
                            { symbol: "2344.TW", name: "華邦電" }, { symbol: "2337.TW", name: "旺宏" }, { symbol: "6770.TW", name: "力積電" },
                            { symbol: "3037.TW", name: "欣興" }, { symbol: "3189.TW", name: "景碩" }, { symbol: "8046.TW", name: "南電" },
                            { symbol: "6239.TW", name: "力成" }, { symbol: "5425.TWO", name: "台半" }, { symbol: "3533.TW", name: "嘉澤" },
                            { symbol: "3661.TW", name: "世芯-KY" }, { symbol: "3443.TW", name: "創意" }, { symbol: "5269.TW", name: "祥碩" },
                            { symbol: "4968.TW", name: "立積" }, { symbol: "2449.TW", name: "京元電子" }, { symbol: "6531.TW", name: "愛普*" },
                            { symbol: "3035.TW", name: "智原" }, { symbol: "6271.TW", name: "同欣電" }, { symbol: "8299.TW", name: "群聯" },
                            { symbol: "4938.TW", name: "和碩" }, { symbol: "2324.TW", name: "仁寶" }, { symbol: "3293.TWO", name: "鈊象" },
                            { symbol: "3008.TW", name: "大立光" }, { symbol: "2379.TW", name: "瑞昱" }, { symbol: "2409.TW", name: "友達" },
                            { symbol: "3481.TW", name: "群創" }, { symbol: "4958.TW", name: "臻鼎-KY" }, { symbol: "6269.TW", name: "台郡" }
                        ]
                    },
                    {
                        name: "🏦 台股 - 金融保險與金控",
                        stocks: [
                            { symbol: "2881.TW", name: "富邦金" }, { symbol: "2882.TW", name: "國泰金" }, { symbol: "2891.TW", name: "中信金" },
                            { symbol: "2884.TW", name: "玉山金" }, { symbol: "2886.TW", name: "兆豐金" }, { symbol: "2885.TW", name: "元大金" },
                            { symbol: "2883.TW", name: "開發金" }, { symbol: "2888.TW", name: "新光金" }, { symbol: "2892.TW", name: "第一金" },
                            { symbol: "5880.TW", name: "合庫金" }, { symbol: "2880.TW", name: "華南金" }, { symbol: "2801.TW", name: "彰銀" },
                            { symbol: "5876.TW", name: "上海商銀" }, { symbol: "2834.TW", name: "臺企銀" }, { symbol: "2890.TW", name: "永豐金" },
                            { symbol: "2887.TW", name: "台新金" }, { symbol: "6005.TW", name: "群益證" }, { symbol: "2855.TW", name: "統一證" },
                            { symbol: "2823.TW", name: "中壽" }, { symbol: "5871.TW", name: "中租-KY" }, { symbol: "9941.TW", name: "裕融" },
                            { symbol: "2809.TW", name: "京城銀" }, { symbol: "2812.TW", name: "台中銀" }, { symbol: "2845.TW", name: "遠東銀" }
                        ]
                    },
                    {
                        name: "🚢 台股 - 傳產、塑化、航運與鋼鐵",
                        stocks: [
                            { symbol: "2603.TW", name: "長榮" }, { symbol: "2609.TW", name: "陽明" }, { symbol: "2615.TW", name: "萬海" },
                            { symbol: "2618.TW", name: "長榮航" }, { symbol: "2610.TW", name: "華航" }, { symbol: "2606.TW", name: "裕民" },
                            { symbol: "1301.TW", name: "台塑" }, { symbol: "1303.TW", name: "南亞" }, { symbol: "1326.TW", name: "台化" },
                            { symbol: "6505.TW", name: "台塑化" }, { symbol: "2002.TW", name: "中鋼" }, { symbol: "2014.TW", name: "中鴻" },
                            { symbol: "1101.TW", name: "台泥" }, { symbol: "1102.TW", name: "亞泥" }, { symbol: "1216.TW", name: "統一" },
                            { symbol: "2912.TW", name: "統一超" }, { symbol: "5903.TW", name: "全家" }, { symbol: "2207.TW", name: "和泰車" },
                            { symbol: "2201.TW", name: "裕隆" }, { symbol: "2105.TW", name: "正新" }, { symbol: "9904.TW", name: "寶成" },
                            { symbol: "1402.TW", name: "遠東新" }, { symbol: "9921.TW", name: "巨大" }, { symbol: "9914.TW", name: "美利達" },
                            { symbol: "3376.TW", name: "新日興" }, { symbol: "1504.TW", name: "東元" }, { symbol: "1519.TW", name: "華城" },
                            { symbol: "1513.TW", name: "中興電" }, { symbol: "1514.TW", name: "亞力" }, { symbol: "1503.TW", name: "士電" },
                            { symbol: "9910.TW", name: "豐泰" }, { symbol: "2903.TW", name: "遠百" }, { symbol: "2206.TW", name: "三陽工業" }
                        ]
                    },
                    {
                        name: "🧬 台股 - 生技醫療、營建與其他",
                        stocks: [
                            { symbol: "4743.TW", name: "合一" }, { symbol: "6446.TW", name: "藥華藥" }, { symbol: "6472.TW", name: "保瑞" },
                            { symbol: "4147.TW", name: "中天" }, { symbol: "1795.TW", name: "美時" }, { symbol: "4123.TW", name: "晟德" },
                            { symbol: "2501.TW", name: "國建" }, { symbol: "2542.TW", name: "興富發" }, { symbol: "5522.TW", name: "遠雄" },
                            { symbol: "9933.TW", name: "中鼎" }, { symbol: "9945.TW", name: "潤泰新" }, { symbol: "2915.TW", name: "潤泰全" },
                            { symbol: "5284.TW", name: "jpp-KY" }
                        ]
                    },
                    {
                        name: "🇺🇸 美國 - 科技巨頭與 AI 概念股",
                        stocks: [
                            { symbol: "NVDA", name: "輝達 (NVIDIA)" }, { symbol: "AAPL", name: "蘋果 (Apple)" }, { symbol: "MSFT", name: "微軟 (Microsoft)" },
                            { symbol: "TSLA", name: "特斯拉 (Tesla)" }, { symbol: "AMZN", name: "亞馬遜 (Amazon)" }, { symbol: "GOOGL", name: "谷歌 (Google)" },
                            { symbol: "META", name: "Meta (Facebook)" }, { symbol: "AMD", name: "超微半導體" }, { symbol: "NFLX", name: "網飛 (Netflix)" },
                            { symbol: "AVGO", name: "博通 (Broadcom)" }, { symbol: "QCOM", name: "高通 (Qualcomm)" }, { symbol: "INTC", name: "英特爾 (Intel)" },
                            { symbol: "SMCI", name: "美超微電腦" }, { symbol: "ARM", name: "安謀控股" }, { symbol: "PLTR", name: "帕蘭蒂爾 (Palantir)" },
                            { symbol: "TSM", name: "台積電ADR" }, { symbol: "ASML", name: "艾司摩爾" }, { symbol: "MU", name: "美光科技" }
                        ]
                    },
                    {
                        name: "💳 美國 - 金融, 消費與傳統巨頭",
                        stocks: [
                            { symbol: "JPM", name: "摩根大通" }, { symbol: "BRK-B", name: "波克夏海瑟威" }, { symbol: "V", name: "Visa" },
                            { symbol: "MA", name: "萬事達卡" }, { symbol: "BAC", name: "美國銀行" }, { symbol: "WMT", name: "沃爾瑪" },
                            { symbol: "JNJ", name: "強生公司" }, { symbol: "PG", name: "寶僑" }, { symbol: "DIS", name: "迪士尼" },
                            { symbol: "KO", name: "可口可樂" }, { symbol: "PEP", name: "百事公司" }, { symbol: "MCD", name: "麥當勞" }
                        ]
                    },
                    {
                        name: "📈 熱門指數, ETF 與海外基金",
                        stocks: [
                            { symbol: "0050.TW", name: "元大台灣50" }, { symbol: "006208.TW", name: "富邦台50" }, { symbol: "00878.TW", name: "國泰永續高股息" },
                            { symbol: "0056.TW", name: "元大高股息" }, { symbol: "00919.TW", name: "群益台灣精選高息" }, { symbol: "00929.TW", name: "復華台灣科技優息" },
                            { symbol: "009816.TW", name: "KGI台灣Top50" }, { symbol: "SPY", name: "標普500 ETF" }, { symbol: "QQQ", name: "那斯達克100 ETF" },
                            { symbol: "SOXX", name: "半導體ETF" }, { symbol: "VTI", name: "全美股票ETF" }, { symbol: "VT", name: "全球股票ETF" }
                        ]
                    }
                ];

                const stockNameMap = {};
                categories.forEach(cat => {
                    cat.stocks.forEach(s => {
                        stockNameMap[s.symbol.toUpperCase()] = s.name;
                    });
                });

                function getWatchlists() {
                    let data = localStorage.getItem('gubao_watchlists');
                    if (!data) {
                        let initial = { 1: [], 2: [], 3: [], 4: [] };
                        localStorage.setItem('gubao_watchlists', JSON.stringify(initial));
                        return initial;
                    }
                    return JSON.parse(data);
                }

                function saveWatchlists(data) {
                    localStorage.setItem('gubao_watchlists', JSON.stringify(data));
                }

                function switchTab(tabNum) {
                    currentTab = tabNum;
                    ['1', '2', '3', '4', 'selector'].forEach(i => {
                        const btn = document.getElementById(`tabBtn${i}`);
                        if (i == tabNum) {
                            btn.className = "pb-2 font-semibold text-sm sm:text-lg tab-active transition cursor-pointer whitespace-nowrap";
                        } else {
                            btn.className = "pb-2 font-semibold text-sm sm:text-lg text-zinc-500 transition cursor-pointer whitespace-nowrap";
                        }
                    });
                    renderContent();
                }

                async function renderContent() {
                    const area = document.getElementById('contentArea');
                    if (currentTab === 'selector') {
                        let html = `
                            <div class="bg-zinc-950 p-4 sm:p-6 rounded-2xl shadow-xl border gold-border space-y-6">
                                <div>
                                    <h2 class="text-lg sm:text-xl font-bold gold-text">🌟 專業選股分類專區</h2>
                                    <p class="text-xs text-zinc-400 mt-1">點擊「＋加入」按鈕可選擇將標的加入您的自選股分頁中！</p>
                                </div>
                        `;
                        categories.forEach(cat => {
                            html += `
                                <div class="space-y-3">
                                    <h3 class="text-xs sm:text-sm font-bold text-zinc-300 border-b border-zinc-900 pb-1">${cat.name}</h3>
                                    <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                            `;
                            cat.stocks.forEach(s => {
                                html += `
                                    <div class="bg-black border gold-border p-3 rounded-xl flex justify-between items-center hover:bg-zinc-900 transition shadow">
                                        <div>
                                            <div class="text-xs sm:text-sm font-bold text-white">${s.name}</div>
                                            <div class="text-xs gold-text mt-0.5">${s.symbol}</div>
                                        </div>
                                        <button onclick="promptAddStock('${s.symbol}', '${s.name}')" class="text-xs bg-zinc-800 hover:bg-[#d4af37] hover:text-black text-zinc-300 px-3 py-1.5 rounded-lg transition font-bold whitespace-nowrap cursor-pointer">＋加入</button>
                                    </div>
                                `;
                            });
                            html += `</div></div>`;
                        });
                        html += `</div>`;
                        area.innerHTML = html;
                    } else {
                        const watchlists = getWatchlists();
                        const stocks = watchlists[currentTab] || [];
                        area.innerHTML = `
                            <div class="bg-zinc-950 p-4 sm:p-6 rounded-2xl shadow-xl border gold-border">
                                <div class="flex justify-between items-center mb-4">
                                    <h2 class="text-lg sm:text-xl font-bold gold-text">📋 自選股 ${currentTab} 管理 (已加入 ${stocks.length}/50 檔)</h2>
                                    <span class="text-xs text-zinc-400">即時日K線與多空預測</span>
                                </div>

                                <div class="flex gap-2 sm:gap-3 mb-6">
                                    <input type="text" id="tickerInput" placeholder="輸入代號 (例: 2330.TW, NVDA, 5425.TWO)" class="flex-1 px-4 py-2 rounded-lg bg-black border gold-border text-[#d4af37] focus:outline-none placeholder-zinc-600 text-xs sm:text-sm">
                                    <button onclick="addStock()" class="gold-bg text-black px-4 sm:px-5 py-2 rounded-lg font-bold text-xs sm:text-sm transition shadow cursor-pointer whitespace-nowrap">新增股票</button>
                                </div>

                                <div id="stockCardsGrid" class="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-2 min-h-[140px] p-3 bg-black rounded-xl border border-zinc-900"></div>
                            </div>
                        `;
                        await renderStocksCards(stocks);
                    }
                }

                async function renderStocksCards(stocks) {
                    const container = document.getElementById('stockCardsGrid');
                    if (!container) return;
                    container.innerHTML = '';
                    if (stocks.length === 0) {
                        container.innerHTML = '<div class="col-span-full text-zinc-500 text-xs italic text-center py-10">目前尚無自選股，請透過上方輸入框新增，或至「🌟 專業選股專區」挑選加入。</div>';
                        return;
                    }

                    let priceData = {};
                    try {
                        const res = await fetch(`/prices/${stocks.join(',')}`);
                        const data = await res.json();
                        priceData = data.prices || {};
                    } catch (e) {
                        console.error("無法取得即時股價", e);
                    }

                    stocks.forEach((ticker, index) => {
                        const name = stockNameMap[ticker] || ticker;
                        const info = priceData[ticker];
                        const price = info ? info.price : '查無報價';
                        const isUp = info ? info.is_up : true;
                        const priceColor = isUp ? 'text-rose-500' : 'text-emerald-400';

                        const card = document.createElement('div');
                        card.className = "bg-zinc-950 border gold-border p-4 rounded-xl flex flex-col justify-between shadow-lg relative";
                        card.innerHTML = `
                            <div class="flex justify-between items-start mb-2">
                                <div>
                                    <div class="text-xs text-zinc-400">${ticker}</div>
                                    <div class="text-base sm:text-lg font-bold text-white tracking-wide">${name}</div>
                                </div>
                                <button onclick="removeStock(${index})" class="text-zinc-500 hover:text-red-400 font-bold text-xl px-2 py-0.5 rounded cursor-pointer" title="刪除">&times;</button>
                            </div>
                            
                            <!-- 內嵌於卡片內的 K 線圖容器 -->
                            <div id="mini-chart-${index}" class="w-full h-32 bg-black rounded-lg border border-zinc-900 my-2 relative overflow-hidden flex items-center justify-center">
                                <span class="text-[10px] text-zinc-500">載入 K 線中...</span>
                            </div>

                            <div class="flex justify-between items-center pt-2 border-t border-zinc-900">
                                <div>
                                    <span class="text-xs text-zinc-400">收盤價：</span>
                                    <span class="text-sm sm:text-base font-bold ${priceColor}">${price}</span>
                                </div>
                                <button onclick="openPredictionModal('${ticker}', '${name}', '${price}')" class="text-xs gold-text font-bold px-3 py-1.5 rounded bg-black border gold-border hover:bg-[#d4af37] hover:text-black transition cursor-pointer">
                                    🚀 AI 預測分析
                                </button>
                            </div>
                        `;
                        container.appendChild(card);
                    });

                    requestAnimationFrame(() => {
                        stocks.forEach((ticker, index) => {
                            loadMiniCandlestickChart(ticker, `mini-chart-${index}`);
                        });
                    });
                }

                // ==================== 修復：補齊前端 K 線繪製函式 ====================
                async function loadMiniCandlestickChart(ticker, containerId) {
                    const container = document.getElementById(containerId);
                    if (!container) return;

                    try {
                        const res = await fetch(`/chart-data/${encodeURIComponent(ticker)}`);
                        const data = await res.json();

                        container.innerHTML = '';

                        if (data.error || !data.candles || data.candles.length === 0) {
                            container.innerHTML = '<span class="text-[10px] text-red-400">無法載入 K 線</span>';
                            return;
                        }

                        const width = container.clientWidth || 280;
                        const height = container.clientHeight || 128;

                        const chart = LightweightCharts.createChart(container, {
                            width: width,
                            height: height,
                            layout: {
                                background: { type: 'solid', color: '#000000' },
                                textColor: '#d4af37',
                            },
                            grid: {
                                vertLines: { color: '#18181b' },
                                horzLines: { color: '#18181b' },
                            },
                            timeScale: {
                                borderColor: '#27272a',
                                visible: false,
                            },
                            rightPriceScale: {
                                borderColor: '#27272a',
                            }
                        });

                        const candlestickSeries = chart.addCandlestickSeries({
                            upColor: '#ef4444',      // 漲 (台股習慣紅)
                            downColor: '#22c55e',    // 跌 (台股習慣綠)
                            borderVisible: false,
                            wickUpColor: '#ef4444',
                            wickDownColor: '#22c55e',
                        });

                        candlestickSeries.setData(data.candles);
                        chart.timeScale().fitContent();

                    } catch (err) {
                        console.error("Chart load error:", err);
                        container.innerHTML = '<span class="text-[10px] text-red-400">K 線載入異常</span>';
                    }
                }
                // ================================================================

                function addStock() {
                    const input = document.getElementById('tickerInput');
                    if (!input) return;
                    let val = input.value.trim().toUpperCase();
                    if (!val) return;

                    let watchlists = getWatchlists();
                    let stocks = watchlists[currentTab] || [];
                    if (stocks.length >= 50) {
                        alert("每組自選股最多只能加入 50 檔標的！");
                        return;
                    }
                    if (stocks.includes(val)) {
                        alert("此標的已在自選股清單中！");
                        return;
                    }

                    stocks.push(val);
                    watchlists[currentTab] = stocks;
                    saveWatchlists(watchlists);
                    input.value = '';
                    renderContent();
                }

                function removeStock(index) {
                    let watchlists = getWatchlists();
                    let stocks = watchlists[currentTab] || [];
                    stocks.splice(index, 1);
                    watchlists[currentTab] = stocks;
                    saveWatchlists(watchlists);
                    renderContent();
                }

                function promptAddStock(symbol, name) {
                    if (confirm(`確定要將 ${name} (${symbol}) 加入目前的自選股 ${currentTab} 嗎？`)) {
                        let watchlists = getWatchlists();
                        let stocks = watchlists[currentTab] || [];
                        if (stocks.length >= 50) {
                            alert("目前自選股分頁已滿 (上限 50 檔)！");
                            return;
                        }
                        if (!stocks.includes(symbol)) {
                            stocks.push(symbol);
                            watchlists[currentTab] = stocks;
                            saveWatchlists(watchlists);
                            alert(`成功將 ${name} 加入自選股 ${currentTab}！`);
                        } else {
                            alert("該標的已經在目前的自選股分頁中了！");
                        }
                    }
                }

                function openPredictionModal(ticker, name, price) {
                    const modal = document.getElementById('stockModal');
                    const content = document.getElementById('modalContent');
                    modal.classList.remove('hidden');
                    content.innerHTML = `
                        <div class="space-y-4">
                            <div class="border-b border-zinc-800 pb-3">
                                <h3 class="text-lg font-bold gold-text">🚀 AI 多空預測引擎</h3>
                                <p class="text-xs text-zinc-400">${ticker} - ${name} (現價: ${price})</p>
                            </div>
                            <div class="text-xs text-zinc-300 space-y-2">
                                <p>系統正在透過機器學習模型進行特徵工程與多空預測分析...</p>
                                <div class="p-3 bg-black rounded border border-zinc-800 text-center text-zinc-500 italic">
                                    (請在此處串接您的預測 API 邏輯)
                                </div>
                            </div>
                            <button onclick="closeModal()" class="w-full bg-zinc-800 hover:bg-[#d4af37] hover:text-black text-zinc-300 font-bold py-2 rounded transition cursor-pointer text-xs">關閉視窗</button>
                        </div>
                    `;
                }

                function closeModal() {
                    document.getElementById('stockModal').classList.add('hidden');
                }

                window.onload = function() {
                    renderContent();
                };
            </script>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)