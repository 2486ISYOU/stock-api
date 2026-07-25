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

app = FastAPI(title="股佳寶", version="2.0")

# 設定安全 Cookie 驗證機制與你的專屬密碼
COOKIE_NAME = "stock_session"
MY_SECRET_PASSWORD = "ChiaPaoKU1688940318skrskr"
cookie_sec = APIKeyCookie(name=COOKIE_NAME, auto_error=False)

# 載入訓練好的機器學習模型
reg_high = joblib.load("stock_high_regressor.pkl")
reg_low = joblib.load("stock_low_regressor.pkl")

feature_cols = [
    'Ret_1D', 'Ret_5D', 'Ret_20D',
    'Bias_5D', 'Bias_20D', 'Vol_Change_5D', 'RSI_14', 'ATR_14',
    'VIX_Level', 'VIX_Change_5D', 'SP500_Ret_5D', 'TWII_Ret_5D',
    'Oil_Price', 'Oil_Change_5D',
    'ES_Ret_1D', 'NQ_Ret_1D'
]

# 1. 奢華黑金登入畫面
@app.get("/login", response_class=HTMLResponse)
def login_page():
    return """
    <html>
        <head>
            <meta charset="UTF-8">
            <title>登入 - 股佳寶</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                .gold-text { color: #d4af37; }
                .gold-border { border-color: #d4af37; }
                .gold-bg { background-color: #d4af37; }
                .gold-bg:hover { background-color: #aa8c2c; }
            </style>
        </head>
        <body class="bg-black text-[#d4af37] flex items-center justify-center h-screen font-sans">
            <div class="bg-zinc-950 p-8 rounded-2xl shadow-2xl w-96 border gold-border">
                <div class="text-center mb-6">
                    <h1 class="text-3xl font-bold tracking-wider gold-text">⚜️ 股佳寶 ⚜️</h1>
                    <p class="text-xs text-zinc-400 mt-1">頂級股市多空預測系統</p>
                </div>
                <form action="/login" method="post" class="space-y-4">
                    <div>
                        <label class="block text-sm text-zinc-400 mb-2">請輸入存取密碼</label>
                        <input type="password" name="password" required class="w-full px-4 py-2 rounded bg-black border gold-border text-[#d4af37] focus:outline-none focus:ring-1 focus:ring-[#d4af37]">
                    </div>
                    <button type="submit" class="w-full gold-bg text-black font-bold py-2 rounded-lg transition shadow-lg cursor-pointer">解鎖系統</button>
                </form>
            </div>
        </body>
    </html>
    """

# 2. 處理登入並核發通行憑證
@app.post("/login")
def login(password: str = Form(...)):
    if secrets.compare_digest(password, MY_SECRET_PASSWORD):
        response = RedirectResponse(url="/disclaimer", status_code=status.HTTP_303_SEE_OTHER)
        response.set_cookie(key=COOKIE_NAME, value="authenticated", httponly=True)
        return response
    return HTMLResponse("<body style='background:black; color:#d4af37; font-family:sans-serif; text-align:center; padding-top:100px;'><h3>密碼錯誤！<a href='/login' style='color:#d4af37;'>點此重試</a></h3></body>", status_code=401)

# 3. 檢查登入狀態
def verify_session(cookie: str = Depends(cookie_sec)):
    if cookie != "authenticated":
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login"}
        )

# 4. 免責聲明畫面（必須滑到最底並勾選）
@app.get("/disclaimer", response_class=HTMLResponse)
def disclaimer_page(user: str = Depends(verify_session)):
    return """
    <html>
        <head>
            <meta charset="UTF-8">
            <title>免責聲明 - 股佳寶</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                .gold-text { color: #d4af37; }
                .gold-border { border-color: #d4af37; }
                .gold-bg { background-color: #d4af37; }
            </style>
        </head>
        <body class="bg-black text-[#d4af37] flex items-center justify-center h-screen font-sans">
            <div class="bg-zinc-950 p-8 rounded-2xl shadow-2xl w-[500px] border gold-border flex flex-col h-[550px]">
                <h2 class="text-2xl font-bold text-center mb-4 gold-text">📜 使用者免責聲明</h2>
                
                <div id="termsBox" onscroll="checkScroll()" class="flex-1 bg-black border border-zinc-800 p-4 rounded-lg overflow-y-auto text-sm text-zinc-300 space-y-3 mb-4">
                    <p class="font-bold gold-text">歡迎使用「股佳寶」系統。在您開始使用本系統提供的所有預測數據與分析工具前，請務必詳細閱讀以下條款：</p>
                    <p>1. <strong>參考性質</strong>：本系統所產出之所有多空預測結果、趨勢分析及數據指標，僅供學術研究與內部參考之用，不構成任何形式的投資建議、買賣邀約或保證獲利承諾。</p>
                    <p>2. <strong>投資風險</strong>：金融市場瞬息萬變，歷史數據與機器學習模型無法完全預測未來突發事件。使用者須自行評估市場風險，並對自身的投資決策負全責。</p>
                    <p>3. <strong>免責範圍</strong>：開發者與本系統營運團隊不對因使用本系統數據而導致的任何直接或間接財務損失承擔法律責任。</p>
                    <p>4. <strong>資料正確性</strong>：系統透過公開渠道（如 Yahoo Finance 等）抓取即時與歷史行情，若遇網路延遲或數據源異常，系統將不負預期中斷之責。</p>
                    <p class="text-zinc-500 italic">【請完整捲動至最底部，方可解鎖同意按鈕】</p>
                </div>

                <div class="space-y-3">
                    <label class="flex items-center space-x-2 text-sm text-zinc-400 select-none cursor-not-allowed" id="agreeLabel">
                        <input type="checkbox" id="agreeCheck" disabled onchange="toggleBtn()" class="w-4 h-4 accent-[#d4af37] cursor-not-allowed">
                        <span>我已詳細閱讀並同意上述所有免責聲明條款</span>
                    </label>
                    <button id="enterBtn" onclick="acceptDisclaimer()" disabled class="w-full bg-zinc-800 text-zinc-500 font-bold py-2 rounded-lg transition cursor-not-allowed">進入系統</button>
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

# 5. 主畫面（支援 4 個自選股分頁，每頁 50 支股票）
@app.get("/", response_class=HTMLResponse)
def home(request: Request, user: str = Depends(verify_session)):
    # 檢查是否已同意免責聲明
    disclaimer_cookie = request.cookies.get("disclaimer")
    if disclaimer_cookie != "agreed":
        return RedirectResponse(url="/disclaimer", status_code=303)

    return """
    <html>
        <head>
            <meta charset="UTF-8">
            <title>股佳寶 - 頂級股市多空預測儀表板</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                .gold-text { color: #d4af37; }
                .gold-border { border-color: #d4af37; }
                .gold-bg { background-color: #d4af37; }
                .gold-bg:hover { background-color: #aa8c2c; }
                .tab-active { border-bottom: 2px solid #d4af37; color: #d4af37; }
            </style>
        </head>
        <body class="bg-black text-[#d4af37] min-h-screen flex flex-col justify-between font-sans selection:bg-[#d4af37] selection:text-black">
            
            <div class="max-w-5xl mx-auto w-full p-6">
                <header class="flex justify-between items-center mb-6 border-b border-zinc-800 pb-4">
                    <div>
                        <h1 class="text-3xl font-bold tracking-wider gold-text">⚜️ 股佳寶 ⚜️</h1>
                        <p class="text-xs text-zinc-400 mt-1">頂級多空預測引擎</p>
                    </div>
                    <div>
                        <a href="/login" class="text-xs text-zinc-500 hover:text-[#d4af37] transition">重新登入</a>
                    </div>
                </header>

                <!-- 自選股分頁按鈕 -->
                <div class="flex border-b border-zinc-800 mb-6 gap-6">
                    <button onclick="switchTab(1)" id="tabBtn1" class="pb-2 font-semibold text-lg tab-active transition cursor-pointer">自選股 1</button>
                    <button onclick="switchTab(2)" id="tabBtn2" class="pb-2 font-semibold text-lg text-zinc-500 transition cursor-pointer">自選股 2</button>
                    <button onclick="switchTab(3)" id="tabBtn3" class="pb-2 font-semibold text-lg text-zinc-500 transition cursor-pointer">自選股 3</button>
                    <button onclick="switchTab(4)" id="tabBtn4" class="pb-2 font-semibold text-lg text-zinc-500 transition cursor-pointer">自選股 4</button>
                </div>

                <!-- 分頁管理區 -->
                <div class="bg-zinc-950 p-6 rounded-2xl shadow-xl border gold-border mb-6">
                    <div class="flex justify-between items-center mb-4">
                        <h2 id="watchlistTitle" class="text-xl font-bold gold-text">📋 自選股 1 管理 (上限 50 支)</h2>
                        <span id="stockCount" class="text-xs text-zinc-400">已儲存: 0 / 50</span>
                    </div>

                    <div class="flex gap-3 mb-4">
                        <input type="text" id="tickerInput" placeholder="輸入股票代號 (例: 2330.TW, 2454.TW, NVDA)" class="flex-1 px-4 py-2 rounded-lg bg-black border gold-border text-[#d4af37] focus:outline-none placeholder-zinc-600 text-sm">
                        <button onclick="addStock()" class="gold-bg text-black px-5 py-2 rounded-lg font-bold text-sm transition shadow cursor-pointer">新增股票</button>
                    </div>

                    <div id="stockChips" class="flex flex-wrap gap-2 mb-6 min-h-[50px] p-3 bg-black rounded-lg border border-zinc-900"></div>

                    <button onclick="runPrediction()" id="predictBtn" class="w-full gold-bg text-black font-extrabold py-3 rounded-lg transition shadow-xl text-lg cursor-pointer">🚀 開始一鍵多空預測</button>
                </div>

                <!-- 結果顯示區 -->
                <div id="resultSection" class="bg-zinc-950 p-6 rounded-2xl shadow-xl border gold-border hidden">
                    <h3 class="text-lg font-bold mb-3 gold-text">📊 分析預測報告</h3>
                    <pre id="outputResult" class="bg-black p-4 rounded-xl text-emerald-400 overflow-x-auto text-xs border border-zinc-900"></pre>
                </div>
            </div>

            <!-- 底部名片角落 (開發者資訊) -->
            <footer class="w-full border-t border-zinc-900 py-4 px-6 text-right text-xs text-zinc-500 bg-black tracking-wider">
                開發者: 顧家寶 | 開發者信箱: <a href="mailto:jgu9410@gmail.com" class="hover:text-[#d4af37] underline">jgu9410@gmail.com</a>
            </footer>

            <script>
                let currentTab = 1;
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
                    for (let i = 1; i <= 4; i++) {
                        const btn = document.getElementById(`tabBtn${i}`);
                        if (i === tabNum) {
                            btn.className = "pb-2 font-semibold text-lg tab-active transition cursor-pointer";
                        } else {
                            btn.className = "pb-2 font-semibold text-lg text-zinc-500 transition cursor-pointer";
                        }
                    }
                    document.getElementById('watchlistTitle').innerText = `📋 自選股 ${tabNum} 管理 (上限 50 支)`;
                    renderStocks();
                }

                function renderStocks() {
                    const watchlists = getWatchlists();
                    const stocks = watchlists[currentTab] || [];
                    const container = document.getElementById('stockChips');
                    document.getElementById('stockCount').innerText = `已儲存: ${stocks.length} / 50`;

                    container.innerHTML = '';
                    if (stocks.length === 0) {
                        container.innerHTML = '<span class="text-zinc-600 text-xs italic self-center">目前尚無自選股，請於上方輸入代號新增。</span>';
                        return;
                    }

                    stocks.forEach((ticker, index) => {
                        const chip = document.createElement('div');
                        chip.className = "flex items-center gap-2 bg-zinc-900 border gold-border px-3 py-1.5 rounded-md text-sm text-[#d4af37]";
                        chip.innerHTML = `
                            <span>${ticker}</span>
                            <button onclick="removeStock(${index})" class="text-zinc-500 hover:text-red-400 font-bold ml-1 cursor-pointer">&times;</button>
                        `;
                        container.appendChild(chip);
                    });
                }

                function addStock() {
                    const input = document.getElementById('tickerInput');
                    const ticker = input.value.trim().toUpperCase();
                    if (!ticker) return;

                    let watchlists = getWatchlists();
                    let stocks = watchlists[currentTab];

                    if (stocks.length >= 50) {
                        alert('每個自選股分頁最多只能儲存 50 支股票！');
                        return;
                    }

                    if (stocks.includes(ticker)) {
                        alert('此股票已在清單中！');
                        return;
                    }

                    stocks.push(ticker);
                    watchlists[currentTab] = stocks;
                    saveWatchlists(watchlists);
                    input.value = '';
                    renderStocks();
                }

                function removeStock(index) {
                    let watchlists = getWatchlists();
                    watchlists[currentTab].splice(index, 1);
                    saveWatchlists(watchlists);
                    renderStocks();
                }

                async function runPrediction() {
                    let watchlists = getWatchlists();
                    let stocks = watchlists[currentTab];
                    if (stocks.length === 0) {
                        alert('請先在此分頁新增至少一支股票！');
                        return;
                    }

                    const btn = document.getElementById('predictBtn');
                    const resultSec = document.getElementById('resultSection');
                    const output = document.getElementById('outputResult');

                    btn.innerText = '分析運算中...';
                    btn.disabled = true;

                    const tickerString = stocks.join(',');
                    try {
                        const response = await fetch(`/predict/${encodeURIComponent(tickerString)}`);
                        const data = await response.json();
                        resultSec.classList.remove('hidden');
                        output.innerText = JSON.stringify(data, null, 2);
                    } catch (e) {
                        alert('預測請求失敗，請稍後再試');
                    } finally {
                        btn.innerText = '🚀 開始一鍵多空預測';
                        btn.disabled = false;
                    }
                }

                renderStocks();
            </script>
        </body>
    </html>
    """

# 6. 核心預測 API（串接你的真實 AI 模型與 yfinance）
@app.get("/predict/{tickers}")
def predict_stocks(tickers: str, user: str = Depends(verify_session)):
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        macro_tickers = ['^VIX', '^GSPC', '^TWII', 'CL=F', 'ES=F', 'NQ=F']
        
        all_data = yf.download(ticker_list + macro_tickers, period="6mo", interval="1d", auto_adjust=True, progress=False)
        
        close_df = all_data['Close'].ffill().bfill()
        high_df = all_data['High'].ffill().bfill()
        low_df = all_data['Low'].ffill().bfill()
        volume_df = all_data['Volume'].ffill().bfill()
        
        results = []
        
        for t in ticker_list:
            if t not in close_df.columns:
                results.append({"ticker": t, "error": "找不到此標的資料"})
                continue
                
            c = close_df[t]
            h = high_df[t]
            l = low_df[t]
            v = volume_df[t]
            
            df = pd.DataFrame({'Close': c, 'High': h, 'Low': l, 'Volume': v})
            
            df['Ret_1D'] = df['Close'].pct_change(1)
            df['Ret_5D'] = df['Close'].pct_change(5)
            df['Ret_20D'] = df['Close'].pct_change(20)
            
            ma5 = df['Close'].rolling(5).mean()
            ma20 = df['Close'].rolling(20).mean()
            df['Bias_5D'] = (df['Close'] - ma5) / ma5
            df['Bias_20D'] = (df['Close'] - ma20) / ma20
            df['Vol_Change_5D'] = df['Volume'].pct_change(5)
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            df['RSI_14'] = 100 - (100 / (1 + rs))
            
            tr = np.maximum(df['High'] - df['Low'], 
                            np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                     abs(df['Low'] - df['Close'].shift(1))))
            df['ATR_14'] = tr.rolling(14).mean() / df['Close']
            
            df['VIX_Level'] = close_df['^VIX']
            df['VIX_Change_5D'] = close_df['^VIX'].pct_change(5)
            df['SP500_Ret_5D'] = close_df['^GSPC'].pct_change(5)
            df['TWII_Ret_5D'] = close_df['^TWII'].pct_change(5)
            df['Oil_Price'] = close_df['CL=F']
            df['Oil_Change_5D'] = close_df['CL=F'].pct_change(5)
            df['ES_Ret_1D'] = close_df['ES=F'].pct_change(1)
            df['NQ_Ret_1D'] = close_df['NQ=F'].pct_change(1)
            
            latest_features = df[feature_cols].iloc[[-1]].replace([np.inf, -np.inf], np.nan).fillna(0)
            latest_close = float(df.iloc[-1]['Close'])
            latest_date = df.index[-1].strftime('%Y-%m-%d')
            
            pred_max = float(reg_high.predict(latest_features)[0])
            pred_min = float(reg_low.predict(latest_features)[0])
            
            results.append({
                "ticker": t,
                "date": latest_date,
                "latest_close": latest_close,
                "predicted_max_return_pct": round(pred_max * 100, 2),
                "predicted_min_return_pct": round(pred_min * 100, 2),
                "estimated_high_price": round(latest_close * (1 + pred_max), 2),
                "estimated_low_price": round(latest_close * (1 + pred_min), 2)
            })
            
        return {"predictions": results}
        
    except Exception as e:
        return {"error": str(e)}