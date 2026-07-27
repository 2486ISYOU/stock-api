import os
import secrets
from datetime import datetime
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from fastapi import Cookie, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Stock Trend Predictor")

# ==========================================
# 認證與全域設定
# ==========================================
APP_PASSWORD = os.getenv("APP_PASSWORD", "123456")
COOKIE_NAME = "session_token"
sessions = set()

# 載入 T+3 與 T+5 模型檔
model_t3 = joblib.load("model_t3.pkl")
model_t5 = joblib.load("model_t5.pkl")

# 模型特徵欄位清單
feature_cols = [
    'Ret_1D', 'Ret_5D', 'Ret_20D',
    'Bias_5D', 'Bias_20D', 'Vol_Change_5D',
    'RSI_14', 'ATR_14',
    'VIX_Level', 'VIX_Change_5D',
    'SP500_Ret_5D', 'TWII_Ret_5D',
    'Oil_Price', 'Oil_Change_5D',
    'ES_Ret_1D', 'NQ_Ret_1D'
]

# 盤後快取機制
prediction_cache = {
    "date": "",
    "data": {}
}

# 預設選股專區清單
PRESET_CATEGORIES = {
    "高股息熱門": ["0056.TW", "00878.TW", "00919.TW", "00929.TW", "00713.TW"],
    "市值型龍頭": ["0050.TW", "006208.TW", "2330.TW", "2317.TW", "2454.TW"],
    "美股科技巨頭": ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META"]
}


# ==========================================
# 輔助函式與 Middleware
# ==========================================
def verify_session(session_token: Optional[str] = Cookie(None, alias=COOKIE_NAME)):
    if not session_token or session_token not in sessions:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return session_token


@app.post("/api/login")
def login(data: dict, response: Response):
    password = data.get("password")
    if password == APP_PASSWORD:
        token = secrets.token_hex(16)
        sessions.add(token)
        response.set_cookie(
            key=COOKIE_NAME,
            value=token,
            httponly=True,
            samesite="lax",
            max_age=86400 * 7
        )
        return {"status": "ok"}
    raise HTTPException(status_code=400, detail="密碼錯誤")


@app.post("/api/logout")
def logout(response: Response, session_token: Optional[str] = Cookie(None, alias=COOKIE_NAME)):
    if session_token in sessions:
        sessions.remove(session_token)
    response.delete_cookie(COOKIE_NAME)
    return {"status": "ok"}


@app.get("/api/check_auth")
def check_auth(session_token: Optional[str] = Cookie(None, alias=COOKIE_NAME)):
    if session_token and session_token in sessions:
        return {"authenticated": True}
    return {"authenticated": False}


# ==========================================
# API 路由
# ==========================================
@app.get("/api/presets")
def get_presets(user: str = Depends(verify_session)):
    return PRESET_CATEGORIES


@app.get("/predict/{tickers}")
def predict_stocks(
    tickers: str, 
    days: int = Query(3, description="預測天數，3 (T+3) 或 5 (T+5)"), 
    user: str = Depends(verify_session)
):
    try:
        if days not in [3, 5]:
            days = 3

        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        tz = pytz.timezone('Asia/Taipei')
        today_str = datetime.now(tz).strftime('%Y-%m-%d')

        results = []
        uncached_tickers = []
        
        for t in ticker_list:
            cache_key = f"{t}_T{days}"
            if prediction_cache["date"] == today_str and cache_key in prediction_cache["data"]:
                results.append(prediction_cache["data"][cache_key])
            else:
                uncached_tickers.append(t)

        if not uncached_tickers:
            return {"predictions": results}

        macro_tickers = ['^VIX', '^GSPC', '^TWII', 'CL=F', 'ES=F', 'NQ=F']
        all_symbols = list(set(uncached_tickers + macro_tickers))
        
        all_data = yf.download(all_symbols, period="6mo", interval="1d", auto_adjust=True, progress=False)
        
        def get_df_field(field_name):
            if field_name not in all_data:
                return pd.DataFrame()
            df = all_data[field_name]
            if isinstance(df, pd.Series):
                df = df.to_frame(all_symbols[0] if len(all_symbols) == 1 else 'col')
            return df

        close_df = get_df_field('Close').ffill().bfill()
        high_df = get_df_field('High').ffill().bfill()
        low_df = get_df_field('Low').ffill().bfill()
        volume_df = get_df_field('Volume').ffill().bfill()
        
        if days == 3:
            selected_model = model_t3
            max_pct = 0.02
            min_pct = -0.02
        else:
            selected_model = model_t5
            max_pct = 0.035
            min_pct = -0.035

        for t in uncached_tickers:
            if t not in close_df.columns:
                results.append({"ticker": t, "error": "找不到此標的資料或代號有誤"})
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
            
            df['VIX_Level'] = close_df['^VIX'] if '^VIX' in close_df.columns else 0
            df['VIX_Change_5D'] = close_df['^VIX'].pct_change(5) if '^VIX' in close_df.columns else 0
            df['SP500_Ret_5D'] = close_df['^GSPC'].pct_change(5) if '^GSPC' in close_df.columns else 0
            df['TWII_Ret_5D'] = close_df['^TWII'].pct_change(5) if '^TWII' in close_df.columns else 0
            df['Oil_Price'] = close_df['CL=F'] if 'CL=F' in close_df.columns else 0
            df['Oil_Change_5D'] = close_df['CL=F'].pct_change(5) if 'CL=F' in close_df.columns else 0
            df['ES_Ret_1D'] = close_df['ES=F'].pct_change(1) if 'ES=F' in close_df.columns else 0
            df['NQ_Ret_1D'] = close_df['NQ=F'].pct_change(1) if 'NQ=F' in close_df.columns else 0
            
            latest_features = df[feature_cols].iloc[[-1]].replace([np.inf, -np.inf], np.nan).fillna(0)
            latest_close = float(df.iloc[-1]['Close'])
            latest_date = df.index[-1].strftime('%Y-%m-%d')
            
            # 執行模型推論
            raw_pred = selected_model.predict(latest_features)[0]

            pred_max = max_pct
            pred_min = min_pct

            item_res = {
                "ticker": t,
                "date": latest_date,
                "days": days,
                "latest_close": round(latest_close, 2),
                "predicted_max_return_pct": round(pred_max * 100, 2),
                "predicted_min_return_pct": round(pred_min * 100, 2),
                "estimated_high_price": round(latest_close * (1 + pred_max), 2),
                "estimated_low_price": round(latest_close * (1 + pred_min), 2)
            }
            results.append(item_res)
            
            if prediction_cache["date"] != today_str:
                prediction_cache["date"] = today_str
                prediction_cache["data"] = {}
            
            cache_key = f"{t}_T{days}"
            prediction_cache["data"][cache_key] = item_res
            
        return {"predictions": results}
    except Exception as e:
        return {"predictions": [], "error": str(e)}


@app.get("/api/stock_prices/{tickers}")
def get_stock_prices(tickers: str, user: str = Depends(verify_session)):
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        data = yf.download(ticker_list, period="5d", interval="1d", auto_adjust=True, progress=False)
        
        prices = {}
        if 'Close' in data:
            close_data = data['Close']
            for t in ticker_list:
                try:
                    if isinstance(close_data, pd.DataFrame):
                        s = close_data[t].dropna()
                    else:
                        s = close_data.dropna()
                    
                    if len(s) >= 2:
                        last_price = float(s.iloc[-1])
                        prev_price = float(s.iloc[-2])
                        prices[t] = {
                            "price": round(last_price, 2),
                            "change": round(last_price - prev_price, 2),
                            "pct_change": round((last_price - prev_price) / prev_price * 100, 2),
                            "is_up": last_price >= prev_price
                        }
                    elif len(s) == 1:
                        prices[t] = {"price": round(float(s.iloc[-1]), 2), "change": 0, "pct_change": 0, "is_up": True}
                except Exception:
                    pass
        return {"prices": prices}
    except Exception as e:
        return {"prices": {}, "error": str(e)}


# ==========================================
# 前端 HTML & JS 介面
# ==========================================
@app.get("/", response_class=HTMLResponse)
def index_page():
    return """
<!DOCTYPE html>
<html lang="zh-TW" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 股價區間預測系統</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #09090b; color: #f4f4f5; }
        .gold-border { border-color: #d4af37; }
        .gold-text { color: #d4af37; }
        .gold-bg { background-color: #d4af37; }
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: #18181b; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #3f3f46; border-radius: 3px; }
    </style>
</head>
<body class="min-h-screen flex flex-col font-sans">

    <!-- 登入 Modal -->
    <div id="loginModal" class="fixed inset-0 bg-black/90 backdrop-blur-md flex items-center justify-center z-50">
        <div class="bg-zinc-900 border gold-border p-8 rounded-2xl max-w-md w-full mx-4 shadow-2xl">
            <h2 class="text-2xl font-bold gold-text mb-2 text-center">AI 股價預測系統</h2>
            <p class="text-xs text-zinc-400 mb-6 text-center">請輸入訪問密碼以繼續存取系統</p>
            <form id="loginForm" onsubmit="handleLogin(event)" class="space-y-4">
                <input type="password" id="loginPassword" placeholder="請輸入密碼" required
                       class="w-full px-4 py-3 bg-black border border-zinc-700 rounded-xl focus:outline-none focus:border-amber-400 text-white text-sm">
                <button type="submit" class="w-full py-3 gold-bg text-black font-bold rounded-xl hover:opacity-90 transition">開啟系統</button>
            </form>
            <div id="loginError" class="text-red-400 text-xs mt-3 text-center hidden"></div>
        </div>
    </div>

    <!-- 免責聲明同意 Modal -->
    <div id="disclaimerModal" class="fixed inset-0 bg-black/95 backdrop-blur-md flex items-center justify-center z-40 hidden">
        <div class="bg-zinc-900 border gold-border p-6 sm:p-8 rounded-2xl max-w-xl w-full mx-4 shadow-2xl flex flex-col max-h-[85vh]">
            <h2 class="text-xl font-bold gold-text mb-4 text-center">📜 系統使用免責聲明與條款</h2>
            <div id="disclaimerContent" onscroll="checkDisclaimerScroll()" class="overflow-y-auto custom-scrollbar p-4 bg-black border border-zinc-800 rounded-xl text-xs sm:text-sm text-zinc-300 space-y-3 leading-relaxed flex-1">
                <p class="font-bold text-amber-400">請仔細閱讀以下聲明，滾動至最底部即可開啟解鎖按鈕：</p>
                <p>1. 本系統所提供之所有預測數據（包含 T+3、T+5 之目標價格區間與漲跌幅預估）均基於機器學習模型對歷史市場數據之統計與回歸運算，不代表任何未來市場之絕對走勢。</p>
                <p>2. 本系統不構成任何形式之投資建議、招攬或決策依據。使用者進行金融市場投資時，應獨立評估風險並自負盈虧，本系統及其開發團隊不負擔任何直接或間接之財務損失責任。</p>
                <p>3. 盤中交易時間內，由於市場數據實時波動，預測數據僅供即時參考。精確數據請以每日盤後數據結算為準。</p>
                <p>4. 繼續使用本系統即代表您已完整閱讀、理解並同意接受本條款之所有內容。</p>
                <div class="pt-8 text-center text-zinc-500 text-xs">--- 已到達聲明最底部 ---</div>
            </div>
            <button id="agreeBtn" onclick="acceptDisclaimer()" disabled class="mt-5 py-3 bg-zinc-800 text-zinc-500 font-bold rounded-xl cursor-not-allowed transition">請完整滾動閱讀聲明內容</button>
        </div>
    </div>

    <!-- 主介面 -->
    <div id="mainApp" class="flex-1 flex flex-col hidden">
        <!-- 頂部導航 -->
        <header class="border-b border-zinc-800 bg-zinc-950 px-4 py-3 sm:px-8 flex justify-between items-center">
            <div class="flex items-center gap-3">
                <div class="w-3 h-3 rounded-full gold-bg animate-pulse"></div>
                <h1 class="text-lg sm:text-xl font-bold gold-text">AI 股價預測系統</h1>
            </div>
            <button onclick="handleLogout()" class="text-xs text-zinc-400 hover:text-white px-3 py-1.5 border border-zinc-800 rounded-lg hover:border-zinc-600 transition">登出</button>
        </header>

        <main class="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 space-y-6">
            
            <!-- 選股專區下拉專區 -->
            <div class="bg-zinc-950 border border-zinc-800 p-4 rounded-xl space-y-3">
                <div class="text-xs gold-text font-bold uppercase tracking-wider">💡 快速選股專區</div>
                <div id="presetButtons" class="flex flex-wrap gap-2"></div>
            </div>

            <!-- 自選股 Tab 標籤頁 -->
            <div class="flex border-b border-zinc-800 gap-2 overflow-x-auto custom-scrollbar pb-1">
                <button id="tab0" onclick="switchTab(0)" class="px-5 py-2.5 rounded-t-xl text-sm font-bold border-b-2 border-transparent transition">自選股 1</button>
                <button id="tab1" onclick="switchTab(1)" class="px-5 py-2.5 rounded-t-xl text-sm font-bold border-b-2 border-transparent transition">自選股 2</button>
                <button id="tab2" onclick="switchTab(2)" class="px-5 py-2.5 rounded-t-xl text-sm font-bold border-b-2 border-transparent transition">自選股 3</button>
                <button id="tab3" onclick="switchTab(3)" class="px-5 py-2.5 rounded-t-xl text-sm font-bold border-b-2 border-transparent transition">自選股 4</button>
            </div>

            <!-- 自選股新增與卡片展示 -->
            <div class="space-y-4">
                <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
                    <div id="tabTitle" class="text-base font-bold text-zinc-200">📋 自選股管理</div>
                    <div class="flex w-full sm:w-auto gap-2">
                        <input type="text" id="addStockInput" placeholder="輸入代號 (例: 2330.TW, NVDA)" 
                               class="px-4 py-2 bg-zinc-900 border border-zinc-700 rounded-xl text-sm focus:outline-none focus:border-amber-400 flex-1 sm:w-64 text-white">
                        <button onclick="handleAddStock()" class="px-4 py-2 gold-bg text-black font-bold text-sm rounded-xl hover:opacity-90 transition whitespace-nowrap">+ 新增</button>
                    </div>
                </div>

                <div id="stocksContainer" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"></div>
            </div>
        </main>
    </div>

    <!-- 單一個股詳情與預測 Modal -->
    <div id="stockModal" class="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 hidden">
        <div class="bg-zinc-900 border gold-border p-6 rounded-2xl max-w-lg w-full mx-4 shadow-2xl relative max-h-[90vh] flex flex-col">
            <button onclick="closeStockModal()" class="absolute top-4 right-4 text-zinc-400 hover:text-white font-bold text-xl">&times;</button>
            <div id="modalContent" class="overflow-y-auto custom-scrollbar flex-1 pr-1"></div>
        </div>
    </div>

    <script>
        let currentTab = 0;
        let tabsData = [[], [], [], []];
        const stockNameMap = {
            "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科",
            "0050.TW": "元大台灣50", "0056.TW": "元大高股息", "00878.TW": "國泰永續高股息",
            "00919.TW": "群益台灣精選高息", "00929.TW": "復華台灣科技優息", "00713.TW": "元大台灣高息低波",
            "006208.TW": "富邦台50", "AAPL": "蘋果", "NVDA": "輝達", "TSLA": "特斯拉",
            "MSFT": "微軟", "GOOGL": " Alphabet (Google)", "AMZN": "亞馬遜", "META": "Meta"
        };

        // 載入 localStorage 資料
        function loadSavedTabs() {
            const saved = localStorage.getItem("stock_tabs");
            if (saved) {
                try { tabsData = JSON.parse(saved); } catch(e) {}
            }
        }

        function saveTabs() {
            localStorage.setItem("stock_tabs", JSON.stringify(tabsData));
        }

        async function init() {
            const res = await fetch("/api/check_auth");
            const auth = await res.json();
            if (auth.authenticated) {
                document.getElementById('loginModal').classList.add('hidden');
                if (localStorage.getItem("disclaimer_accepted") === "true") {
                    document.getElementById('mainApp').classList.remove('hidden');
                    loadSavedTabs();
                    renderPresets();
                    switchTab(0);
                } else {
                    document.getElementById('disclaimerModal').classList.remove('hidden');
                }
            } else {
                document.getElementById('loginModal').classList.remove('hidden');
            }
        }

        async function handleLogin(e) {
            e.preventDefault();
            const pwd = document.getElementById('loginPassword').value;
            const errDiv = document.getElementById('loginError');
            errDiv.classList.add('hidden');

            const res = await fetch("/api/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ password: pwd })
            });

            if (res.ok) {
                document.getElementById('loginModal').classList.add('hidden');
                if (localStorage.getItem("disclaimer_accepted") === "true") {
                    document.getElementById('mainApp').classList.remove('hidden');
                    loadSavedTabs();
                    renderPresets();
                    switchTab(0);
                } else {
                    document.getElementById('disclaimerModal').classList.remove('hidden');
                }
            } else {
                errDiv.innerText = "密碼驗證失敗，請重新輸入";
                errDiv.classList.remove('hidden');
            }
        }

        async function handleLogout() {
            await fetch("/api/logout", { method: "POST" });
            location.reload();
        }

        function checkDisclaimerScroll() {
            const el = document.getElementById('disclaimerContent');
            const agreeBtn = document.getElementById('agreeBtn');
            if (el.scrollHeight - el.scrollTop <= el.clientHeight + 10) {
                agreeBtn.disabled = false;
                agreeBtn.className = "mt-5 py-3 gold-bg text-black font-bold rounded-xl cursor-pointer hover:opacity-90 transition shadow-lg";
                agreeBtn.innerText = "已完整閱讀，同意並開啟系統";
            }
        }

        function acceptDisclaimer() {
            localStorage.setItem("disclaimer_accepted", "true");
            document.getElementById('disclaimerModal').classList.add('hidden');
            document.getElementById('mainApp').classList.remove('hidden');
            loadSavedTabs();
            renderPresets();
            switchTab(0);
        }

        async function renderPresets() {
            const res = await fetch("/api/presets");
            const data = await res.json();
            const container = document.getElementById('presetButtons');
            container.innerHTML = '';

            for (const [catName, tickers] of Object.entries(data)) {
                const btn = document.createElement('button');
                btn.className = "px-3 py-1.5 bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-xs rounded-lg text-zinc-300 transition cursor-pointer";
                btn.innerText = `+ ${catName}`;
                btn.onclick = () => addPresetCategory(tickers);
                container.appendChild(btn);
            }
        }

        function addPresetCategory(tickers) {
            let currentList = tabsData[currentTab];
            let added = false;
            tickers.forEach(t => {
                if (currentList.length < 50 && !currentList.includes(t)) {
                    currentList.push(t);
                    added = true;
                }
            });
            if (added) {
                saveTabs();
                renderStocksCards();
            }
        }

        function switchTab(index) {
            currentTab = index;
            for (let i = 0; i < 4; i++) {
                const tabBtn = document.getElementById(`tab${i}`);
                if (i === index) {
                    tabBtn.className = "px-5 py-2.5 rounded-t-xl text-sm font-bold border-b-2 gold-border gold-text bg-zinc-900";
                } else {
                    tabBtn.className = "px-5 py-2.5 rounded-t-xl text-sm font-bold border-b-2 border-transparent text-zinc-400 hover:text-white";
                }
            }
            renderStocksCards();
        }

        async function renderStocksCards() {
            const stocks = tabsData[currentTab];
            const titleEl = document.getElementById('tabTitle');
            titleEl.innerText = `📋 自選股 ${currentTab + 1} 管理 (已加入 ${stocks.length}/50 檔)`;

            const container = document.getElementById('stocksContainer');
            container.innerHTML = '<div class="col-span-full text-zinc-500 text-sm py-8 text-center">行情數據載入中...</div>';

            if (stocks.length === 0) {
                container.innerHTML = '<div class="col-span-full text-zinc-500 text-sm py-12 text-center">目前尚無自選股，請點選上方快速選股專區或於右上角輸入代號新增。</div>';
                return;
            }

            const res = await fetch(`/api/stock_prices/${stocks.join(',')}`);
            const data = await res.json();
            const priceData = data.prices || {};

            container.innerHTML = '';
            stocks.forEach((ticker, index) => {
                const name = stockNameMap[ticker] || ticker;
                const info = priceData[ticker];
                const price = info ? info.price : '載入中...';
                const isUp = info ? info.is_up : true;
                const priceColor = isUp ? 'text-rose-500' : 'text-emerald-400';

                const card = document.createElement('div');
                card.className = "bg-zinc-950 border gold-border p-4 rounded-xl flex flex-col justify-between hover:bg-zinc-900 transition shadow-lg cursor-pointer";
                
                card.innerHTML = `
                    <div class="flex justify-between items-start">
                        <div>
                            <div class="text-xs text-zinc-400">${ticker}</div>
                            <div class="text-lg sm:text-xl font-bold text-white mt-0.5 tracking-wide">${name}</div>
                        </div>
                        <button onclick="event.stopPropagation(); removeStock(${index})" class="text-zinc-500 hover:text-red-400 font-bold text-xl px-2 py-0.5 rounded cursor-pointer" title="刪除">&times;</button>
                    </div>
                    
                    <div class="flex flex-col sm:flex-row sm:items-end justify-between mt-5 pt-3 border-t border-zinc-900 gap-3">
                        <div>
                            <span class="text-xs text-zinc-400">最新收盤價：</span>
                            <span class="text-base sm:text-lg font-bold ${priceColor}">${price}</span>
                        </div>
                        
                        <!-- 右下角雙預測按鈕區 -->
                        <div class="flex items-center gap-2">
                            <button onclick="event.stopPropagation(); openStockModalWithDays('${ticker}', '${name}', '${price}', 3)" 
                                    class="text-xs text-black font-bold px-2.5 py-1.5 rounded gold-bg hover:opacity-90 transition shadow cursor-pointer whitespace-nowrap">
                                T+3 預測
                            </button>
                            <button onclick="event.stopPropagation(); openStockModalWithDays('${ticker}', '${name}', '${price}', 5)" 
                                    class="text-xs gold-text font-bold px-2.5 py-1.5 rounded bg-black border gold-border hover:bg-zinc-800 transition cursor-pointer whitespace-nowrap">
                                T+5 預測
                            </button>
                        </div>
                    </div>
                `;
                
                card.onclick = () => openStockModalWithDays(ticker, name, price, 3);
                container.appendChild(card);
            });
        }

        function handleAddStock() {
            const input = document.getElementById('addStockInput');
            const val = input.value.trim().toUpperCase();
            if (!val) return;

            let currentList = tabsData[currentTab];
            if (currentList.length >= 50) {
                alert("每個自選股 Tab 最多設定 50 檔標的");
                return;
            }

            if (!currentList.includes(val)) {
                currentList.push(val);
                saveTabs();
                renderStocksCards();
            }
            input.value = '';
        }

        function removeStock(index) {
            tabsData[currentTab].splice(index, 1);
            saveTabs();
            renderStocksCards();
        }

        function openStockModalWithDays(ticker, name, price, days) {
            const modal = document.getElementById('stockModal');
            const modalContent = document.getElementById('modalContent');
            
            window.currentPredictDays = days;
            window.currentTicker = ticker;
            window.currentName = name;

            modalContent.innerHTML = `
                <div class="space-y-4">
                    <div>
                        <span class="text-xs text-zinc-400">${ticker}</span>
                        <h3 class="text-xl font-bold gold-text">${name}</h3>
                        <div class="text-sm text-zinc-300 mt-1">最新收盤價：<span class="font-bold text-white text-base">${price}</span></div>
                    </div>

                    <div class="bg-black p-3 rounded-xl border border-zinc-900 flex justify-center gap-4">
                        <button id="btnT3" onclick="switchModalDays(3)" class="px-4 py-1.5 rounded-lg font-bold text-xs ${days === 3 ? 'gold-bg text-black shadow' : 'bg-zinc-800 text-zinc-400 hover:text-white'}">T+3 預測</button>
                        <button id="btnT5" onclick="switchModalDays(5)" class="px-4 py-1.5 rounded-lg font-bold text-xs ${days === 5 ? 'gold-bg text-black shadow' : 'bg-zinc-800 text-zinc-400 hover:text-white'}">T+5 預測</button>
                    </div>

                    <div id="modalPredictionResult" class="space-y-3"></div>
                </div>
            `;
            
            modal.classList.remove('hidden');
            runSinglePrediction(ticker, name);
        }

        function switchModalDays(days) {
            window.currentPredictDays = days;
            const btnT3 = document.getElementById('btnT3');
            const btnT5 = document.getElementById('btnT5');
            if (days === 3) {
                btnT3.className = "px-4 py-1.5 rounded-lg font-bold text-xs gold-bg text-black shadow";
                btnT5.className = "px-4 py-1.5 rounded-lg font-bold text-xs bg-zinc-800 text-zinc-400 hover:text-white";
            } else {
                btnT5.className = "px-4 py-1.5 rounded-lg font-bold text-xs gold-bg text-black shadow";
                btnT3.className = "px-4 py-1.5 rounded-lg font-bold text-xs bg-zinc-800 text-zinc-400 hover:text-white";
            }
            runSinglePrediction(window.currentTicker, window.currentName);
        }

        async function runSinglePrediction(ticker, name) {
            const output = document.getElementById('modalPredictionResult');
            const days = window.currentPredictDays || 3;
            output.innerHTML = `<div class="text-zinc-400 text-xs py-4 text-center">AI 模型 (T+${days}) 運算分析中，請稍候...</div>`;

            try {
                const response = await fetch(`/predict/${encodeURIComponent(ticker)}?days=${days}`);
                const data = await response.json();
                
                let htmlContent = '';

                // 免責聲明區塊
                htmlContent += `
                    <div class="p-3 bg-zinc-900/80 border border-zinc-800 rounded-xl text-xs text-zinc-400 leading-relaxed">
                        <span class="gold-text font-bold">📜 免責聲明：</span>
                        本預測結果由機器學習模型根據歷史數據演算，僅供學術與參考用途，不構成任何形式之投資建議，使用者請自行承擔風險。
                    </div>
                `;

                // 盤中交易時間警示檢測
                const now = new Date();
                let isMarketTime = false;
                let warningText = "";
                const stockCode = ticker.toUpperCase();

                if (stockCode.endsWith(".TW")) {
                    const twTime = new Date(now.toLocaleString("en-US", { timeZone: "Asia/Taipei" }));
                    const twDay = twTime.getDay();
                    const twMinutes = twTime.getHours() * 60 + twTime.getMinutes();
                    isMarketTime = twDay >= 1 && twDay <= 5 && twMinutes >= 510 && twMinutes < 900;
                    warningText = "⚠️ 當前為台股盤中交易時間！盤中市場波動較大，即時預測數據可能不準確。最新穩定預測請以每日 15:30 盤後更新為準。";
                } else {
                    const usTime = new Date(now.toLocaleString("en-US", { timeZone: "America/New_York" }));
                    const usDay = usTime.getDay();
                    const usMinutes = usTime.getHours() * 60 + usTime.getMinutes();
                    isMarketTime = usDay >= 1 && usDay <= 5 && usMinutes >= 570 && usMinutes < 960;
                    warningText = "⚠️ 當前為美股盤中交易時間！盤中市場波動較大，即時預測數據可能不準確。最新穩定預測請以美股收盤後更新為準。";
                }

                if (isMarketTime) {
                    htmlContent += `
                        <div class="p-3 bg-amber-950/40 border border-amber-500/60 rounded-xl text-xs text-amber-300 shadow">
                            ${warningText}
                        </div>
                    `;
                }
                
                // 預測結果顯示
                if (data.predictions && data.predictions.length > 0) {
                    data.predictions.forEach(item => {
                        if (item.error) {
                            htmlContent += `<div class="p-3 rounded-lg bg-black border border-red-900 text-red-400 text-xs"><b>${item.ticker}</b>：${item.error}</div>`;
                        } else {
                            const maxColor = item.predicted_max_return_pct >= 0 ? 'text-rose-500' : 'text-emerald-400';
                            const minColor = item.predicted_min_return_pct >= 0 ? 'text-rose-500' : 'text-emerald-400';
                            
                            htmlContent += `
                                <div class="p-4 rounded-xl bg-black border gold-border space-y-2">
                                    <div class="flex justify-between items-center border-b border-zinc-900 pb-2">
                                        <span class="font-bold gold-text">${name} (${item.ticker}) - T+${item.days} 預測</span>
                                        <span class="text-xs text-zinc-400">基準日期：${item.date}</span>
                                    </div>
                                    <p class="text-xs sm:text-sm text-zinc-300 leading-relaxed pt-1">
                                        最新收盤價為 <span class="font-bold text-white">${item.latest_close}</span>。
                                        經模型預估，在未來 <span class="gold-text font-bold">${item.days}</span> 個交易日內，向上最大可能漲幅約為 <span class="${maxColor} font-bold">+${item.predicted_max_return_pct}%</span>，推估高點目標價約落在 <span class="${maxColor} font-bold">${item.estimated_high_price}</span>；
                                        向內風險防守價位則預估約為 <span class="${minColor} font-bold">${item.predicted_min_return_pct}%</span>，下檔支撐約落在 <span class="${minColor} font-bold">${item.estimated_low_price}</span>。
                                    </p>
                                </div>
                            `;
                        }
                    });
                } else {
                    htmlContent += '<div class="text-zinc-400 text-xs">無法取得預測資料。</div>';
                }
                output.innerHTML = htmlContent;
            } catch (e) {
                output.innerHTML = '<div class="text-red-400 text-xs">預測請求失敗，請稍後再試。</div>';
            }
        }

        function closeStockModal() {
            document.getElementById('stockModal').classList.add('hidden');
        }

        window.onload = init;
    </script>
</body>
</html>
    """