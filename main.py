from fastapi import FastAPI, Request, Form, HTTPException, status, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import APIKeyCookie
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import secrets
from datetime import datetime, time
import pytz

warnings.filterwarnings('ignore')

app = FastAPI(title="股佳寶", version="2.7")

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

# 記憶體快取：儲存盤後預測結果，避免重複呼叫 API
prediction_cache = {
    "update_time": None,
    "date": None,
    "data": {}
}

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
                
                <div id="termsBox" class="flex-1 bg-black border border-zinc-800 p-4 rounded-lg overflow-y-auto text-xs sm:text-sm text-zinc-300 space-y-3 mb-4">
                    <p class="font-bold gold-text">歡迎使用「股佳寶 (GoodJob)」系統。在您開始使用本系統提供的所有預測數據與分析工具前，請務必詳細閱讀以下條款：</p>
                    <p>1. <strong>參考性質</strong>：本系統所產出之所有多空預測結果、趨勢分析及數據指標，僅供學術研究與內部參考之用，不構成任何形式的投資建議、買賣邀約或保證獲利承諾。</p>
                    <p>2. <strong>投資風險</strong>：金融市場瞬息萬變，歷史數據與機器學習模型無法完全預測未來突發事件。使用者須自行評估市場風險，並對自身的投資決策負全責。</p>
                    <p>3. <strong>免責範圍</strong>：開發者與本系統營運團隊不對因使用本系統數據而導致的任何直接或間接財務損失承擔法律責任。</p>
                    <p>4. <strong>資料正確性</strong>：系統透過公開渠道（如 Yahoo Finance 等）抓取即時與歷史行情，若遇網路延遲或數據源異常，系統將不負預期中斷之責。</p>
                    <p class="text-zinc-500 italic">【請完整捲動至最底部，方可解鎖同意按鈕（若畫面過大無法捲動將自動解鎖）】</p>
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

                function checkScrollNeed() {
                    if (box.scrollHeight <= box.clientHeight + 10) {
                        scrolledToBottom = true;
                        enableCheckbox();
                    }
                }

                function checkScroll() {
                    if (box.scrollTop + box.clientHeight >= box.scrollHeight - 30) {
                        scrolledToBottom = true;
                        enableCheckbox();
                    }
                }

                function enableCheckbox() {
                    checkbox.disabled = false;
                    checkbox.parentElement.classList.remove('cursor-not-allowed');
                    checkbox.classList.remove('cursor-not-allowed');
                    agreeLabel.classList.remove('cursor-not-allowed');
                    agreeLabel.classList.add('cursor-pointer', 'text-[#d4af37]');
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

                box.addEventListener('scroll', checkScroll);

                window.addEventListener('load', () => {
                    setTimeout(checkScrollNeed, 100);
                });

                window.addEventListener('resize', () => {
                    checkScrollNeed();
                });
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
                <header class="flex justify-between items-center mb-4 border-b border-zinc-800 pb-4">
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

            <!-- 彈出互動視窗 Modal -->
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
                            { symbol: "6239.TW", name: "力成" }, { symbol: "5425.TW", name: "台半" }, { symbol: "3533.TW", name: "嘉澤" },
                            { symbol: "3661.TW", name: "世芯-KY" }, { symbol: "3443.TW", name: "創意" }, { symbol: "5269.TW", name: "祥碩" },
                            { symbol: "4968.TW", name: "立積" }, { symbol: "2449.TW", name: "京元電子" }, { symbol: "6531.TW", name: "愛普*" },
                            { symbol: "3035.TW", name: "智原" }, { symbol: "6271.TW", name: "同欣電" }, { symbol: "8299.TW", name: "群聯" },
                            { symbol: "4938.TW", name: "和碩" }, { symbol: "2324.TW", name: "仁寶" }, { symbol: "3293.TW", name: "鈊象" },
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
                                    <span class="text-xs text-zinc-400">點擊卡片以開啟詳情與預測</span>
                                </div>

                                <div class="flex gap-2 sm:gap-3 mb-6">
                                    <input type="text" id="tickerInput" placeholder="輸入代號 (例: 2330.TW, NVDA)" class="flex-1 px-4 py-2 rounded-lg bg-black border gold-border text-[#d4af37] focus:outline-none placeholder-zinc-600 text-xs sm:text-sm">
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
                        const price = info ? info.price : '載入中...';
                        const isUp = info ? info.is_up : true;
                        const priceColor = isUp ? 'text-rose-500' : 'text-emerald-400';

                        const card = document.createElement('div');
                        card.className = "bg-zinc-950 border gold-border p-4 rounded-xl flex flex-col justify-between hover:bg-zinc-900 transition cursor-pointer shadow-lg";
                        card.onclick = () => openStockModal(ticker, name, price);
                        card.innerHTML = `
                            <div class="flex justify-between items-start">
                                <div>
                                    <div class="text-xs text-zinc-400">${ticker}</div>
                                    <div class="text-lg sm:text-xl font-bold text-white mt-0.5 tracking-wide">${name}</div>
                                </div>
                                <button onclick="event.stopPropagation(); removeStock(${index})" class="text-zinc-500 hover:text-red-400 font-bold text-xl px-2 py-0.5 rounded" title="刪除">&times;</button>
                            </div>
                            <div class="flex justify-between items-end mt-5 pt-3 border-t border-zinc-900">
                                <div>
                                    <span class="text-xs text-zinc-400">最新收盤價：</span>
                                    <span class="text-base sm:text-lg font-bold ${priceColor}">${price}</span>
                                </div>
                                <span class="text-xs gold-text font-bold px-2.5 py-1 rounded bg-black border gold-border">詳情與預測 ▶</span>
                            </div>
                        `;
                        container.appendChild(card);
                    });
                }

                function openStockModal(ticker, name, price) {
                    const modal = document.getElementById('stockModal');
                    const modalContent = document.getElementById('modalContent');
                    
                    modalContent.innerHTML = `
                        <div class="space-y-4">
                            <div>
                                <span class="text-xs text-zinc-400">${ticker}</span>
                                <h3 class="text-xl font-bold gold-text">${name}</h3>
                                <div class="text-sm text-zinc-300 mt-1">最新收盤價：<span class="font-bold text-white text-base">${price}</span></div>
                            </div>
                            <div class="bg-black p-4 rounded-xl border border-zinc-900 text-center">
                                <button onclick="runSinglePrediction('${ticker}', '${name}')" class="gold-bg text-black font-bold px-6 py-2.5 rounded-lg transition shadow-lg cursor-pointer text-sm w-full">
                                    🚀 執行 AI 多空預測分析
                                </button>
                            </div>
                            <div id="modalPredictionResult" class="space-y-3"></div>
                        </div>
                    `;
                    modal.classList.remove('hidden');
                }

                function closeModal() {
                    document.getElementById('stockModal').classList.add('hidden');
                }

                async function runSinglePrediction(ticker, name) {
                    const output = document.getElementById('modalPredictionResult');
                    output.innerHTML = '<div class="text-zinc-400 text-xs py-4 text-center">AI 模型運算分析中，請稍候...</div>';

                    try {
                        const response = await fetch(`/predict/${encodeURIComponent(ticker)}`);
                        const data = await response.json();
                        
                        let htmlContent = '';

                        // 🌟 【測試模式】強制設為 true，確保隨時點擊都能看到黃色提醒框
                        const isMarketTime = true; 

                        if (isMarketTime) {
                            htmlContent += `
                                <div class="p-3 bg-zinc-900 border border-amber-500/50 rounded-xl text-xs text-amber-300 flex items-center gap-2 shadow">
                                    <span>⚠️</span>
                                    <span>台股盤中即時數據計算中，目前顯示的是昨日收盤後的最終預測結果。盤中數據僅供參考，最新盤後預測將於每日 15:30 更新。</span>
                                </div>
                            `;
                        }
                        
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
                                                <span class="font-bold gold-text">${name} (${item.ticker})</span>
                                                <span class="text-xs text-zinc-400">基準日期：${item.date}</span>
                                            </div>
                                            <p class="text-xs sm:text-sm text-zinc-300 leading-relaxed pt-1">
                                                最新收盤價為 <span class="font-bold text-white">${item.latest_close}</span>。
                                                經模型預估，在未來 2 個交易日內，向上最大可能漲幅約為 <span class="${maxColor} font-bold">+${item.predicted_max_return_pct}%</span>，推估高點目標價約落在 <span class="${maxColor} font-bold">${item.estimated_high_price}</span>；
                                                向下風險防守價位則預估約為 <span class="${minColor} font-bold">${item.predicted_min_return_pct}%</span>，下檔支撐約落在 <span class="${minColor} font-bold">${item.estimated_low_price}</span>。
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

                function promptAddStock(symbol, name) {
                    let targetTab = prompt(`請選擇要將「${name} (${symbol})」加入哪一個自選股分頁？\n請輸入數字 1、2、3 或 4：`, "1");
                    if (!targetTab) return;
                    targetTab = targetTab.trim();
                    if (!['1', '2', '3', '4'].includes(targetTab)) {
                        alert('輸入錯誤，請輸入 1 到 4 之間的數字！');
                        return;
                    }

                    let watchlists = getWatchlists();
                    let stocks = watchlists[targetTab];

                    if (stocks.length >= 50) {
                        alert(`「自選股 ${targetTab}」已達上限 50 支！`);
                        return;
                    }
                    if (stocks.includes(symbol)) {
                        alert(`${symbol} 已經存在於「自選股 ${targetTab}」中！`);
                        return;
                    }

                    stocks.push(symbol);
                    watchlists[targetTab] = stocks;
                    saveWatchlists(watchlists);
                    alert(`成功將 ${symbol} (${name}) 加入「自選股 ${targetTab}」！`);
                }

                async function addStock() {
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
                    await renderStocksCards(stocks);
                }

                async function removeStock(index) {
                    let watchlists = getWatchlists();
                    watchlists[currentTab].splice(index, 1);
                    saveWatchlists(watchlists);
                    await renderStocksCards(watchlists[currentTab]);
                }

                renderContent();
            </script>
        </body>
    </html>
    """

@app.get("/prices/{tickers}")
def get_stock_prices(tickers: str, user: str = Depends(verify_session)):
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        if not ticker_list or ticker_list == ['']:
            return {"prices": {}}
        
        data = yf.download(ticker_list, period="5d", interval="1d", auto_adjust=True, progress=False)
        close_df = data['Close']
        if isinstance(close_df, pd.Series):
            close_df = close_df.to_frame(ticker_list[0])
        
        res = {}
        for t in ticker_list:
            if t in close_df.columns:
                s = close_df[t].dropna()
                if len(s) >= 2:
                    curr = float(s.iloc[-1])
                    prev = float(s.iloc[-2])
                    change = curr - prev
                    res[t] = {
                        "price": round(curr, 2),
                        "is_up": change >= 0
                    }
                elif len(s) == 1:
                    curr = float(s.iloc[-1])
                    res[t] = {
                        "price": round(curr, 2),
                        "is_up": True
                    }
        return {"prices": res}
    except Exception as e:
        return {"prices": {}, "error": str(e)}

@app.get("/predict/{tickers}")
def predict_stocks(tickers: str, user: str = Depends(verify_session)):
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        tz = pytz.timezone('Asia/Taipei')
        today_str = datetime.now(tz).strftime('%Y-%m-%d')

        # 檢查快取是否有該標的之今日盤後預測結果
        results = []
        uncached_tickers = []
        
        for t in ticker_list:
            if prediction_cache["date"] == today_str and t in prediction_cache["data"]:
                results.append(prediction_cache["data"][t])
            else:
                uncached_tickers.append(t)

        if not uncached_tickers:
            return {"predictions": results}

        # 針對未快取的標的進行運算
        macro_tickers = ['^VIX', '^GSPC', '^TWII', 'CL=F', 'ES=F', 'NQ=F']
        all_symbols = uncached_tickers + macro_tickers
        
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
        
        for t in uncached_tickers:
            if t not in close_df.columns:
                err_res = {"ticker": t, "error": "找不到此標的資料或代號有誤"}
                results.append(err_res)
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
            
            pred_max = float(reg_high.predict(latest_features)[0])
            pred_min = float(reg_low.predict(latest_features)[0])
            
            item_res = {
                "ticker": t,
                "date": latest_date,
                "latest_close": round(latest_close, 2),
                "predicted_max_return_pct": round(pred_max * 100, 2),
                "predicted_min_return_pct": round(pred_min * 100, 2),
                "estimated_high_price": round(latest_close * (1 + pred_max), 2),
                "estimated_low_price": round(latest_close * (1 + pred_min), 2)
            }
            results.append(item_res)
            
            # 更新快取
            if prediction_cache["date"] != today_str:
                prediction_cache["date"] = today_str
                prediction_cache["data"] = {}
            prediction_cache["data"][t] = item_res
            
        return {"predictions": results}
        
    except Exception as e:
        return {"error": str(e)}