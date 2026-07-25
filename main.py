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

app = FastAPI(title="Stock AI Prediction API", version="2.0")

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

# 1. 簡潔好看的登入畫面
@app.get("/login", response_class=HTMLResponse)
def login_page():
    return """
    <html>
        <head><meta charset="utf-8"><title>系統登入</title></head>
        <body style="font-family: sans-serif; text-align: center; margin-top: 100px; background-color: #f4f7f6;">
            <div style="display: inline-block; padding: 40px; background: white; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
                <h2>📈 股市 AI 預測系統登入</h2>
                <form action="/login" method="post">
                    <input type="password" name="password" placeholder="請輸入解鎖密碼" style="padding: 12px; font-size: 16px; width: 280px; border: 1px solid #ccc; border-radius: 5px;" required>
                    <br><br>
                    <button type="submit" style="padding: 12px 24px; font-size: 16px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;">解鎖進入系統</button>
                </form>
            </div>
        </body>
    </html>
    """

# 2. 處理登入並核發通行憑證 (Cookie)
@app.post("/login")
def login(password: str = Form(...)):
    if secrets.compare_digest(password, MY_SECRET_PASSWORD):
        response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        response.set_cookie(key=COOKIE_NAME, value="authenticated", httponly=True)
        return response
    return HTMLResponse("<h3 style='text-align:center; margin-top:50px;'>密碼錯誤！<a href='/login'>點此重試</a></h3>", status_code=401)

# 3. 檢查登入狀態的相依保護函數
def verify_session(cookie: str = Depends(cookie_sec)):
    if cookie != "authenticated":
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login"}
        )

# 4. 首頁（需登入）
@app.get("/", response_class=HTMLResponse)
def home(user: str = Depends(verify_session)):
    return """
    <html>
        <head><meta charset="utf-8"><title>股市 AI 預測系統</title></head>
        <body style="font-family: sans-serif; text-align: center; margin-top: 50px;">
            <h2>🚀 股市多空預測 API 運行中</h2>
            <p>請在網址列輸入查詢路徑，例如：</p>
            <p><a href="/predict/2330.TW,2454.TW,NVDA" target="_blank" style="font-size: 18px; color: #007bff;">/predict/2330.TW,2454.TW,NVDA</a></p>
        </body>
    </html>
    """

# 5. 核心預測 API（需登入）
@app.get("/predict/{tickers}")
def predict_stocks(tickers: str, user: str = Depends(verify_session)):
    """
    支援一次輸入多個股票代號（用逗號隔開，例如：2330.TW,2454.TW,NVDA）
    """
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        macro_tickers = ['^VIX', '^GSPC', '^TWII', 'CL=F', 'ES=F', 'NQ=F']
        
        # 一次下載所有目標標的與總經數據
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
            
            # 即時計算技術指標與特徵
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