from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import concurrent.futures

import akshare as ak
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

import warnings

warnings.filterwarnings("ignore")

app = FastAPI()


class AnalyzeRequest(BaseModel):
    strategy: str
    symbols: str
    notes: Optional[str] = None
    scoring: Optional[Dict[str, Any]] = None
    score: Optional[bool] = True


N = 2  # 倍量要求从4倍降低到2倍，更容易匹配
M = 120
P = 30
PRICE_RANGE = 1.05
LONG_RAISE = 1.03  # 涨幅要求从5%降低到3%，更容易匹配
TURBULENCE = 1.03

NEEDED_DAYS = M + 5
TODAY = datetime.now().strftime("%Y%m%d")
CALENDAR_DAYS_NEEDED = int(NEEDED_DAYS * 1.8)
START_DATE = (datetime.now() - timedelta(days=CALENDAR_DAYS_NEEDED)).strftime(
    "%Y%m%d"
)

MAX_WORKERS = 8
TIMEOUT = 5


def parse_symbols(raw: str) -> List[str]:
    cleaned = raw.replace(",", " ").replace("\n", " ")
    return [item.strip() for item in cleaned.split(" ") if item.strip()]


def get_stock_list_fast() -> pd.DataFrame:
    try:
        try:
            stock_df = ak.stock_zh_a_spot_em()
        except Exception:
            stock_df = ak.stock_info_a_code_name()

        if stock_df.empty:
            return pd.DataFrame(
                {
                    "代码": ["000001", "000002", "600519", "000858", "300750"],
                    "名称": ["平安银行", "万科A", "贵州茅台", "五粮液", "宁德时代"],
                }
            )

        if "名称" in stock_df.columns:
            stock_df = stock_df[
                ~stock_df["名称"].astype(str).str.contains("ST")
            ]

        if "代码" in stock_df.columns:
            stock_df = stock_df[
                stock_df["代码"].str.startswith("60")
                | stock_df["代码"].str.startswith("000")
            ]

        return stock_df
    except Exception:
        return pd.DataFrame()


def get_minimal_stock_data(stock_code: str) -> pd.DataFrame:
    try:
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=START_DATE,
            end_date=TODAY,
            adjust="qfq",
            timeout=TIMEOUT,
        )
        if df.empty or len(df) < M + 5:
            return pd.DataFrame()

        df = df[["日期", "开盘", "最高", "最低", "收盘", "成交量"]]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        if len(df) < M + 5:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def calculate_indicators_exact(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) < M + 5:
        return None

    df = df.copy()
    df["最低价_M"] = df["low"].rolling(window=M, min_periods=M).min()
    df["相对底部"] = df["low"] <= df["最低价_M"] * PRICE_RANGE
    df["HHV_H_P"] = df["high"].rolling(window=P, min_periods=P).max()
    df["LLV_L_P"] = df["low"].rolling(window=P, min_periods=P).min()
    df["震荡幅度计算"] = (df["HHV_H_P"] - df["LLV_L_P"]) / df["LLV_L_P"] * 100
    df["震荡幅度达标"] = df["震荡幅度计算"] <= P
    df["底部位置"] = df["相对底部"] | df["震荡幅度达标"]
    df["涨幅"] = df["close"] / df["close"].shift(1)
    df["长阳"] = (df["涨幅"] > LONG_RAISE) & (df["close"] > df["open"])
    df["前3日最高价"] = df["high"].shift(1).rolling(window=3, min_periods=3).max()
    df["前3日最低价"] = df["low"].shift(1).rolling(window=3, min_periods=3).min()
    df["前3日震荡幅度"] = df["前3日最高价"] / df["前3日最低价"]
    df["突兀"] = df["前3日震荡幅度"] < TURBULENCE
    df["倍量"] = df["volume"] / df["volume"].shift(1) >= N
    df["底部暴力K线"] = (
        df["底部位置"] & df["长阳"] & df["突兀"] & df["倍量"]
    )
    return df


def evaluate_symbol(code: str, name: Optional[str]) -> Optional[dict]:
    try:
        df = get_minimal_stock_data(code)
        if df.empty:
            return None
        df_with = calculate_indicators_exact(df)
        if df_with is None or len(df_with) < 2:
            return None

        latest = df_with.iloc[-1]
        if not latest.get("底部暴力K线", False):
            return None

        volume_ratio = round(
            df_with.iloc[-1]["volume"] / df_with.iloc[-2]["volume"], 2
        )

        return {
            "symbol": code,
            "name": name or code,
            "date": latest["date"].strftime("%Y-%m-%d"),
            "close": round(latest["close"], 2),
            "change_pct": round((latest["涨幅"] - 1) * 100, 2),
            "volume_ratio": volume_ratio,
            "turbulence_pct": round(latest["震荡幅度计算"], 2),
            "min_price_m": round(latest["最低价_M"], 2),
        }
    except Exception:
        # 静默处理错误，避免影响整体处理
        return None


def extract_metric_value(df: pd.DataFrame, keys: List[str]) -> Optional[float]:
    for key in keys:
        if key in df.columns:
            value = df.iloc[-1][key]
            if pd.isna(value):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def find_column_by_keyword(df: pd.DataFrame, keyword: str) -> Optional[str]:
    for col in df.columns:
        if keyword in str(col):
            return col
    return None


def fetch_fundamentals(code: str) -> dict:
    try:
        df = ak.stock_a_indicator_lg(symbol=code)
    except Exception:
        return {"pe_ttm": None, "market_cap_billion": None, "net_profit": None}

    if df.empty:
        return {"pe_ttm": None, "market_cap_billion": None, "net_profit": None}

    pe_value = extract_metric_value(
        df,
        ["pe_ttm", "pe", "市盈率(TTM)", "市盈率"],
    )

    market_value = extract_metric_value(
        df,
        ["total_mv", "总市值", "总市值(万元)", "总市值(亿)"],
    )

    market_cap_billion = None
    if market_value is not None:
        if market_value > 1e5:
            market_cap_billion = market_value / 10000
        else:
            market_cap_billion = market_value

    net_profit = None
    profit_column = find_column_by_keyword(df, "净利润")
    if profit_column:
        try:
            net_profit = float(df.iloc[-1][profit_column])
        except (TypeError, ValueError):
            net_profit = None

    return {
        "pe_ttm": pe_value,
        "market_cap_billion": market_cap_billion,
        "net_profit": net_profit,
    }


def apply_scoring(item: dict, scoring: Dict[str, Any]) -> dict:
    pe_max = scoring.get("pe_max", 150)
    market_cap_min = scoring.get("market_cap_min", 100)
    require_profit = scoring.get("require_profit", True)

    fundamentals = fetch_fundamentals(item["symbol"])
    score = 0
    reasons = []

    pe_value = fundamentals.get("pe_ttm")
    if pe_value is not None and pe_value <= pe_max:
        score += 1
        reasons.append("市盈率达标")

    market_cap = fundamentals.get("market_cap_billion")
    if market_cap is not None and market_cap >= market_cap_min:
        score += 1
        reasons.append("市值达标")

    net_profit = fundamentals.get("net_profit")
    if require_profit:
        if net_profit is not None and net_profit > 0:
            score += 1
            reasons.append("盈利")
    else:
        reasons.append("未强制盈利")

    item.update(
        {
            "score": score,
            "score_reasons": reasons,
            "pe_ttm": pe_value,
            "market_cap_billion": market_cap,
            "net_profit": net_profit,
        }
    )
    return item


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    symbols = parse_symbols(request.symbols)
    code_name_map = {}
    if not symbols:
        stock_df = get_stock_list_fast()
        if stock_df.empty:
            return {"matches": [], "stats": {"total": 0, "matched": 0}}
        code_name_map = dict(zip(stock_df["代码"], stock_df["名称"]))
        symbols = stock_df["代码"].tolist()
    else:
        code_name_map = {code: code for code in symbols}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for item in executor.map(
            lambda code: evaluate_symbol(code, code_name_map.get(code)),
            symbols,
        ):
            if item:
                results.append(item)

    scoring = request.scoring or {
        "pe_max": 150,
        "market_cap_min": 100,
        "require_profit": True,
    }

    scored = results
    if request.score:
        scored = [apply_scoring(item, scoring) for item in results]
        scored.sort(key=lambda x: (x["score"], x["change_pct"]), reverse=True)

    stats = {
        "total": len(symbols),
        "matched": len(results),
    }

    return {
        "strategy": request.strategy,
        "matches": scored,
        "stats": stats,
        "scoring": scoring,
        "score_enabled": bool(request.score),
    }
