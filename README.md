# stock-service

基于 FastAPI + AkShare 的股票分析服务，提供底部暴力 K 线筛选与基础面打分能力。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 启动

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 调用示例

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "bottom-k",
    "symbols": "000001,600519",
    "score": true,
    "scoring": {
      "pe_max": 150,
      "market_cap_min": 100,
      "require_profit": true
    }
  }'
```

不传 `symbols` 时，将自动拉取 A 股列表进行扫描。
