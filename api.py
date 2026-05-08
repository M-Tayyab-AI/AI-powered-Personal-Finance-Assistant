# pylint: disable=all
# fmt: off
# flake8: noqa

import os, logging
import pandas as pd
import uvicorn
from fastapi import FastAPI,status, HTTPException
from dotenv import load_dotenv
from schema import ChatRequest

from main import chat
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)
# ===== Load env (keys for your tools' internals) =====
load_dotenv()
FINANCIAL_DATA_PATH = r"D:\Work\Personal\AI-powered-Personal-Finance-Assistant\users_financial_data"

# ===== FastAPI app =====
app = FastAPI(
    title="AI-powered Personal Finance Assistant API",
    description="Users personal finance assistant app.",
    version="1.0.0",
)
# ===== Basic health + root =====
@app.get("/")
def root():
    return {"message": "AI-powered Personal Finance Assistant", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "message": "API is running"}

@app.get("/transactions")
async def transactions(req: ChatRequest):
    file_path = os.path.join(FINANCIAL_DATA_PATH, f"user_{req.user_id}.xlsx")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"User {req.user_id} ki file nahi mili")

    # Pandas fastest hai Excel k liye — openpyxl engine use karo
    df = pd.read_excel(file_path)

    # Saara data text mein convert karo
    data_text = df.to_string(index=False)

    return {
        "user_id": req.user_id,
        "query": req.query,
        "transactions": data_text
    }

@app.post("/chat/")
def finance_chatbot(req: ChatRequest):
    try:
       return chat(req)
    except Exception as e:
        raise e

# ===== Run locally =====
if __name__ == "__main__":
    uvicorn.run(
        app,  # run the FastAPI app directly
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False
    )
