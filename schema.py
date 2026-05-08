from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_id: int
    query: str

class ChatResponse(BaseModel):
    user_id: int
    response: str