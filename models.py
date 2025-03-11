from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Data(BaseModel):
    title: str
    content: str
    category: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = None

class AnalyzeRequest(BaseModel):
    query: str
    category: Optional[str] = None

class AskAIRequest(BaseModel):
    question: str
    category: Optional[str] = None

class SQLQueryRequest(BaseModel):
    query: str 