from fastapi import FastAPI, HTTPException, Request, Form, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import json
import asyncio
import uvicorn
import logging
from database import get_data_from_database, get_database_schema, execute_sql_query, db_manager
from openai_service import OpenAIService
from models import Data
import decimal
from datetime import datetime, date

# ตั้งค่าการบันทึกล็อก
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# สร้าง JSONEncoder ที่สามารถจัดการกับ Decimal ได้
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super(CustomJSONEncoder, self).default(obj)

# ฟังก์ชันสำหรับแปลงข้อมูลเป็น JSON
def custom_json_dumps(data):
    return json.dumps(data, cls=CustomJSONEncoder, ensure_ascii=False)

app = FastAPI()

# เพิ่ม CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# กำหนด templates directory
templates = Jinja2Templates(directory="templates")

# กำหนด static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# สร้าง OpenAI service
openai_service = OpenAIService()

# สร้าง model สำหรับรับข้อมูล
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
    question: str

class PromptUpdateRequest(BaseModel):
    prompt: str

class DatabaseConnectionRequest(BaseModel):
    db_type: str
    host: str
    port: str
    user: str
    password: str
    database: str
    mongodb_uri: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    response = openai_service.chat_with_bot(
        chat_request.message, 
        chat_request.conversation_history
    )
    return {"response": response}

@app.post("/analyze")
async def analyze(analyze_request: AnalyzeRequest):
    response = openai_service.analyze_data(
        analyze_request.query, 
        analyze_request.category
    )
    return {"response": response}

@app.post("/ask-ai")
async def ask_ai(ask_request: AskAIRequest):
    response = openai_service.ask_ai_with_db_data(
        ask_request.question, 
        ask_request.category
    )
    return {"response": response}

@app.get("/debug/data")
async def get_debug_data(category: Optional[str] = None):
    """
    API endpoint สำหรับดึงข้อมูลดิบจากฐานข้อมูลเพื่อการตรวจสอบ
    """
    data = get_data_from_database(category)
    result = []
    for item in data:
        result.append({
            "id": item.id,
            "title": item.title,
            "content": item.content,
            "category": item.category
        })
    return {"data": result, "count": len(result)}

@app.get("/db/schema")
async def get_schema():
    """
    API endpoint สำหรับดึงโครงสร้างฐานข้อมูล
    """
    schema = get_database_schema()
    return {"schema": schema}

@app.post("/db/query")
async def run_sql_query(query_request: SQLQueryRequest):
    """
    API endpoint สำหรับรันคำสั่ง SQL โดยตรง
    """
    try:
        result = execute_sql_query(query_request.question)
        return {"result": result}
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการรันคำสั่ง SQL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/sql-query")
async def ai_sql_query(query_request: SQLQueryRequest):
    """
    สร้างและรันคำสั่ง SQL จากคำถามภาษาธรรมชาติ
    """
    try:
        # ดึงโครงสร้างฐานข้อมูล
        schema = get_database_schema()
        if not schema:
            raise HTTPException(status_code=500, detail="ไม่สามารถดึงโครงสร้างฐานข้อมูลได้")
            
        # ดึงประเภทฐานข้อมูลปัจจุบัน
        db_type = db_manager.db_type
        logger.info(f"ประเภทฐานข้อมูลที่ใช้: {db_type}")
        
        # สร้างคำสั่ง SQL
        openai_service = OpenAIService()
        sql_query = await openai_service.generate_sql_from_question(query_request.question, schema, db_type)
        
        # รันคำสั่ง SQL
        result = execute_sql_query(sql_query)
        
        # วิเคราะห์ผลลัพธ์
        analysis = await openai_service.analyze_sql_result(query_request.question, sql_query, result, db_type)
        
        # ส่งผลลัพธ์กลับไปยังผู้ใช้
        return {
            "question": query_request.question,
            "sql_query": sql_query,
            "result": result,
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการสร้างและรันคำสั่ง SQL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

@app.get("/stream-chat")
async def stream_chat(message: str):
    """
    API endpoint สำหรับการแชทแบบ streaming
    """
    async def generate():
        # ใช้ asyncio.Queue เพื่อรับข้อความจาก callback
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        
        # สร้าง callback function
        def callback(content):
            asyncio.run_coroutine_threadsafe(queue.put(content), loop)
        
        # เริ่มการสร้างข้อความใน thread แยก
        asyncio.create_task(
            asyncio.to_thread(
                openai_service.generate_text_with_stream,
                message,
                callback
            )
        )
        
        # ส่งข้อความกลับเป็น Server-Sent Events
        while True:
            try:
                content = await queue.get()
                if content:
                    yield f"data: {custom_json_dumps({'content': content})}\n\n"
                else:
                    break
            except Exception as e:
                yield f"data: {custom_json_dumps({'error': str(e)})}\n\n"
                break
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/stream/chat")
async def stream_chat_with_history(chat_request: ChatRequest):
    """
    API endpoint สำหรับการแชทแบบ streaming พร้อมประวัติการสนทนา
    """
    async def generate():
        # ใช้ asyncio.Queue เพื่อรับข้อความจาก callback
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        
        # สร้าง callback function
        def callback(content):
            asyncio.run_coroutine_threadsafe(queue.put(content), loop)
        
        # เริ่มการสร้างข้อความใน thread แยก
        asyncio.create_task(
            asyncio.to_thread(
                openai_service.chat_with_bot,
                chat_request.message,
                chat_request.conversation_history,
                callback
            )
        )
        
        # ส่งข้อความกลับเป็น Server-Sent Events
        while True:
            try:
                content = await queue.get()
                if content:
                    yield f"data: {custom_json_dumps({'content': content})}\n\n"
                else:
                    break
            except Exception as e:
                yield f"data: {custom_json_dumps({'error': str(e)})}\n\n"
                break
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/stream/analyze")
async def stream_analyze(analyze_request: AnalyzeRequest):
    """
    API endpoint สำหรับการวิเคราะห์ข้อมูลแบบ streaming
    """
    async def generate():
        # ใช้ asyncio.Queue เพื่อรับข้อความจาก callback
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        
        # สร้าง callback function
        def callback(content):
            asyncio.run_coroutine_threadsafe(queue.put(content), loop)
        
        # เริ่มการสร้างข้อความใน thread แยก
        asyncio.create_task(
            asyncio.to_thread(
                openai_service.analyze_data,
                analyze_request.query,
                analyze_request.category,
                callback
            )
        )
        
        # ส่งข้อความกลับเป็น Server-Sent Events
        while True:
            try:
                content = await queue.get()
                if content:
                    yield f"data: {custom_json_dumps({'content': content})}\n\n"
                else:
                    break
            except Exception as e:
                yield f"data: {custom_json_dumps({'error': str(e)})}\n\n"
                break
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/stream/ask-ai")
async def stream_ask_ai(ask_request: AskAIRequest):
    """
    API endpoint สำหรับการถาม AI แบบ streaming
    """
    async def generate():
        # ใช้ asyncio.Queue เพื่อรับข้อความจาก callback
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        
        # สร้าง callback function
        def callback(content):
            asyncio.run_coroutine_threadsafe(queue.put(content), loop)
        
        # เริ่มการสร้างข้อความใน thread แยก
        asyncio.create_task(
            asyncio.to_thread(
                openai_service.ask_ai_with_db_data,
                ask_request.question,
                ask_request.category,
                callback
            )
        )
        
        # ส่งข้อความกลับเป็น Server-Sent Events
        while True:
            try:
                content = await queue.get()
                if content:
                    yield f"data: {custom_json_dumps({'content': content})}\n\n"
                else:
                    break
            except Exception as e:
                yield f"data: {custom_json_dumps({'error': str(e)})}\n\n"
                break
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/stream/sql-query")
async def stream_sql_query(query_request: SQLQueryRequest):
    """
    สร้างและรันคำสั่ง SQL จากคำถามภาษาธรรมชาติ และส่งผลลัพธ์แบบ streaming
    """
    question = query_request.question
    logger.info(f"คำถาม SQL: {question}")
    
    async def generate():
        try:
            # ดึงโครงสร้างฐานข้อมูล
            schema = get_database_schema()
            if not schema:
                yield f"data: {json.dumps({'error': 'ไม่สามารถดึงโครงสร้างฐานข้อมูลได้'}, ensure_ascii=False)}\n\n"
                return
                
            # ดึงประเภทฐานข้อมูลปัจจุบัน
            db_type = db_manager.db_type
            logger.info(f"ประเภทฐานข้อมูลที่ใช้: {db_type}")
            
            # แจ้งสถานะการสร้าง SQL
            yield f"data: {json.dumps({'status': 'generating_sql'}, ensure_ascii=False)}\n\n"
            
            # สร้างคำสั่ง SQL
            openai_service = OpenAIService()
            sql_query = await openai_service.generate_sql_from_question(question, schema, db_type)
            
            # ส่งคำสั่ง SQL กลับไปยังผู้ใช้
            yield f"data: {json.dumps({'sql_query': sql_query}, ensure_ascii=False)}\n\n"
            
            # แจ้งสถานะการรันคำสั่ง SQL
            yield f"data: {json.dumps({'status': 'executing_sql'}, ensure_ascii=False)}\n\n"
            
            # รันคำสั่ง SQL
            try:
                result = execute_sql_query(sql_query)
                
                # ส่งผลลัพธ์กลับไปยังผู้ใช้
                yield f"data: {json.dumps({'result': result}, cls=CustomJSONEncoder, ensure_ascii=False)}\n\n"
                
                # แจ้งสถานะการวิเคราะห์ผลลัพธ์
                yield f"data: {json.dumps({'status': 'analyzing_result'}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'analysis_start': True}, ensure_ascii=False)}\n\n"
                
                # วิเคราะห์ผลลัพธ์
                def callback(content):
                    return content
                
                # ใช้ asyncio.Queue เพื่อรับข้อความจาก callback
                queue = asyncio.Queue()
                
                async def analysis_callback(content):
                    await queue.put(content)
                
                # เริ่มการวิเคราะห์ในอีก task หนึ่ง
                analysis_task = asyncio.create_task(
                    openai_service.analyze_sql_result(
                        question, sql_query, result, 
                        callback=analysis_callback
                    )
                )
                
                # รอรับข้อความจาก callback และส่งกลับไปยังผู้ใช้
                timeout = 60  # เพิ่มเวลา timeout เป็น 60 วินาที
                try:
                    while True:
                        try:
                            content = await asyncio.wait_for(queue.get(), timeout=timeout)
                            if content:
                                yield f"data: {json.dumps({'analysis_chunk': content}, ensure_ascii=False)}\n\n"
                            else:
                                # ถ้าได้รับข้อความว่างให้ตรวจสอบว่า task เสร็จสิ้นแล้วหรือไม่
                                if analysis_task.done():
                                    break
                        except asyncio.TimeoutError:
                            # ถ้าเกิด timeout ให้ตรวจสอบว่า task เสร็จสิ้นแล้วหรือไม่
                            if analysis_task.done():
                                break
                            else:
                                logger.warning("เกิด timeout ในการรอข้อความจาก OpenAI API")
                                yield f"data: {json.dumps({'analysis_error': 'เกิด timeout ในการวิเคราะห์ผลลัพธ์'}, ensure_ascii=False)}\n\n"
                                break
                    
                    # แจ้งว่าการวิเคราะห์เสร็จสิ้น
                    yield f"data: {json.dumps({'analysis_complete': True}, ensure_ascii=False)}\n\n"
                    
                except Exception as e:
                    logger.error(f"เกิดข้อผิดพลาดในการวิเคราะห์ผลลัพธ์: {str(e)}")
                    yield f"data: {json.dumps({'analysis_error': str(e)}, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการรันคำสั่ง SQL: {str(e)}")
                yield f"data: {json.dumps({'error': f'เกิดข้อผิดพลาดในการรันคำสั่ง SQL: {str(e)}'}, ensure_ascii=False)}\n\n"
                
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการสร้างคำสั่ง SQL: {str(e)}")
            yield f"data: {json.dumps({'error': f'เกิดข้อผิดพลาดในการสร้างคำสั่ง SQL: {str(e)}'}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/api/prompt")
async def get_prompt():
    """ดึงคำแนะนำสำหรับ AI"""
    try:
        return {"prompt": openai_service.default_sql_analysis_prompt}
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการดึงคำแนะนำ: {str(e)}")
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

@app.post("/api/prompt")
async def update_prompt(prompt_request: PromptUpdateRequest):
    """อัปเดตคำแนะนำสำหรับ AI"""
    try:
        success = openai_service.save_prompts(prompt_request.prompt)
        if success:
            return {"status": "success", "message": "อัปเดตคำแนะนำสำเร็จ"}
        else:
            raise HTTPException(status_code=500, detail="ไม่สามารถบันทึกคำแนะนำได้")
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการอัปเดตคำแนะนำ: {str(e)}")
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

@app.get("/api/db/connection")
async def get_db_connection():
    """ดึงข้อมูลการเชื่อมต่อฐานข้อมูลปัจจุบัน"""
    try:
        # ไม่ส่งคืนรหัสผ่านเพื่อความปลอดภัย
        connection_info = {
            'db_type': db_manager.db_type,
            'host': db_manager.connection_params['host'],
            'port': db_manager.connection_params['port'],
            'user': db_manager.connection_params['user'],
            'database': db_manager.connection_params['database'],
            'mongodb_uri': db_manager.connection_params['mongodb_uri'] if db_manager.connection_params['mongodb_uri'] else None
        }
        return connection_info
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูลการเชื่อมต่อฐานข้อมูล: {str(e)}")
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

@app.post("/api/db/connection")
async def update_db_connection(connection_request: DatabaseConnectionRequest):
    """อัปเดตการเชื่อมต่อฐานข้อมูล"""
    try:
        # อัปเดตการเชื่อมต่อ
        success = db_manager.update_connection(
            connection_request.db_type,
            connection_request.host,
            connection_request.port,
            connection_request.user,
            connection_request.password,
            connection_request.database,
            connection_request.mongodb_uri
        )
        
        if success:
            # บันทึกการตั้งค่าลงในไฟล์ .env
            db_manager.save_connection_to_env()
            return {"status": "success", "message": "อัปเดตการเชื่อมต่อฐานข้อมูลสำเร็จ"}
        else:
            raise HTTPException(status_code=500, detail="ไม่สามารถเชื่อมต่อกับฐานข้อมูลได้")
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการอัปเดตการเชื่อมต่อฐานข้อมูล: {str(e)}")
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

@app.post("/api/db/connection/test")
async def test_db_connection(connection_request: DatabaseConnectionRequest):
    """ทดสอบการเชื่อมต่อฐานข้อมูล"""
    try:
        # สร้าง DatabaseManager ชั่วคราวเพื่อทดสอบการเชื่อมต่อ
        from database import DatabaseManager
        test_manager = DatabaseManager()
        
        # อัปเดตการเชื่อมต่อแต่ไม่บันทึกลงในไฟล์ .env
        success = test_manager.update_connection(
            connection_request.db_type,
            connection_request.host,
            connection_request.port,
            connection_request.user,
            connection_request.password,
            connection_request.database,
            connection_request.mongodb_uri
        )
        
        # ทดสอบการเชื่อมต่อ
        if success:
            connection_test = test_manager.test_connection()
            if connection_test:
                return {"status": "success", "message": "การเชื่อมต่อสำเร็จ"}
            else:
                return {"status": "error", "message": "ไม่สามารถเชื่อมต่อกับฐานข้อมูลได้"}
        else:
            return {"status": "error", "message": "ไม่สามารถเชื่อมต่อกับฐานข้อมูลได้"}
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการทดสอบการเชื่อมต่อฐานข้อมูล: {str(e)}")
        return {"status": "error", "message": f"เกิดข้อผิดพลาด: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 