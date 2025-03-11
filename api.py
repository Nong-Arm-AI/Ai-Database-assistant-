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
from database import get_data_from_database, get_database_schema, execute_sql_query
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
    API endpoint สำหรับให้ AI สร้างและรันคำสั่ง SQL จากคำถามภาษาธรรมชาติ
    """
    try:
        # ดึงโครงสร้างฐานข้อมูล
        db_schema = get_database_schema()
        
        # ให้ AI สร้างคำสั่ง SQL
        sql_query = openai_service.generate_sql_from_question(query_request.question, db_schema)
        
        # ตรวจสอบว่าคำสั่ง SQL ถูกสร้างขึ้นหรือไม่
        if sql_query.startswith("เกิดข้อผิดพลาด"):
            return {"error": sql_query}
        
        # รันคำสั่ง SQL
        result = execute_sql_query(sql_query)
        
        # ให้ AI วิเคราะห์ผลลัพธ์
        analysis = openai_service.analyze_sql_result(query_request.question, sql_query, result)
        
        return {
            "question": query_request.question,
            "sql_query": sql_query,
            "result": result,
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการสร้างและรันคำสั่ง SQL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    API endpoint สำหรับการสร้างและรันคำสั่ง SQL แบบ streaming
    """
    async def generate():
        try:
            # ดึงโครงสร้างฐานข้อมูล
            logger.info(f"กำลังดึงโครงสร้างฐานข้อมูลสำหรับคำถาม: {query_request.question}")
            db_schema = get_database_schema()
            
            # ใช้ asyncio.Queue เพื่อรับข้อความจาก callback
            queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            
            # สร้าง callback function
            def callback(content):
                if content is None:
                    logger.info("ได้รับสัญญาณสิ้นสุดการส่งข้อมูล")
                elif content == "":
                    logger.info("ได้รับสัญญาณเริ่มต้นการส่งข้อมูล")
                else:
                    logger.info(f"ได้รับการวิเคราะห์: {content[:50]}...")  # ล็อกเพื่อตรวจสอบ
                asyncio.run_coroutine_threadsafe(queue.put(content), loop)
            
            # ส่งข้อความแจ้งสถานะ
            yield f"data: {custom_json_dumps({'status': 'generating_sql'})}\n\n"
            
            # ให้ AI สร้างคำสั่ง SQL
            logger.info(f"กำลังสร้างคำสั่ง SQL สำหรับคำถาม: {query_request.question}")
            sql_query = await asyncio.to_thread(
                openai_service.generate_sql_from_question,
                query_request.question,
                db_schema
            )
            
            logger.info(f"สร้างคำสั่ง SQL: {sql_query}")  # ล็อกคำสั่ง SQL
            
            # ตรวจสอบว่าคำสั่ง SQL ถูกสร้างขึ้นหรือไม่
            if sql_query.startswith("เกิดข้อผิดพลาด"):
                logger.error(f"เกิดข้อผิดพลาดในการสร้างคำสั่ง SQL: {sql_query}")
                yield f"data: {custom_json_dumps({'error': sql_query})}\n\n"
                return
            
            # ส่งคำสั่ง SQL กลับไปยังผู้ใช้
            yield f"data: {custom_json_dumps({'sql_query': sql_query})}\n\n"
            
            # ส่งข้อความแจ้งสถานะ
            yield f"data: {custom_json_dumps({'status': 'executing_sql'})}\n\n"
            
            # รันคำสั่ง SQL
            try:
                logger.info(f"กำลังรันคำสั่ง SQL: {sql_query}")
                result = execute_sql_query(sql_query)
                if isinstance(result, list) and len(result) > 0:
                    logger.info(f"ผลลัพธ์ SQL: {str(result)[:100]}...")  # ล็อกผลลัพธ์
                else:
                    logger.warning(f"ผลลัพธ์ SQL เป็นค่าว่างหรือไม่ใช่รูปแบบที่คาดหวัง: {str(result)}")
                
                # ส่งผลลัพธ์กลับไปยังผู้ใช้
                yield f"data: {custom_json_dumps({'result': result})}\n\n"
                
                # ส่งข้อความแจ้งสถานะ
                yield f"data: {custom_json_dumps({'status': 'analyzing_result'})}\n\n"
                
                # วิเคราะห์ผลลัพธ์
                logger.info(f"เริ่มวิเคราะห์ผลลัพธ์สำหรับคำถาม: {query_request.question}")
                
                # เรียกใช้ analyze_sql_result แบบ async
                await asyncio.to_thread(
                    openai_service.analyze_sql_result,
                    query_request.question,
                    sql_query,
                    result,
                    callback
                )
                
                # ส่งการวิเคราะห์กลับเป็น Server-Sent Events
                content_received = False
                timeout_count = 0
                max_timeout_count = 60  # รอสูงสุด 60 วินาที
                
                while True:
                    try:
                        content = await asyncio.wait_for(queue.get(), timeout=1.0)
                        if content is None:  # สิ้นสุดการส่งข้อมูล
                            logger.info("ได้รับสัญญาณสิ้นสุดการส่งข้อมูลการวิเคราะห์")
                            break
                        elif content == "":  # เริ่มต้นการส่งข้อมูล
                            logger.info("ได้รับสัญญาณเริ่มต้นการส่งข้อมูลการวิเคราะห์")
                            # ส่งข้อความเริ่มต้นเพื่อให้ UI รู้ว่าเริ่มการวิเคราะห์แล้ว
                            yield f"data: {custom_json_dumps({'analysis_start': True})}\n\n"
                            continue
                        elif content:
                            content_received = True
                            logger.info(f"ส่งการวิเคราะห์กลับไปยังผู้ใช้: {content[:50]}...")  # ล็อกการส่งกลับ
                            # ส่งข้อความแต่ละชิ้นทันทีที่ได้รับ
                            yield f"data: {custom_json_dumps({'analysis_chunk': content})}\n\n"
                        else:
                            logger.warning("ได้รับข้อความว่างจาก queue")
                    except asyncio.TimeoutError:
                        timeout_count += 1
                        logger.warning(f"รอการวิเคราะห์... ({timeout_count}/{max_timeout_count})")
                        if timeout_count >= max_timeout_count:
                            logger.warning("หมดเวลารอการวิเคราะห์")
                            if not content_received:
                                error_msg = 'ขออภัย ไม่สามารถวิเคราะห์ผลลัพธ์ได้ในขณะนี้ โปรดลองอีกครั้งในภายหลัง'
                                logger.error(f"ไม่ได้รับการวิเคราะห์ภายในเวลาที่กำหนด: {error_msg}")
                                yield f"data: {custom_json_dumps({'analysis_error': error_msg})}\n\n"
                            break
                
                # ส่งสัญญาณว่าการวิเคราะห์เสร็จสิ้น
                if content_received:
                    yield f"data: {custom_json_dumps({'analysis_complete': True})}\n\n"
                # ตรวจสอบว่าได้รับการวิเคราะห์หรือไม่
                elif not content_received:
                    logger.warning("ไม่ได้รับการวิเคราะห์จาก OpenAI API")
                    yield f"data: {custom_json_dumps({'analysis_error': 'ขออภัย ไม่สามารถวิเคราะห์ผลลัพธ์ได้ในขณะนี้ โปรดลองอีกครั้งในภายหลัง'})}\n\n"
            except Exception as e:
                error_message = f"เกิดข้อผิดพลาดในการรันคำสั่ง SQL หรือวิเคราะห์ผลลัพธ์: {str(e)}"
                logger.error(error_message)
                yield f"data: {custom_json_dumps({'error': error_message})}\n\n"
        except Exception as e:
            error_message = f"เกิดข้อผิดพลาดในการสร้างและรันคำสั่ง SQL แบบ streaming: {str(e)}"
            logger.error(error_message)
            yield f"data: {custom_json_dumps({'error': error_message})}\n\n"
    
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

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 