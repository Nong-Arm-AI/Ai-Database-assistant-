import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import json
from database import get_data_as_dataframe, get_data_from_database, get_database_schema, execute_sql_query
import httpx
import logging

# ตั้งค่าการบันทึกล็อก
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# โหลดค่าจากไฟล์ .env
load_dotenv()

# สร้าง OpenAI client โดยไม่ใช้ proxies
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

class OpenAIService:
    def __init__(self):
        self.model = "gpt-4o"
        self.max_tokens = 1000
        self.temperature = 0.7
        self.top_p = 1.0
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0
        
        # ใช้ client ที่สร้างไว้แล้วที่ระดับโมดูล
        self.client = client
        
        # ตรวจสอบว่า API key ถูกตั้งค่าหรือไม่
        if not api_key:
            logger.error("ไม่พบ OPENAI_API_KEY ในไฟล์ .env กรุณาตรวจสอบการตั้งค่า")
            
        # คำแนะนำเริ่มต้นสำหรับ AI
        self.default_sql_analysis_prompt = """คุณเป็นผู้เชี่ยวชาญในการวิเคราะห์ข้อมูลและตอบคำถามจากผลลัพธ์ของคำสั่ง SQL
คุณจะได้รับคำถามภาษาธรรมชาติ, คำสั่ง SQL ที่ใช้, และผลลัพธ์จากการ execute คำสั่ง SQL
งานของคุณคือวิเคราะห์ผลลัพธ์และตอบคำถามให้ชัดเจน เข้าใจง่าย และมีรายละเอียดเชิงลึก

คำแนะนำสำคัญ:
1. ตอบเป็นภาษาไทยและให้ข้อมูลที่เป็นประโยชน์ ละเอียด และครบถ้วน
2. ตอบในรูปแบบข้อความที่เป็นธรรมชาติ เหมือนผู้เชี่ยวชาญกำลังอธิบาย ไม่ใช่แค่แสดงตัวเลขหรือข้อมูลดิบ
3. ใช้ภาษาที่เป็นกันเอง เข้าใจง่าย และสุภาพ
4. อธิบายความหมายของข้อมูลที่พบอย่างละเอียด พร้อมให้ข้อสังเกตและข้อมูลเชิงลึก
5. ถ้าเป็นการนับจำนวน ให้ระบุชัดเจนว่านับจากตารางอะไร และอธิบายความสำคัญของตัวเลขนั้น
6. ถ้าเป็นการคำนวณ ให้อธิบายว่าคำนวณอะไร ได้ผลลัพธ์เท่าไร และมีความหมายอย่างไรในบริบทของธุรกิจหรือการใช้งาน
7. ถ้ามีหลายข้อมูล ให้สรุปประเด็นสำคัญให้เข้าใจง่าย และเรียงลำดับความสำคัญ
8. ตอบให้ครบถ้วนและตรงประเด็นกับคำถามที่ถาม
9. เพิ่มการวิเคราะห์เชิงลึกที่อาจเป็นประโยชน์ต่อผู้ใช้ เช่น แนวโน้มที่น่าสนใจ ข้อสังเกตพิเศษ หรือคำแนะนำที่เกี่ยวข้อง"""
        
        # โหลดคำแนะนำจากฐานข้อมูลหรือไฟล์ (ถ้ามี)
        self.load_prompts()
    
    def load_prompts(self):
        """โหลดคำแนะนำจากฐานข้อมูลหรือไฟล์"""
        try:
            # ตรวจสอบว่ามีไฟล์ prompts.json หรือไม่
            if os.path.exists('prompts.json'):
                with open('prompts.json', 'r', encoding='utf-8') as f:
                    prompts = json.load(f)
                    if 'sql_analysis_prompt' in prompts:
                        self.default_sql_analysis_prompt = prompts['sql_analysis_prompt']
                        logger.info("โหลดคำแนะนำสำหรับ AI จากไฟล์ prompts.json สำเร็จ")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการโหลดคำแนะนำ: {str(e)}")
    
    def save_prompts(self, sql_analysis_prompt):
        """บันทึกคำแนะนำลงในไฟล์"""
        try:
            prompts = {'sql_analysis_prompt': sql_analysis_prompt}
            with open('prompts.json', 'w', encoding='utf-8') as f:
                json.dump(prompts, f, ensure_ascii=False, indent=2)
            self.default_sql_analysis_prompt = sql_analysis_prompt
            logger.info("บันทึกคำแนะนำสำหรับ AI ลงในไฟล์ prompts.json สำเร็จ")
            return True
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการบันทึกคำแนะนำ: {str(e)}")
            return False
    
    def prepare_context_from_database(self, category=None):
        """
        ดึงข้อมูลจากฐานข้อมูลและเตรียมข้อมูลสำหรับส่งให้ OpenAI
        """
        df = get_data_as_dataframe(category)
        if df.empty:
            return "ไม่พบข้อมูลในฐานข้อมูล"
        
        # แปลงข้อมูลเป็น JSON เพื่อส่งให้ AI
        data_list = []
        for _, row in df.iterrows():
            data_list.append({
                'title': row['title'],
                'content': row['content'],
                'category': row['category']
            })
        
        return json.dumps(data_list, ensure_ascii=False)
    
    def analyze_data(self, query, category=None, callback=None):
        """
        วิเคราะห์ข้อมูลจากฐานข้อมูลตามคำถามที่ได้รับ
        
        Args:
            query: คำถามที่ต้องการวิเคราะห์
            category: หมวดหมู่ข้อมูล (ไม่บังคับ)
            callback: ฟังก์ชันที่จะถูกเรียกเมื่อได้รับข้อความแต่ละส่วน
        
        Returns:
            ผลการวิเคราะห์
        """
        # ดึงข้อมูลจากฐานข้อมูลในรูปแบบ JSON
        db_data = self.prepare_context_from_database(category)
        
        # สร้าง prompt ในรูปแบบเดียวกับตัวอย่าง JavaScript
        prompt = f"User ถามว่า: {query}\nข้อมูลที่ดึงมาจาก Database: {db_data}\nให้ AI สรุปคำตอบให้สั้นและชัดเจน:"
        
        system_prompt = "คุณเป็นผู้ช่วยที่ช่วยดึงข้อมูลจาก Database"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            if callback:
                # ถ้ามี callback ให้ใช้ stream mode
                full_response = ""
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        callback(content)
                
                return full_response
            else:
                # ถ้าไม่มี callback ให้ใช้ non-stream mode
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stream=False
                )
                return response.choices[0].message.content
        except Exception as e:
            error_message = f"เกิดข้อผิดพลาดในการเชื่อมต่อกับ OpenAI API: {str(e)}"
            logger.error(error_message)
            if callback:
                callback(error_message)
            return error_message
    
    def chat_with_bot(self, user_message, conversation_history=None, callback=None):
        """
        สนทนากับ AI โดยใช้ประวัติการสนทนา
        
        Args:
            user_message: ข้อความจากผู้ใช้
            conversation_history: ประวัติการสนทนา (ไม่บังคับ)
            callback: ฟังก์ชันที่จะถูกเรียกเมื่อได้รับข้อความแต่ละส่วน
        
        Returns:
            ข้อความตอบกลับจาก AI
        """
        if conversation_history is None:
            conversation_history = []
        
        system_prompt = "คุณเป็น AI ผู้ช่วยที่เป็นมิตรและให้ข้อมูลที่เป็นประโยชน์ คุณสามารถตอบคำถามได้ทั้งภาษาไทยและภาษาอังกฤษ"
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # เพิ่มประวัติการสนทนา
        for message in conversation_history:
            messages.append(message)
        
        # เพิ่มข้อความล่าสุดของผู้ใช้
        messages.append({"role": "user", "content": user_message})
        
        try:
            if callback:
                # ถ้ามี callback ให้ใช้ stream mode
                full_response = ""
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        callback(content)
                
                return full_response
            else:
                # ถ้าไม่มี callback ให้ใช้ non-stream mode
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stream=False
                )
                return response.choices[0].message.content
        except Exception as e:
            error_message = f"เกิดข้อผิดพลาดในการเชื่อมต่อกับ OpenAI API: {str(e)}"
            logger.error(error_message)
            if callback:
                callback(error_message)
            return error_message
            
    def ask_ai_with_db_data(self, question, category=None, callback=None):
        """
        ฟังก์ชันใหม่ที่ทำงานคล้ายกับ askAI ในตัวอย่าง JavaScript
        ส่งคำถามและข้อมูลจากฐานข้อมูลไปให้ AI โดยตรง
        
        Args:
            question: คำถามที่ต้องการถาม
            category: หมวดหมู่ข้อมูล (ไม่บังคับ)
            callback: ฟังก์ชันที่จะถูกเรียกเมื่อได้รับข้อความแต่ละส่วน
        
        Returns:
            คำตอบจาก AI
        """
        # ดึงข้อมูลจากฐานข้อมูล
        data = get_data_from_database(category)
        
        # แปลงข้อมูลเป็นรูปแบบที่เหมาะสม
        db_data = []
        for item in data:
            db_data.append({
                'id': item.id,
                'title': item.title,
                'content': item.content,
                'category': item.category
            })
        
        # บันทึกล็อกเพื่อตรวจสอบ
        logger.info(f"จำนวนข้อมูลที่ดึงได้: {len(db_data)}")
        if len(db_data) > 0:
            logger.info(f"ตัวอย่างข้อมูล: {db_data[0]}")
        else:
            logger.warning("ไม่พบข้อมูลในฐานข้อมูล")
        
        # สร้าง prompt
        prompt = f"User ถามว่า: {question}\nข้อมูลที่ดึงมาจาก Database: {json.dumps(db_data, ensure_ascii=False)}\nให้ AI สรุปคำตอบให้สั้นและชัดเจน:"
        
        try:
            if callback:
                # ถ้ามี callback ให้ใช้ stream mode
                full_response = ""
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "คุณเป็นผู้ช่วยที่ช่วยดึงข้อมูลจาก Database"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        callback(content)
                
                return full_response
            else:
                # ถ้าไม่มี callback ให้ใช้ non-stream mode
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "คุณเป็นผู้ช่วยที่ช่วยดึงข้อมูลจาก Database"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                return response.choices[0].message.content
        except Exception as e:
            error_message = f"เกิดข้อผิดพลาดในการเชื่อมต่อกับ OpenAI API: {str(e)}"
            logger.error(error_message)
            if callback:
                callback(error_message)
            return error_message
    
    def generate_sql_from_question(self, question, db_schema, callback=None):
        """
        ฟังก์ชันใหม่สำหรับสร้างคำสั่ง SQL จากคำถามภาษาธรรมชาติ
        
        Args:
            question: คำถามภาษาธรรมชาติ
            db_schema: โครงสร้างฐานข้อมูล
            callback: ฟังก์ชันที่จะถูกเรียกเมื่อได้รับข้อความแต่ละส่วน
        
        Returns:
            คำสั่ง SQL ที่สร้างขึ้น
        """
        # แปลงโครงสร้างฐานข้อมูลเป็น JSON
        schema_json = json.dumps(db_schema, ensure_ascii=False)
        
        # สร้าง prompt
        system_prompt = """คุณเป็นผู้เชี่ยวชาญในการแปลงคำถามภาษาธรรมชาติเป็นคำสั่ง SQL
คุณจะได้รับโครงสร้างฐานข้อมูลและคำถามภาษาธรรมชาติ
งานของคุณคือสร้างคำสั่ง SQL ที่ถูกต้องเพื่อตอบคำถามนั้น
ตอบเฉพาะคำสั่ง SQL เท่านั้น ไม่ต้องมีคำอธิบายหรือข้อความอื่นใด
ห้ามใช้เครื่องหมาย ``` หรือ ```sql ในคำตอบ ให้ตอบเฉพาะคำสั่ง SQL ล้วนๆ เท่านั้น"""
        
        user_prompt = f"""โครงสร้างฐานข้อมูล:
{schema_json}

คำถาม: {question}

คำสั่ง SQL:"""
        
        try:
            if callback:
                # ถ้ามี callback ให้ใช้ stream mode
                full_response = ""
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=500,
                    temperature=0.1,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        callback(content)
                
                # ลบ markdown code block syntax ถ้ามี
                full_response = full_response.replace('```sql', '').replace('```', '').strip()
                return full_response
            else:
                # ถ้าไม่มี callback ให้ใช้ non-stream mode
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=500,
                    temperature=0.1  # ใช้ temperature ต่ำเพื่อให้ได้คำตอบที่แน่นอน
                )
                sql_query = response.choices[0].message.content.strip()
                
                # ลบ markdown code block syntax ถ้ามี
                sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
                
                return sql_query
        except Exception as e:
            error_message = f"เกิดข้อผิดพลาดในการสร้างคำสั่ง SQL: {str(e)}"
            logger.error(error_message)
            if callback:
                callback(error_message)
            return error_message
    
    def analyze_sql_result(self, question, sql_query, result_data, callback=None):
        """
        ฟังก์ชันใหม่สำหรับวิเคราะห์ผลลัพธ์จากการ execute คำสั่ง SQL
        
        Args:
            question: คำถามภาษาธรรมชาติ
            sql_query: คำสั่ง SQL ที่ใช้
            result_data: ผลลัพธ์จากการ execute คำสั่ง SQL
            callback: ฟังก์ชันที่จะถูกเรียกเมื่อได้รับข้อความแต่ละส่วน
        
        Returns:
            การวิเคราะห์ผลลัพธ์
        """
        try:
            # ส่งข้อความว่างเพื่อเริ่มต้น callback
            if callback:
                callback("")
                
            # ตรวจสอบว่ามี client หรือไม่
            if not hasattr(self, 'client') or self.client is None:
                error_message = "ไม่สามารถเชื่อมต่อกับ OpenAI API ได้ (client ไม่ถูกกำหนด)"
                logger.error(error_message)
                if callback:
                    callback(error_message)
                    callback(None)
                return error_message
                
            # แปลงผลลัพธ์เป็น JSON
            result_json = json.dumps(result_data, ensure_ascii=False)
            
            # ใช้คำแนะนำที่กำหนดไว้
            system_prompt = self.default_sql_analysis_prompt
            
            user_prompt = f"""คำถาม: {question}

คำสั่ง SQL ที่ใช้:
{sql_query}

ผลลัพธ์จากการ execute คำสั่ง SQL:
{result_json}

กรุณาวิเคราะห์ผลลัพธ์และตอบคำถามอย่างละเอียด"""

            # ถ้ามี callback ให้ใช้ stream mode
            if callback:
                try:
                    logger.info("เริ่มการวิเคราะห์แบบ streaming")
                    full_response = ""
                    stream = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        stream=True,
                        temperature=0.7
                    )
                    
                    for chunk in stream:
                        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            callback(content)
                    
                    # ส่ง None เพื่อบอกว่าสิ้นสุดการส่งข้อมูล
                    logger.info("สิ้นสุดการวิเคราะห์แบบ streaming")
                    callback(None)
                    return full_response
                except Exception as e:
                    error_message = f"เกิดข้อผิดพลาดในการวิเคราะห์แบบ streaming: {str(e)}"
                    logger.error(error_message)
                    callback(f"ขออภัย เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}")
                    callback(None)
                    return error_message
            else:
                # ถ้าไม่มี callback ให้ใช้ non-stream mode
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.7
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    error_message = f"เกิดข้อผิดพลาดในการวิเคราะห์แบบไม่ streaming: {str(e)}"
                    logger.error(error_message)
                    return error_message
        except Exception as e:
            error_message = f"เกิดข้อผิดพลาดในการวิเคราะห์ผลลัพธ์: {str(e)}"
            logger.error(error_message)
            if callback:
                callback(f"ขออภัย เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}")
                callback(None)
            return error_message
    
    def generate_text_with_stream(self, user_message, callback=None):
        """
        สร้างข้อความแบบ stream (ทีละส่วน) เพื่อแสดงผลแบบ real-time
        
        Args:
            user_message: ข้อความจากผู้ใช้
            callback: ฟังก์ชันที่จะถูกเรียกเมื่อได้รับข้อความแต่ละส่วน
        
        Returns:
            ข้อความทั้งหมดที่สร้างขึ้น
        """
        system_prompt = "คุณเป็น AI ผู้ช่วยที่เป็นมิตรและให้ข้อมูลที่เป็นประโยชน์ คุณสามารถตอบคำถามได้ทั้งภาษาไทยและภาษาอังกฤษ"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        try:
            full_response = ""
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    if callback:
                        callback(content)
            
            return full_response
        except Exception as e:
            error_message = f"เกิดข้อผิดพลาดในการเชื่อมต่อกับ OpenAI API: {str(e)}"
            logger.error(error_message)
            if callback:
                callback(error_message)
            return error_message 