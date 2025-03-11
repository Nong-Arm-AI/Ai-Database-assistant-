import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, MetaData, Table, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import json
import decimal

# โหลดค่าจากไฟล์ .env
load_dotenv()

# ข้อมูลการเชื่อมต่อฐานข้อมูล
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# สร้าง connection string
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# สร้าง engine
engine = create_engine(DATABASE_URL)

# สร้าง session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# สร้าง Base class
Base = declarative_base()

# โมเดลสำหรับตารางข้อมูล
class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), index=True)
    user_message = Column(Text)
    bot_response = Column(Text)
    timestamp = Column(DateTime, default=datetime.now)

class DataSource(Base):
    __tablename__ = "data_source"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255))
    content = Column(Text)
    category = Column(String(100))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

# ฟังก์ชันสำหรับดึงข้อมูลจากฐานข้อมูล
def get_data_from_database(category=None):
    db = SessionLocal()
    try:
        print(f"กำลังดึงข้อมูลจากฐานข้อมูล, หมวดหมู่: {category}")
        query = db.query(DataSource)
        if category:
            query = query.filter(DataSource.category == category)
        data = query.all()
        print(f"พบข้อมูล {len(data)} รายการ")
        return data
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")
        return []
    finally:
        db.close()

# ฟังก์ชันสำหรับแปลงข้อมูลเป็น DataFrame
def get_data_as_dataframe(category=None):
    data = get_data_from_database(category)
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame([{
        'id': item.id,
        'title': item.title,
        'content': item.content,
        'category': item.category,
        'created_at': item.created_at,
        'updated_at': item.updated_at
    } for item in data])
    
    return df

# ฟังก์ชันใหม่สำหรับดึงโครงสร้างฐานข้อมูล
def get_database_schema():
    try:
        inspector = inspect(engine)
        schema = {}
        
        for table_name in inspector.get_table_names():
            columns = []
            for column in inspector.get_columns(table_name):
                columns.append({
                    'name': column['name'],
                    'type': str(column['type']),
                    'nullable': column['nullable']
                })
            
            # ดึง primary key columns
            pk_columns = []
            for pk in inspector.get_pk_constraint(table_name).get('constrained_columns', []):
                pk_columns.append(pk)
            
            # ดึง foreign keys
            foreign_keys = []
            for fk in inspector.get_foreign_keys(table_name):
                foreign_keys.append({
                    'referred_table': fk['referred_table'],
                    'referred_columns': fk['referred_columns'],
                    'constrained_columns': fk['constrained_columns']
                })
            
            schema[table_name] = {
                'columns': columns,
                'primary_keys': pk_columns,
                'foreign_keys': foreign_keys
            }
        
        return schema
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการดึงโครงสร้างฐานข้อมูล: {e}")
        return {}

# ฟังก์ชันใหม่สำหรับ execute คำสั่ง SQL โดยตรง
def execute_sql_query(sql_query):
    db = SessionLocal()
    try:
        print(f"กำลัง execute คำสั่ง SQL: {sql_query}")
        result = db.execute(text(sql_query))
        
        if sql_query.strip().upper().startswith(('SELECT', 'SHOW')):
            # สำหรับคำสั่ง SELECT หรือ SHOW
            columns = result.keys()
            rows = []
            for row in result:
                # แปลงค่า Decimal เป็น float
                processed_row = {}
                for i, column in enumerate(columns):
                    value = row[i]
                    if isinstance(value, decimal.Decimal):
                        processed_row[column] = float(value)
                    else:
                        processed_row[column] = value
                rows.append(processed_row)
            
            print(f"พบข้อมูล {len(rows)} รายการ")
            return rows
        else:
            # สำหรับคำสั่ง INSERT, UPDATE, DELETE
            db.commit()
            return {"message": "คำสั่ง SQL ทำงานสำเร็จ"}
    except Exception as e:
        db.rollback()
        error_message = f"เกิดข้อผิดพลาดในการ execute คำสั่ง SQL: {str(e)}"
        print(error_message)
        raise Exception(error_message)
    finally:
        db.close()

# เมื่อรันไฟล์นี้โดยตรง
if __name__ == "__main__":
    print("เริ่มต้นโมดูลฐานข้อมูล") 