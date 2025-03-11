import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, MetaData, Table, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import json
import decimal
import logging
import pymongo
from pymongo import MongoClient
import urllib.parse

# ตั้งค่าการบันทึกล็อก
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# โหลดค่าจากไฟล์ .env
load_dotenv()

# ข้อมูลการเชื่อมต่อฐานข้อมูล
DB_TYPE = os.getenv("DB_TYPE", "mysql")  # ค่าเริ่มต้นเป็น mysql
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
MONGODB_URI = os.getenv("MONGODB_URI")

# สร้าง Base class สำหรับ SQLAlchemy
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

# คลาสสำหรับจัดการการเชื่อมต่อฐานข้อมูล
class DatabaseManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        self.db_type = DB_TYPE
        self.engine = None
        self.session_local = None
        self.mongo_client = None
        self.mongo_db = None
        self.connection_params = {
            'host': DB_HOST,
            'port': DB_PORT,
            'user': DB_USER,
            'password': DB_PASSWORD,
            'database': DB_NAME,
            'mongodb_uri': MONGODB_URI
        }
        self.connect()
    
    def connect(self):
        """เชื่อมต่อกับฐานข้อมูลตามประเภทที่กำหนด"""
        try:
            if self.db_type.lower() == 'mysql':
                self._connect_mysql()
            elif self.db_type.lower() == 'postgresql':
                self._connect_postgresql()
            elif self.db_type.lower() == 'mongodb':
                self._connect_mongodb()
            else:
                logger.error(f"ไม่รองรับฐานข้อมูลประเภท {self.db_type}")
                raise ValueError(f"ไม่รองรับฐานข้อมูลประเภท {self.db_type}")
            
            logger.info(f"เชื่อมต่อกับฐานข้อมูล {self.db_type} สำเร็จ")
            return True
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อกับฐานข้อมูล: {str(e)}")
            return False
    
    def _connect_mysql(self):
        """เชื่อมต่อกับฐานข้อมูล MySQL"""
        connection_string = f"mysql+pymysql://{self.connection_params['user']}:{self.connection_params['password']}@{self.connection_params['host']}:{self.connection_params['port']}/{self.connection_params['database']}"
        self.engine = create_engine(connection_string)
        self.session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def _connect_postgresql(self):
        """เชื่อมต่อกับฐานข้อมูล PostgreSQL"""
        connection_string = f"postgresql+psycopg2://{self.connection_params['user']}:{self.connection_params['password']}@{self.connection_params['host']}:{self.connection_params['port']}/{self.connection_params['database']}"
        self.engine = create_engine(connection_string)
        self.session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def _connect_mongodb(self):
        """เชื่อมต่อกับฐานข้อมูล MongoDB"""
        if self.connection_params['mongodb_uri']:
            self.mongo_client = MongoClient(self.connection_params['mongodb_uri'])
        else:
            # สร้าง URI สำหรับ MongoDB
            username = urllib.parse.quote_plus(self.connection_params['user'])
            password = urllib.parse.quote_plus(self.connection_params['password'])
            connection_string = f"mongodb://{username}:{password}@{self.connection_params['host']}:{self.connection_params['port']}"
            self.mongo_client = MongoClient(connection_string)
        
        self.mongo_db = self.mongo_client[self.connection_params['database']]
    
    def update_connection(self, db_type, host, port, user, password, database, mongodb_uri=None):
        """อัปเดตการเชื่อมต่อฐานข้อมูล"""
        self.db_type = db_type
        self.connection_params = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'database': database,
            'mongodb_uri': mongodb_uri
        }
        
        # ปิดการเชื่อมต่อเดิม
        self.close_connection()
        
        # เชื่อมต่อใหม่
        return self.connect()
    
    def close_connection(self):
        """ปิดการเชื่อมต่อฐานข้อมูล"""
        try:
            if self.mongo_client:
                self.mongo_client.close()
                self.mongo_client = None
                self.mongo_db = None
            
            # สำหรับ SQLAlchemy ไม่จำเป็นต้องปิด engine โดยตรง
            # แต่เราจะตั้งค่าเป็น None เพื่อให้สามารถสร้างใหม่ได้
            self.engine = None
            self.session_local = None
            
            logger.info("ปิดการเชื่อมต่อฐานข้อมูลเรียบร้อย")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการปิดการเชื่อมต่อฐานข้อมูล: {str(e)}")
    
    def get_session(self):
        """สร้างและคืนค่า session สำหรับ SQLAlchemy"""
        if not self.session_local:
            raise ValueError("ยังไม่ได้เชื่อมต่อกับฐานข้อมูล SQL")
        return self.session_local()
    
    def test_connection(self):
        """ทดสอบการเชื่อมต่อฐานข้อมูล"""
        try:
            if self.db_type.lower() in ['mysql', 'postgresql']:
                if not self.engine:
                    return False
                
                # ทดสอบการเชื่อมต่อโดยการสร้าง connection
                with self.engine.connect() as connection:
                    # ทดสอบด้วยคำสั่ง SQL ง่ายๆ
                    if self.db_type.lower() == 'mysql':
                        result = connection.execute(text("SELECT 1"))
                    else:  # postgresql
                        result = connection.execute(text("SELECT 1"))
                    
                    # ถ้าไม่มีข้อผิดพลาด แสดงว่าเชื่อมต่อสำเร็จ
                    return True
            
            elif self.db_type.lower() == 'mongodb':
                if not self.mongo_client:
                    return False
                
                # ทดสอบการเชื่อมต่อโดยการเรียกดูข้อมูลฐานข้อมูล
                self.mongo_client.server_info()
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"การทดสอบการเชื่อมต่อล้มเหลว: {str(e)}")
            return False
    
    def save_connection_to_env(self):
        """บันทึกการตั้งค่าการเชื่อมต่อลงในไฟล์ .env"""
        try:
            # อ่านไฟล์ .env เดิม
            env_content = {}
            if os.path.exists('.env'):
                with open('.env', 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_content[key] = value
            
            # อัปเดตค่าการเชื่อมต่อ
            env_content['DB_TYPE'] = self.db_type
            env_content['DB_HOST'] = self.connection_params['host']
            env_content['DB_PORT'] = self.connection_params['port']
            env_content['DB_USER'] = self.connection_params['user']
            env_content['DB_PASSWORD'] = self.connection_params['password']
            env_content['DB_NAME'] = self.connection_params['database']
            
            if self.connection_params['mongodb_uri']:
                env_content['MONGODB_URI'] = self.connection_params['mongodb_uri']
            
            # เขียนไฟล์ .env ใหม่
            with open('.env', 'w', encoding='utf-8') as f:
                for key, value in env_content.items():
                    f.write(f"{key}={value}\n")
            
            logger.info("บันทึกการตั้งค่าการเชื่อมต่อลงในไฟล์ .env สำเร็จ")
            return True
        
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการบันทึกการตั้งค่าการเชื่อมต่อ: {str(e)}")
            return False

# สร้าง instance ของ DatabaseManager
db_manager = DatabaseManager()

# ฟังก์ชันสำหรับดึงข้อมูลจากฐานข้อมูล
def get_data_from_database(category=None):
    """ดึงข้อมูลจากฐานข้อมูล"""
    if db_manager.db_type.lower() in ['mysql', 'postgresql']:
        return _get_data_from_sql(category)
    elif db_manager.db_type.lower() == 'mongodb':
        return _get_data_from_mongodb(category)
    else:
        logger.error(f"ไม่รองรับฐานข้อมูลประเภท {db_manager.db_type}")
        return []

def _get_data_from_sql(category=None):
    """ดึงข้อมูลจากฐานข้อมูล SQL"""
    db = db_manager.get_session()
    try:
        query = db.query(DataSource)
        if category:
            query = query.filter(DataSource.category == category)
        data = query.all()
        return [
            {
                'id': item.id,
                'title': item.title,
                'content': item.content,
                'category': item.category,
                'created_at': item.created_at,
                'updated_at': item.updated_at
            } for item in data
        ]
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูลจาก SQL: {str(e)}")
        return []
    finally:
        db.close()

def _get_data_from_mongodb(category=None):
    """ดึงข้อมูลจากฐานข้อมูล MongoDB"""
    try:
        collection = db_manager.mongo_db['data_source']
        query = {}
        if category:
            query['category'] = category
        
        data = list(collection.find(query))
        
        # แปลง ObjectId เป็น string
        for item in data:
            if '_id' in item:
                item['id'] = str(item['_id'])
                del item['_id']
        
        return data
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูลจาก MongoDB: {str(e)}")
        return []

# ฟังก์ชันสำหรับดึงข้อมูลในรูปแบบ DataFrame
def get_data_as_dataframe(category=None):
    """ดึงข้อมูลในรูปแบบ DataFrame"""
    data = get_data_from_database(category)
    df = pd.DataFrame(data)
    return df

# ฟังก์ชันสำหรับดึงโครงสร้างฐานข้อมูล
def get_database_schema():
    """ดึงโครงสร้างฐานข้อมูล"""
    if db_manager.db_type.lower() in ['mysql', 'postgresql']:
        return _get_sql_schema()
    elif db_manager.db_type.lower() == 'mongodb':
        return _get_mongodb_schema()
    else:
        logger.error(f"ไม่รองรับฐานข้อมูลประเภท {db_manager.db_type}")
        return {}

def _get_sql_schema():
    """ดึงโครงสร้างฐานข้อมูล SQL"""
    try:
        inspector = inspect(db_manager.engine)
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
            pk_constraint = inspector.get_pk_constraint(table_name)
            pk_columns = pk_constraint.get('constrained_columns', []) if pk_constraint else []
            
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
        logger.error(f"เกิดข้อผิดพลาดในการดึงโครงสร้างฐานข้อมูล SQL: {str(e)}")
        return {}

def _get_mongodb_schema():
    """ดึงโครงสร้างฐานข้อมูล MongoDB"""
    try:
        schema = {}
        
        # ดึงรายชื่อ collections
        collections = db_manager.mongo_db.list_collection_names()
        
        for collection_name in collections:
            # ดึงตัวอย่างเอกสารเพื่อวิเคราะห์โครงสร้าง
            sample = db_manager.mongo_db[collection_name].find_one()
            
            if sample:
                # วิเคราะห์โครงสร้างจากตัวอย่างเอกสาร
                fields = []
                for field, value in sample.items():
                    if field != '_id':  # ข้ามฟิลด์ _id
                        fields.append({
                            'name': field,
                            'type': type(value).__name__
                        })
                
                schema[collection_name] = {
                    'fields': fields
                }
            else:
                schema[collection_name] = {
                    'fields': []
                }
        
        return schema
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการดึงโครงสร้างฐานข้อมูล MongoDB: {str(e)}")
        return {}

# ฟังก์ชันสำหรับ execute คำสั่ง SQL หรือ MongoDB query
def execute_sql_query(query):
    """Execute คำสั่ง SQL หรือ MongoDB query"""
    if db_manager.db_type.lower() in ['mysql', 'postgresql']:
        return _execute_sql(query)
    elif db_manager.db_type.lower() == 'mongodb':
        return _execute_mongodb_query(query)
    else:
        error_message = f"ไม่รองรับฐานข้อมูลประเภท {db_manager.db_type}"
        logger.error(error_message)
        raise Exception(error_message)

def _execute_sql(sql_query):
    """Execute คำสั่ง SQL"""
    db = db_manager.get_session()
    try:
        logger.info(f"กำลัง execute คำสั่ง SQL: {sql_query}")
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
            
            logger.info(f"พบข้อมูล {len(rows)} รายการ")
            return rows
        else:
            # สำหรับคำสั่ง INSERT, UPDATE, DELETE
            db.commit()
            return {"message": "คำสั่ง SQL ทำงานสำเร็จ"}
    except Exception as e:
        db.rollback()
        error_message = f"เกิดข้อผิดพลาดในการ execute คำสั่ง SQL: {str(e)}"
        logger.error(error_message)
        raise Exception(error_message)
    finally:
        db.close()

def _execute_mongodb_query(query_str):
    """Execute MongoDB query ในรูปแบบ JSON string"""
    try:
        # แปลง query string เป็น dict
        query = json.loads(query_str)
        
        # ตรวจสอบว่ามี collection หรือไม่
        if 'collection' not in query:
            raise ValueError("ต้องระบุ 'collection' ในคำสั่ง MongoDB")
        
        collection_name = query['collection']
        collection = db_manager.mongo_db[collection_name]
        
        # ตรวจสอบประเภทของคำสั่ง
        if 'find' in query:
            # คำสั่ง find
            filter_query = query.get('find', {})
            projection = query.get('projection', None)
            limit = query.get('limit', 0)
            
            cursor = collection.find(filter_query, projection)
            if limit > 0:
                cursor = cursor.limit(limit)
            
            result = list(cursor)
            
            # แปลง ObjectId เป็น string
            for item in result:
                if '_id' in item:
                    item['_id'] = str(item['_id'])
            
            return result
        
        elif 'insert' in query:
            # คำสั่ง insert
            documents = query['insert']
            if isinstance(documents, list):
                result = collection.insert_many(documents)
                return {"inserted_ids": [str(id) for id in result.inserted_ids]}
            else:
                result = collection.insert_one(documents)
                return {"inserted_id": str(result.inserted_id)}
        
        elif 'update' in query:
            # คำสั่ง update
            filter_query = query.get('filter', {})
            update_data = query['update']
            
            if query.get('many', False):
                result = collection.update_many(filter_query, update_data)
                return {"matched_count": result.matched_count, "modified_count": result.modified_count}
            else:
                result = collection.update_one(filter_query, update_data)
                return {"matched_count": result.matched_count, "modified_count": result.modified_count}
        
        elif 'delete' in query:
            # คำสั่ง delete
            filter_query = query['delete']
            
            if query.get('many', False):
                result = collection.delete_many(filter_query)
                return {"deleted_count": result.deleted_count}
            else:
                result = collection.delete_one(filter_query)
                return {"deleted_count": result.deleted_count}
        
        elif 'aggregate' in query:
            # คำสั่ง aggregate
            pipeline = query['aggregate']
            result = list(collection.aggregate(pipeline))
            
            # แปลง ObjectId เป็น string
            for item in result:
                if '_id' in item:
                    item['_id'] = str(item['_id'])
            
            return result
        
        else:
            raise ValueError("ไม่รองรับคำสั่ง MongoDB นี้")
    
    except json.JSONDecodeError:
        error_message = "รูปแบบ JSON ไม่ถูกต้อง"
        logger.error(error_message)
        raise Exception(error_message)
    except Exception as e:
        error_message = f"เกิดข้อผิดพลาดในการ execute คำสั่ง MongoDB: {str(e)}"
        logger.error(error_message)
        raise Exception(error_message)

# เมื่อรันไฟล์นี้โดยตรง
if __name__ == "__main__":
    print("เริ่มต้นโมดูลฐานข้อมูล") 