# db_session.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db_models import Base

# adjust to your DB
DATABASE_URL = "postgresql+psycopg2://user:password@localhost:5432/lic_bot"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
