import json
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from werkzeug.security import generate_password_hash, check_password_hash

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///lulc.db")
Base = declarative_base()
engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(512), nullable=True)
    full_name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    joined_date = Column(String(50), nullable=True)
    history = relationship("History", back_populates="user", cascade="all, delete-orphan")


class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    input_image = Column(Text, nullable=False)
    output_image = Column(Text, nullable=False)
    stats = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    user = relationship("User", back_populates="history")


def _user_to_dict(user):
    if not user:
        return None
    return {
        "id": user.id,
        "username": user.username,
        "password_hash": user.password_hash,
        "full_name": user.full_name,
        "email": user.email,
        "joined_date": user.joined_date,
    }


def _history_to_dict(item):
    parsed_stats = None
    if item.stats:
        try:
            parsed_stats = json.loads(item.stats)
        except json.JSONDecodeError:
            parsed_stats = None
    return {
        "id": item.id,
        "user_id": item.user_id,
        "input_image": item.input_image,
        "output_image": item.output_image,
        "stats": parsed_stats,
        "timestamp": item.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
    }


def init_db():
    """Initialize DB schema and ensure demo user exists."""
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        existing = session.query(User).filter_by(username="demo_user").first()
        if existing is None:
            demo = User(
                username="demo_user",
                password_hash=generate_password_hash("password123"),
                full_name="LULC Researcher",
                email="researcher@example.com",
                joined_date=datetime.now().strftime("%Y-%m-%d"),
            )
            session.add(demo)
            session.commit()


def add_prediction(user_id, input_image, output_image, stats=None):
    stats_json = json.dumps(stats) if stats else None
    with SessionLocal() as session:
        item = History(
            user_id=user_id,
            input_image=input_image,
            output_image=output_image,
            stats=stats_json,
        )
        session.add(item)
        session.commit()


def get_user_history(user_id):
    with SessionLocal() as session:
        rows = (
            session.query(History)
            .filter_by(user_id=user_id)
            .order_by(History.timestamp.desc())
            .all()
        )
        return [_history_to_dict(row) for row in rows]


def get_history_item(user_id, history_id):
    with SessionLocal() as session:
        row = session.query(History).filter_by(user_id=user_id, id=history_id).first()
        return _history_to_dict(row) if row else None


def get_user(user_id):
    with SessionLocal() as session:
        row = session.query(User).filter_by(id=user_id).first()
        return _user_to_dict(row)


def create_user(username, email, password, full_name=""):
    with SessionLocal() as session:
        existing = session.query(User).filter_by(username=username).first()
        if existing:
            return False
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            full_name=full_name,
            joined_date=datetime.now().strftime("%Y-%m-%d"),
        )
        session.add(user)
        session.commit()
        return True


def verify_user(username, password):
    with SessionLocal() as session:
        user = session.query(User).filter_by(username=username).first()
    if user and user.password_hash and check_password_hash(user.password_hash, password):
        return _user_to_dict(user)
    return None
