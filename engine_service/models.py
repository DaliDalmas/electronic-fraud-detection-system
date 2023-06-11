import uuid

from sqlalchemy import Column, String, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(String(32), primary_key=True, index=True)
    user_name = Column(String(100), nullable=False)
    email = Column(String(50), nullable=False)
    hash_password = Column(String(512), nullable=False)

    __table_args__ = (UniqueConstraint('email'), UniqueConstraint('user_name'),)

    def __init__(self, user_name, email, hash_password):
        self.id = str(uuid.uuid4().hex)
        self.user_name = user_name
        self.email = email
        self.hash_password = hash_password