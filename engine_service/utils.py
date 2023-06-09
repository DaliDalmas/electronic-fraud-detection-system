from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from passlib.context import CryptContext
from starlette import status

from engine_service.database import get_db_session
from engine_service.models import User
from engine_service.schemas.token import TokenData

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = "3811da59b5dd895ab681ed341fa9bc1e864cbae36c17f4ed4a43db156e7fdf64!"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(user_name):
    with get_db_session() as db_session:
        return db_session.query(User).filter_by(user_name=user_name).first()


def authenticate_user(username: str, password: str):
    with get_db_session() as db_session:
        user: User = db_session.query(User).filter_by(user_name=username).first()
        if not user:
            return False
        if not verify_password(password, user.hash_password):
            return False
        return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(user_name=token_data.username)
    if user is None:
        raise credentials_exception
    return user