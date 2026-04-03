"""
SQLAlchemy ORM models for the AI Writing Assistant platform.
  - AuditLog: records every gateway request for compliance and analytics.
"""
import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text
from .db import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trace_id    = Column(String(64),  nullable=False, index=True)
    user_id     = Column(String(128), nullable=True,  index=True)
    method      = Column(String(10),  nullable=False)
    path        = Column(String(255), nullable=False)
    status_code = Column(Integer,     nullable=False)
    duration_ms = Column(Integer,     nullable=True)
    created_at  = Column(DateTime,    nullable=False, default=datetime.datetime.utcnow)

    def __repr__(self) -> str:
        return f"<AuditLog {self.method} {self.path} {self.status_code}>"
