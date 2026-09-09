"""Shared SQLAlchemy engine. Retrieval and ingest both import this."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from _core.config import settings

engine = create_engine(settings.database_url, echo=False)

__all__ = ["engine", "Session"]
