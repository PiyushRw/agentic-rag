from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config.settings import settings


class Base(DeclarativeBase):
    """All table classes inherit from this so SQLAlchemy can track them."""


engine = create_engine(
    f"sqlite:///{settings.sqlite_path}",
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    from backend.database import models  # noqa: F401

    settings.ensure_dirs()
    Base.metadata.create_all(engine)