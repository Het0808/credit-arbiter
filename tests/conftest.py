import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.database import Base, get_db
from src.api.main import app
from src.api.models import Application


@pytest.fixture
def db_session():
    """Throwaway in-memory sqlite session - never touches the dev test.db."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def complete_application(db_session):
    application = Application(
        external_id="100001",
        name_contract_type="Cash loans",
        amt_income_total=250000,
        amt_credit=300000,
        amt_annuity=15000,
        days_employed=-3650,
        days_birth=-14200,
        code_gender="F",
        status="COMPLETE",
    )
    db_session.add(application)
    db_session.commit()
    db_session.refresh(application)
    return application


@pytest.fixture
def incomplete_application(db_session):
    application = Application(
        external_id="100009",
        name_contract_type="Cash loans",
        amt_income_total=130000,
        amt_credit=400000,
        amt_annuity=None,
        days_employed=-500,
        status="INCOMPLETE",
        missing_fields="AMT_ANNUITY",
    )
    db_session.add(application)
    db_session.commit()
    db_session.refresh(application)
    return application


@pytest.fixture
def client():
    """FastAPI TestClient wired to its own throwaway in-memory sqlite DB via
    a get_db dependency override - never touches the dev test.db."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    test_client.SessionLocal = TestingSessionLocal  # exposed so tests can seed data directly
    try:
        yield test_client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def auth_headers(client):
    response = client.post(
        "/api/auth/register", json={"email": "underwriter@test.com", "password": "testpass123"}
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def seeded_application(client):
    session = client.SessionLocal()
    application = Application(
        external_id="100001",
        name_contract_type="Cash loans",
        amt_income_total=250000,
        amt_credit=300000,
        amt_annuity=15000,
        days_employed=-3650,
        status="COMPLETE",
    )
    session.add(application)
    session.commit()
    session.refresh(application)
    session.close()
    return application
