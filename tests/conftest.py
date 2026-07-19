import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.database import Base, get_db
from src.api.main import app
from src.api.models import Application, Document


@pytest.fixture(autouse=True)
def _disable_real_llm_calls(monkeypatch):
    """Keep the test suite deterministic, fast, and free of live network
    calls / API cost by default - explanation generation (US-208) falls back
    to its template path unless a test explicitly re-enables the Groq client
    via monkeypatch. GROQ_API_KEY is loaded from .env at import time (see
    src/api/database.py), so without this the whole suite would otherwise
    hit the real Groq API on every /assess call."""
    import src.api.services.explanation as explanation_module

    monkeypatch.setattr(explanation_module, "_get_client", lambda: None)


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
        loan_scheme="Personal",
        status="COMPLETE",
        # A genuinely complete profile (matches what "complete" now means
        # post-Part-1: these are the model's strongest predictors, and a
        # fixture missing them scores with low ML confidence by construction,
        # not because the happy path is ambiguous).
        ext_source_1=0.78,
        ext_source_2=0.75,
        ext_source_3=0.72,
        cnt_fam_members=3,
        amt_goods_price=280000,
        cnt_children=1,
        flag_own_car="Y",
        flag_own_realty="Y",
        # Segment values deliberately chosen to fall inside the fairness
        # hard-block's <=5pp band (reports/ml/fairness_thresholds.json) so
        # this "happy path" fixture exercises risk/policy/regulatory logic
        # without incidentally tripping the (separately, directly tested)
        # fairness_check kill-switch.
        name_income_type="Commercial associate",
        name_education_type="Secondary / secondary special",
        name_family_status="Married",
        region_rating_client=2,
        occupation_type="Core staff",
    )
    db_session.add(application)
    db_session.commit()
    db_session.refresh(application)

    # "Complete" spans documents too: seed the Personal scheme's required
    # docs so tests exercising risk/policy/regulatory logic aren't
    # incidentally blocked by the unrelated missing_documents kill-switch
    # (US-301/302, wired into run_assessment in Part 4).
    for doc_type in ["salary_slip", "bank_statement", "id_proof"]:
        db_session.add(
            Document(
                application_id=application.id,
                doc_type=doc_type,
                filename=f"{doc_type}.pdf",
                storage_path=f"/tmp/{doc_type}.pdf",
            )
        )
    db_session.commit()
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
