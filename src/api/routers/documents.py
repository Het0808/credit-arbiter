from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import Application, Document, User
from ..schemas import DocumentOut, DocumentVerificationResponse
from ..services.document_service import UnsupportedDocumentType, store_document, verify_documents

router = APIRouter(prefix="/applications", tags=["documents"])


def _get_application(db: Session, application_id: int) -> Application:
    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    return application


@router.post("/{application_id}/documents", response_model=DocumentOut)
async def upload_document(
    application_id: int,
    doc_type: str = Form(...),
    declared_name: Optional[str] = Form(None),
    declared_income: Optional[float] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    application = _get_application(db, application_id)
    content = await file.read()
    try:
        document = store_document(
            db,
            application,
            doc_type=doc_type,
            filename=file.filename,
            content_type=file.content_type,
            content=content,
            declared_name=declared_name,
            declared_income=declared_income,
        )
    except UnsupportedDocumentType as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return document


@router.get("/{application_id}/documents", response_model=list[DocumentOut])
def list_documents(
    application_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_application(db, application_id)
    return (
        db.query(Document)
        .filter(Document.application_id == application_id)
        .order_by(Document.uploaded_at.desc())
        .all()
    )


@router.get("/{application_id}/documents/verify", response_model=DocumentVerificationResponse)
def verify_application_documents(
    application_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    application = _get_application(db, application_id)
    return verify_documents(db, application)
