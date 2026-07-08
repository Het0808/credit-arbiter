from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import Application, Document, User
from ..schemas import DocumentOut, DocumentVerificationOut
from ..services.document_verification import (
    is_supported_doc_type,
    is_supported_file,
    verify_documents,
)

router = APIRouter(prefix="/applications", tags=["documents"])

UPLOAD_ROOT = Path(__file__).resolve().parents[3] / "uploads"


@router.post("/{application_id}/documents", response_model=DocumentOut)
async def upload_document(
    application_id: int,
    doc_type: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")

    if not is_supported_doc_type(doc_type):
        raise HTTPException(status_code=400, detail=f"Unsupported document type: {doc_type}")
    if not is_supported_file(file.filename):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

    app_dir = UPLOAD_ROOT / str(application_id)
    app_dir.mkdir(parents=True, exist_ok=True)
    storage_path = app_dir / f"{doc_type}_{file.filename}"
    contents = await file.read()
    storage_path.write_bytes(contents)

    document = Document(
        application_id=application_id,
        doc_type=doc_type,
        filename=file.filename,
        storage_path=str(storage_path),
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


@router.get("/{application_id}/documents", response_model=list[DocumentOut])
def list_documents(
    application_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    return db.query(Document).filter(Document.application_id == application_id).all()


@router.get("/{application_id}/documents/verify", response_model=DocumentVerificationOut)
def verify_application_documents(
    application_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")

    uploaded_doc_types = [
        d.doc_type for d in db.query(Document).filter(Document.application_id == application_id).all()
    ]
    return verify_documents(application.loan_scheme, application.external_id, uploaded_doc_types)
