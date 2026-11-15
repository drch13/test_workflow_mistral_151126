import json
from datetime import timedelta
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

import mistralai.workflows as workflows
import mistralai.workflows.plugins.mistralai as mistralai_plugin
from mistralai.workflows.conversational import (
    ConfirmationInput,
    FileField,
    FileWithMetadataValue,
    FormInput,
)

OCR_MODEL = "mistral-ocr-4-1"
CLASSIFIER_MODEL = "mistral-small-2603"
EXTRACTOR_MODEL = "mistral-large-4"
CONFIDENCE_THRESHOLD = 0.7


class DocumentCategory(str, Enum):
    HOSPITAL_INVOICE = "facture_hopital"
    MEDICATION_PRESCRIPTION = "ordonnance_medicaments"
    IMAGING_PRESCRIPTION = "ordonnance_imagerie"
    IMAGING_REPORT = "compte_rendu_imagerie"
    OUT_OF_SCOPE = "hors_categorie"

    @property
    def label(self) -> str:
        return _CATEGORY_LABELS[self]


_CATEGORY_LABELS = {
    DocumentCategory.HOSPITAL_INVOICE: "Facture d'hôpital",
    DocumentCategory.MEDICATION_PRESCRIPTION: "Prescription de médicaments",
    DocumentCategory.IMAGING_PRESCRIPTION: "Prescription d'imagerie",
    DocumentCategory.IMAGING_REPORT: "Compte-rendu d'imagerie",
    DocumentCategory.OUT_OF_SCOPE: "Document hors catégories (non médical)",
}

CATEGORY_OPTIONS = [(category.value, category.label) for category in DocumentCategory]


class PipelineInput(BaseModel):
    document_url: Optional[str] = None
    document_name: Optional[str] = None


class DocumentUploadForm(FormInput):
    document: FileWithMetadataValue = FileField(
        description="Document médical à traiter (PDF)",
        include_metadata=True,
    )


class OcrResult(BaseModel):
    text: str
    page_count: int
    model: str


class ClassificationResult(BaseModel):
    category: DocumentCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class PatientInfo(BaseModel):
    nom: Optional[str] = None
    prenom: Optional[str] = None
    date_naissance: Optional[str] = None
    numero_securite_sociale: Optional[str] = None
    identifiant_dossier: Optional[str] = None


class HospitalInvoiceFields(BaseModel):
    etablissement: Optional[str] = None
    service: Optional[str] = None
    numero_facture: Optional[str] = None
    date_soins: Optional[str] = None
    actes: list[str] = Field(default_factory=list)
    montant_total: Optional[float] = None
    montant_prise_en_charge: Optional[float] = None
    reste_a_charge: Optional[float] = None


class MedicationPrescriptionFields(BaseModel):
    prescripteur: Optional[str] = None
    date_prescription: Optional[str] = None
    pharmacie: Optional[str] = None
    medicaments: list[dict[str, Any]] = Field(default_factory=list)


class ImagingPrescriptionFields(BaseModel):
    prescripteur: Optional[str] = None
    date_prescription: Optional[str] = None
    modalite: Optional[str] = None
    examens: list[str] = Field(default_factory=list)
    clinique: Optional[str] = None
    urgence: Optional[bool] = None


class ImagingReportFields(BaseModel):
    type_examen: Optional[str] = None
    date_examen: Optional[str] = None
    radiologue: Optional[str] = None
    technique: Optional[str] = None
    conclusion: Optional[str] = None
    recommandations: Optional[str] = None


class ExtractionResult(BaseModel):
    patient: PatientInfo
    facture_hopital: Optional[HospitalInvoiceFields] = None
    ordonnance_medicaments: Optional[MedicationPrescriptionFields] = None
    ordonnance_imagerie: Optional[ImagingPrescriptionFields] = None
    compte_rendu_imagerie: Optional[ImagingReportFields] = None
    resume: Optional[str] = None


CLASSIFIER_SYSTEM_PROMPT = """Tu es un classificateur de documents médicaux (documents en français).

Tu dois classer le document dans exactement UNE des 5 catégories suivantes :
- facture_hopital : facture d'hôpital, note d'honoraires, facture de clinique, feuille de soins à facturer
- ordonnance_medicaments : prescription (ordonnance) de médicaments
- ordonnance_imagerie : prescription (ordonnance) d'examen d'imagerie médicale (radio, scanner, IRM, échographie, mammographie, etc.)
- compte_rendu_imagerie : compte-rendu d'examen d'imagerie médicale (résultat d'un examen déjà réalisé)
- hors_categorie : le document n'est PAS un document médical des 4 catégories ci-dessus (par exemple un poème, un courrier personnel, un document administratif sans lien avec des soins). Utilise cette catégorie dès que le contenu ne correspond à aucune des 4 catégories médicales, même si le document évoque vaguement la santé.

Réponds UNIQUEMENT avec un objet JSON conforme au schéma fourni :
- category : la catégorie retenue (une des 5 valeurs ci-dessus)
- confidence : ton score de confiance, un nombre entre 0.0 et 1.0
- reasoning : une justification courte en français
"""

EXTRACTION_SYSTEM_PROMPT = """Tu es un assistant d'extraction d'informations depuis des documents médicaux en français.

À partir du texte du document et de sa catégorie, extrais :
- patient : les informations d'identité du patient (nom, prénom, date de naissance, numéro de sécurité sociale, identifiant de dossier / IPP) quand elles sont présentes ; laisse null sinon
- le bloc correspondant à la catégorie du document :
  - facture_hopital : établissement, service, numéro de facture, date des soins, actes réalisés (codes NGAP/CCAM et libellés), montant total, montant pris en charge, reste à charge
  - ordonnance_medicaments : prescripteur, date de prescription, pharmacie, liste des médicaments (DCI, posologie, durée, quantité)
  - ordonnance_imagerie : prescripteur, date de prescription, modalité (radio, scanner, IRM, échographie...), examens demandés, clinique, urgence
  - compte_rendu_imagerie : type d'examen, date de l'examen, radiologue, technique, conclusion, recommandations
- resume : un résumé en 2 ou 3 phrases en français

Ne remplis QUE le bloc correspondant à la catégorie indiquée, laisse les trois autres à null.
N'invente aucune information : si un champ est absent du document, mets null (ou une liste vide).
Réponds UNIQUEMENT avec un objet JSON conforme au schéma fourni.
"""


def _mistral_client() -> Any:
    return mistralai_plugin.get_mistral_client()


def _json_schema(name: str, model: type[BaseModel]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": model.model_json_schema()},
    }


def _completion_content(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    parts = []
    for chunk in content:
        if isinstance(chunk, dict):
            parts.append(chunk.get("text") or "")
        else:
            parts.append(getattr(chunk, "text", "") or "")
    return "".join(parts)


@workflows.activity(
    start_to_close_timeout=timedelta(minutes=10),
    retry_policy_max_attempts=3,
    retry_policy_backoff_coefficient=2.0,
)
async def extract_text_with_ocr(document_url: str) -> OcrResult:
    client = _mistral_client()
    response = await client.ocr.process_async(
        model=OCR_MODEL,
        document={"type": "document_url", "document_url": document_url},
        table_format="markdown",
        include_image_base64=False,
    )
    pages = list(getattr(response, "pages", None) or [])
    text = "\n\n".join((getattr(page, "markdown", "") or "") for page in pages).strip()
    return OcrResult(
        text=text,
        page_count=len(pages),
        model=str(getattr(response, "model", OCR_MODEL)),
    )


@workflows.activity(
    start_to_close_timeout=timedelta(minutes=5),
    retry_policy_max_attempts=3,
    retry_policy_backoff_coefficient=2.0,
)
async def classify_document(text: str) -> ClassificationResult:
    client = _mistral_client()
    response = await client.chat.complete_async(
        model=CLASSIFIER_MODEL,
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Texte du document à classer :\n\n{text}"},
        ],
        response_format=_json_schema("document_classification", ClassificationResult),
        temperature=0.0,
        top_p=1.0,
    )
    return ClassificationResult.model_validate(json.loads(_completion_content(response)))


@workflows.activity(
    start_to_close_timeout=timedelta(minutes=10),
    retry_policy_max_attempts=3,
    retry_policy_backoff_coefficient=2.0,
)
async def extract_document_fields(text: str, category: str) -> ExtractionResult:
    client = _mistral_client()
    response = await client.chat.complete_async(
        model=EXTRACTOR_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Catégorie du document : {category}\n\nTexte du document :\n\n{text}",
            },
        ],
        response_format=_json_schema("document_extraction", ExtractionResult),
        temperature=0.0,
        top_p=1.0,
    )
    return ExtractionResult.model_validate(json.loads(_completion_content(response)))


def _render_summary(
    document_name: str,
    ocr: OcrResult,
    classification: ClassificationResult,
    category: DocumentCategory,
    reviewed: bool,
    extraction: ExtractionResult,
) -> str:
    patient = extraction.patient
    patient_lines = [
        f"- Nom : {patient.nom or '—'}",
        f"- Prénom : {patient.prenom or '—'}",
        f"- Date de naissance : {patient.date_naissance or '—'}",
        f"- N° de sécurité sociale : {patient.numero_securite_sociale or '—'}",
        f"- Identifiant de dossier : {patient.identifiant_dossier or '—'}",
    ]
    validation = (
        f"Catégorie confirmée/corrigée manuellement (confiance initiale : {classification.confidence:.0%} < {CONFIDENCE_THRESHOLD:.0%})"
        if reviewed
        else f"Classification automatique (confiance : {classification.confidence:.0%} ≥ {CONFIDENCE_THRESHOLD:.0%})"
    )
    return (
        f"# Résultat du traitement de « {document_name} »\n\n"
        f"- Modèle OCR : {ocr.model} ({ocr.page_count} page(s))\n"
        f"- Catégorie : **{category.label}** (`{category.value}`)\n"
        f"- Validation : {validation}\n"
        f"- Justification du classificateur : {classification.reasoning or '—'}\n\n"
        f"## Patient\n" + "\n".join(patient_lines) + "\n\n"
        f"## Champs extraits\n```json\n{json.dumps(extraction.model_dump(exclude={'patient'}, exclude_none=True), ensure_ascii=False, indent=2)}\n```\n\n"
        f"## Résumé\n{extraction.resume or '—'}\n"
    )


def _render_rejection(
    document_name: str,
    ocr: OcrResult,
    classification: ClassificationResult,
    reviewed: bool,
) -> str:
    if reviewed:
        validation = (
            f"Catégorie confirmée/corrigée manuellement (confiance initiale : {classification.confidence:.0%} < {CONFIDENCE_THRESHOLD:.0%})"
        )
    else:
        validation = (
            f"Rejet automatique (confiance : {classification.confidence:.0%} ≥ {CONFIDENCE_THRESHOLD:.0%})"
        )
    return (
        f"# ⛔ Document rejeté : « {document_name} »\n\n"
        f"- Modèle OCR : {ocr.model} ({ocr.page_count} page(s))\n"
        f"- Catégorie : **{DocumentCategory.OUT_OF_SCOPE.label}** (`{DocumentCategory.OUT_OF_SCOPE.value}`)\n"
        f"- Validation : {validation}\n"
        f"- Justification du classificateur : {classification.reasoning or '—'}\n\n"
        f"Ce document ne correspond à aucune des 4 catégories médicales prises en charge "
        f"(facture d'hôpital, ordonnance de médicaments, ordonnance d'imagerie, compte-rendu d'imagerie). "
        f"Le pipeline s'est arrêté avant l'étape d'extraction : aucune donnée patient n'a été extraite.\n"
    )


@workflows.workflow.define(
    name="medical-document-pipeline",
    workflow_display_name="Pipeline de documents médicaux",
    workflow_description=(
        "Traite un document médical PDF : extraction du texte par Mistral OCR 4.1, "
        "classification du type de document par Mistral Small 4 parmi 4 catégories médicales "
        "et une catégorie « hors_categorie » pour les documents non médicaux (rejet automatique "
        "sans extraction), avec pause et validation humaine si le score de confiance est < 0.7, "
        "puis extraction des informations patient et des champs spécifiques par Mistral Large 4."
    ),
    execution_timeout=timedelta(hours=24),
)
class MedicalDocumentPipeline(workflows.InteractiveWorkflow):
    def __init__(self) -> None:
        super().__init__()
        self.status = "demarre"
        self.document_name = ""
        self.category: Optional[str] = None
        self.confidence: Optional[float] = None
        self.awaiting_review = False

    @workflows.workflow.query(name="get_pipeline_status")
    def get_pipeline_status(self) -> dict:
        return {
            "status": self.status,
            "document_name": self.document_name,
            "category": self.category,
            "confidence": self.confidence,
            "awaiting_review": self.awaiting_review,
        }

    @workflows.workflow.entrypoint
    async def run(self, params: PipelineInput | None = None) -> mistralai_plugin.ChatAssistantWorkflowOutput:
        self.status = "recuperation_document"
        if params is not None and params.document_url:
            document_url = params.document_url
            self.document_name = params.document_name or "document.pdf"
        else:
            upload = await self.wait_for_input(
                DocumentUploadForm,
                label="Téléversement du document médical",
            )
            document_url = upload.document.url
            self.document_name = upload.document.filename

        self.status = "ocr"
        ocr = await extract_text_with_ocr(document_url)
        if not ocr.text:
            raise ValueError(f"Aucun texte extrait du document {self.document_name!r} par OCR")

        self.status = "classification"
        classification = await classify_document(ocr.text)
        self.confidence = classification.confidence

        category = classification.category
        reviewed = False

        # Rejet automatique : document hors catégories identifié avec confiance suffisante.
        if (
            category is DocumentCategory.OUT_OF_SCOPE
            and classification.confidence >= CONFIDENCE_THRESHOLD
        ):
            self.category = category.value
            self.status = "rejete"
            summary = _render_rejection(self.document_name, ocr, classification, reviewed=False)
            return mistralai_plugin.ChatAssistantWorkflowOutput(
                content=[mistralai_plugin.TextOutput(text=summary)]
            )

        if classification.confidence < CONFIDENCE_THRESHOLD:
            self.status = "validation_categorie"
            self.awaiting_review = True
            await mistralai_plugin.send_assistant_message(
                f"Score de confiance insuffisant ({classification.confidence:.0%} < {CONFIDENCE_THRESHOLD:.0%}). "
                f"Catégorie prédite : « {category.label} ». "
                "Veuillez vérifier le document et confirmer ou corriger la catégorie pour continuer le pipeline, "
                "ou choisir « hors_categorie » pour rejeter un document non médical."
            )
            review = await self.wait_for_input(
                ConfirmationInput(
                    options=CATEGORY_OPTIONS,
                    description=(
                        f"Catégorie prédite par le classificateur : « {category.label} » "
                        f"(confiance : {classification.confidence:.0%}). "
                        "Confirmez cette catégorie, sélectionnez la bonne, ou choisissez "
                        "« hors_categorie » si le document n'est pas un document médical."
                    ),
                ),
                label="Validation de la catégorie du document",
            )
            category = DocumentCategory(review.choice)
            self.awaiting_review = False
            reviewed = True

        # Rejet après validation humaine : l'utilisateur a confirmé que le document est hors catégories.
        if category is DocumentCategory.OUT_OF_SCOPE:
            self.category = category.value
            self.status = "rejete"
            summary = _render_rejection(self.document_name, ocr, classification, reviewed=reviewed)
            return mistralai_plugin.ChatAssistantWorkflowOutput(
                content=[mistralai_plugin.TextOutput(text=summary)]
            )

        self.category = category.value
        self.status = "extraction"
        extraction = await extract_document_fields(ocr.text, category.value)

        self.status = "termine"
        summary = _render_summary(self.document_name, ocr, classification, category, reviewed, extraction)
        return mistralai_plugin.ChatAssistantWorkflowOutput(
            content=[mistralai_plugin.TextOutput(text=summary)]
        )
