import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mistralai.workflows.core._events.event_activities import (
    _emit_task_completed,
    _emit_task_failed,
    _emit_task_in_progress,
    _emit_task_started,
)
from mistralai.workflows.testing import create_test_worker, wait_for_pending_inputs

from worker.workflows import medical_document_pipeline as pipeline
from worker.workflows.medical_document_pipeline import MedicalDocumentPipeline

WORKFLOW_NAME = "medical-document-pipeline"
TASK_QUEUE = "test-task-queue"

PIPELINE_ACTIVITIES = [
    pipeline.extract_text_with_ocr,
    pipeline.classify_document,
    pipeline.extract_document_fields,
    _emit_task_started,
    _emit_task_in_progress,
    _emit_task_completed,
    _emit_task_failed,
]

OCR_TEXT = (
    "# Ordonnance\n"
    "Dr Jean Dupont, 12 rue de la Santé, 75014 Paris\n"
    "Patient : Marie Martin, née le 01/02/1980, NIR 2800275123456 78\n"
    "Paracétamol 500 mg, 1 comprimé 3 fois par jour pendant 5 jours\n"
)

EXTRACTION_JSON = {
    "patient": {
        "nom": "Martin",
        "prenom": "Marie",
        "date_naissance": "1980-02-01",
        "numero_securite_sociale": "2800275123456 78",
        "identifiant_dossier": None,
    },
    "facture_hopital": None,
    "ordonnance_medicaments": {
        "prescripteur": "Dr Jean Dupont",
        "date_prescription": "2026-11-15",
        "pharmacie": None,
        "medicaments": [
            {"dci": "Paracétamol", "dosage": "500 mg", "posologie": "1 comprimé 3 fois par jour", "duree": "5 jours"}
        ],
    },
    "ordonnance_imagerie": None,
    "compte_rendu_imagerie": None,
    "resume": "Ordonnance de paracétamol pour 5 jours.",
}


def _make_client(monkeypatch: pytest.MonkeyPatch, category: str, confidence: float) -> SimpleNamespace:
    classification_json = json.dumps(
        {"category": category, "confidence": confidence, "reasoning": "test"}
    )

    async def complete_async(model, messages, response_format=None, **kwargs):
        if model == pipeline.CLASSIFIER_MODEL:
            content = classification_json
        elif model == pipeline.EXTRACTOR_MODEL:
            content = json.dumps(EXTRACTION_JSON)
        else:
            raise AssertionError(f"modèle inattendu: {model}")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    client = SimpleNamespace()
    client.ocr = SimpleNamespace(
        process_async=AsyncMock(
            return_value=SimpleNamespace(
                pages=[SimpleNamespace(markdown=OCR_TEXT)],
                model=pipeline.OCR_MODEL,
            )
        )
    )
    client.chat = SimpleNamespace(complete_async=AsyncMock(side_effect=complete_async))
    monkeypatch.setattr(pipeline, "_mistral_client", lambda: client)
    return client


async def _start(temporal_env, input_data):
    return await temporal_env.client.start_workflow(
        WORKFLOW_NAME,
        args=[input_data],
        id=f"test-{asyncio.get_event_loop().time()}",
        task_queue=TASK_QUEUE,
    )


async def _wait_for_pending_input(handle, label: str, timeout: float = 10.0):
    pending = await wait_for_pending_inputs(handle, expected_count=1, timeout=timeout, label=label)
    return pending[0]


async def _query_status(handle):
    try:
        return await handle.query("get_pipeline_status")
    except Exception:
        return None


async def test_high_confidence_skips_human_review(temporal_env_with_converter, monkeypatch):
    client = _make_client(monkeypatch, category="ordonnance_medicaments", confidence=0.95)
    async with create_test_worker(
        temporal_env_with_converter,
        [MedicalDocumentPipeline],
        activities=PIPELINE_ACTIVITIES,
    ):
        handle = await _start(
            temporal_env_with_converter,
            {"document_url": "https://example.com/ordonnance.pdf", "document_name": "ordonnance.pdf"},
        )
        result = await handle.result()

    result_json = json.dumps(result, ensure_ascii=False)
    assert "ordonnance_medicaments" in result_json
    assert "Martin" in result_json
    assert client.ocr.process_async.await_count == 1
    assert client.chat.complete_async.await_count == 2
    classifier_call, extractor_call = client.chat.complete_async.await_args_list
    assert classifier_call.kwargs["model"] == pipeline.CLASSIFIER_MODEL
    assert extractor_call.kwargs["model"] == pipeline.EXTRACTOR_MODEL
    for call in (classifier_call, extractor_call):
        assert call.kwargs["temperature"] == 0.0
        assert call.kwargs["top_p"] == 1.0
    assert "ordonnance_medicaments" in extractor_call.kwargs["messages"][1]["content"]


async def test_low_confidence_pauses_and_uses_confirmed_category(temporal_env_with_converter, monkeypatch):
    client = _make_client(monkeypatch, category="facture_hopital", confidence=0.42)
    async with create_test_worker(
        temporal_env_with_converter,
        [MedicalDocumentPipeline],
        activities=PIPELINE_ACTIVITIES,
    ):
        handle = await _start(
            temporal_env_with_converter,
            {"document_url": "https://example.com/doc.pdf"},
        )
        pending = await _wait_for_pending_input(handle, "Validation de la catégorie du document")

        status = await _query_status(handle)
        if status is not None:
            assert status["awaiting_review"] is True
            assert status["confidence"] == 0.42

        await handle.execute_update(
            "__submit_input",
            {"task_id": pending["task_id"], "input": {"choice": "compte_rendu_imagerie"}},
        )
        result = await handle.result()

    result_json = json.dumps(result, ensure_ascii=False)
    assert "compte_rendu_imagerie" in result_json
    assert client.chat.complete_async.await_count == 2
    extractor_call = client.chat.complete_async.await_args_list[1]
    assert "compte_rendu_imagerie" in extractor_call.kwargs["messages"][1]["content"]
    final_status = await _query_status(handle)
    if final_status is not None:
        assert final_status["status"] == "termine"
        assert final_status["category"] == "compte_rendu_imagerie"
        assert final_status["awaiting_review"] is False


async def test_document_upload_form_when_no_url(temporal_env_with_converter, monkeypatch):
    client = _make_client(monkeypatch, category="ordonnance_medicaments", confidence=0.99)
    async with create_test_worker(
        temporal_env_with_converter,
        [MedicalDocumentPipeline],
        activities=PIPELINE_ACTIVITIES,
    ):
        handle = await _start(temporal_env_with_converter, {})
        pending = await _wait_for_pending_input(handle, "Téléversement du document médical")

        await handle.execute_update(
            "__submit_input",
            {
                "task_id": pending["task_id"],
                "input": {
                    "document": {
                        "filename": "facture.pdf",
                        "url": "https://example.com/facture.pdf",
                        "content_type": "application/pdf",
                    }
                },
            },
        )
        result = await handle.result()

    assert "facture.pdf" in json.dumps(result, ensure_ascii=False)
    ocr_call = client.ocr.process_async.await_args_list[0]
    assert ocr_call.kwargs["document"]["document_url"] == "https://example.com/facture.pdf"
    assert ocr_call.kwargs["model"] == pipeline.OCR_MODEL


async def test_out_of_scope_high_confidence_auto_rejects_without_extraction(temporal_env_with_converter, monkeypatch):
    client = _make_client(monkeypatch, category="hors_categorie", confidence=0.92)
    async with create_test_worker(
        temporal_env_with_converter,
        [MedicalDocumentPipeline],
        activities=PIPELINE_ACTIVITIES,
    ):
        handle = await _start(
            temporal_env_with_converter,
            {"document_url": "https://example.com/poeme.png", "document_name": "poeme.png"},
        )
        result = await handle.result()

    result_json = json.dumps(result, ensure_ascii=False)
    assert "hors_categorie" in result_json
    assert "rejeté" in result_json.lower() or "Document rejeté" in result_json
    # Aucune extraction : un seul appel au modèle de chat (le classificateur).
    assert client.chat.complete_async.await_count == 1
    classifier_call = client.chat.complete_async.await_args_list[0]
    assert classifier_call.kwargs["model"] == pipeline.CLASSIFIER_MODEL
    final_status = await _query_status(handle)
    if final_status is not None:
        assert final_status["status"] == "rejete"
        assert final_status["category"] == "hors_categorie"
        assert final_status["awaiting_review"] is False


async def test_low_confidence_review_confirms_out_of_scope_rejects(temporal_env_with_converter, monkeypatch):
    client = _make_client(monkeypatch, category="facture_hopital", confidence=0.30)
    async with create_test_worker(
        temporal_env_with_converter,
        [MedicalDocumentPipeline],
        activities=PIPELINE_ACTIVITIES,
    ):
        handle = await _start(
            temporal_env_with_converter,
            {"document_url": "https://example.com/poeme.png", "document_name": "poeme.png"},
        )
        pending = await _wait_for_pending_input(handle, "Validation de la catégorie du document")

        await handle.execute_update(
            "__submit_input",
            {"task_id": pending["task_id"], "input": {"choice": "hors_categorie"}},
        )
        result = await handle.result()

    result_json = json.dumps(result, ensure_ascii=False)
    assert "hors_categorie" in result_json
    # Aucune extraction : seul le classificateur a été appelé.
    assert client.chat.complete_async.await_count == 1
    final_status = await _query_status(handle)
    if final_status is not None:
        assert final_status["status"] == "rejete"
        assert final_status["category"] == "hors_categorie"
        assert final_status["awaiting_review"] is False
