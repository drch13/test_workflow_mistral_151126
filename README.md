# medical-doc-pipeline

Pipeline [Mistral Workflows](https://docs.mistral.ai/studio/workflows/overview) de traitement de documents médicaux PDF.

## Étapes du pipeline

1. **Réception du document** : URL du PDF fournie en input d'exécution (`{"document_url": "..."}`), ou téléversement via un formulaire (`FileField`) au démarrage du workflow.
2. **OCR** (`extract_text_with_ocr`) : extraction du texte brut par **Mistral OCR 4.1** (`mistral-ocr-4-1`), page par page, en Markdown.
3. **Classification** (`classify_document`) : un appel à **Mistral Small 4** (`mistral-small-2603`) en structured output classe le document parmi :
   - `facture_hopital` — facture d'hôpital
   - `ordonnance_medicaments` — prescription de médicaments
   - `ordonnance_imagerie` — prescription d'imagerie
   - `compte_rendu_imagerie` — compte-rendu d'imagerie
   avec un `confidence` entre 0 et 1.
4. **Validation humaine (human-in-the-loop)** : si `confidence < 0.7`, le pipeline se met en pause (`wait_for_input` avec une `ConfirmationInput` pré-remplie en description) et attend que l'utilisateur confirme ou corrige la catégorie avant de continuer.
5. **Extraction** (`extract_document_fields`) : un appel à **Mistral Large 4** (`mistral-large-4`) en structured output extrait les informations du patient (nom, prénom, date de naissance, NIR, identifiant de dossier) et les champs spécifiques à la catégorie du document.
6. Le workflow retourne un résumé Markdown (`ChatAssistantWorkflowOutput`).

## Développement local

```bash
uv sync
cp .env.example .env  # renseigner MISTRAL_API_KEY
uv run python -m worker.entrypoints.worker   # ou: mistral dev
```

Lancer une exécution :

```python
from mistralai.client import Mistral

client = Mistral()
execution = client.workflows.execute_workflow(
    workflow_identifier="medical-document-pipeline",
    input={"document_url": "https://.../document.pdf"},
    wait_for_result=False,
)
```

Quand le score de confiance est < 0.7, l'exécution attend une entrée utilisateur : dans Vibe Work via le bouton de confirmation, ou par API :

```python
pending = client.workflows.executions.query_workflow_execution(
    execution_id=execution.execution_id, name="__get_pending_inputs"
)
client.workflows.executions.update_workflow_execution(
    execution_id=execution.execution_id,
    name="__submit_input",
    input={"task_id": pending["pending_inputs"][0]["task_id"], "input": {"choice": "compte_rendu_imagerie"}},
)
```

État d'avancement : query `get_pipeline_status` sur l'exécution.

## Tests

```bash
uv run pytest
```

Les tests utilisent `mistralai.workflows.testing.create_test_worker` avec un environnement Temporal en mémoire (`temporal_env_with_converter`) et un client Mistral mocké : scénarios confiance ≥ 0.7 (pas de pause), confiance < 0.7 (pause + catégorie confirmée/corrigée), et téléversement du PDF via formulaire.
