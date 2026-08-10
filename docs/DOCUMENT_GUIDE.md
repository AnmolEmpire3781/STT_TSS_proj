# Phase 1 document/data guide

## Supported files

| Type | Supported | Notes |
|---|---|---|
| PDF | Yes | Text-based PDF; image-only/scanned PDF needs OCR first |
| DOCX | Yes | Paragraph text; complex tables/images are not explicitly extracted |
| TXT | Yes | Use UTF-8, especially for Hindi |
| Markdown | Yes | Use UTF-8 |
| CSV | Yes | Rows are turned into text; suitable for small reference tables |
| XLSX | No | Convert to CSV or extend the loader |
| PPTX | No | Export relevant text to a supported format or add a parser |
| Images | No | Add OCR/document vision later |
| Audio/video | No | This folder is for RAG knowledge, not STT training data |

## What to ingest

Use factual content users are allowed to ask about:

- policy/SOP documents
- product/user manuals
- FAQs
- troubleshooting guides
- support processes
- warranty/refund/cancellation terms
- small product/reference tables
- approved internal knowledge articles

English and Hindi documents can be mixed in one collection because the configured `BAAI/bge-m3` embedding model is multilingual.

## Cross-language retrieval

The intended behavior is:

- Hindi question → retrieve English document → answer in Hindi
- English question → retrieve Hindi document → answer in English
- mixed/Hinglish query → retrieve whichever document is semantically closest

The LLM is instructed to keep normal technical terms such as API, VPN, FastAPI, AWS, product names and acronyms in English when appropriate.

## Do not ingest blindly

Do not put secrets, credentials, unrestricted personal data, or documents the application user is not authorized to access into one shared collection. Production systems need document-level authorization/metadata filters.

For highly dynamic structured data (millions of DB rows, orders, balances, live inventory), use a database/tool connector rather than periodically embedding the whole database.

## Before ingestion

```bash
python backend/scripts/validate_docs.py --path backend/data/docs
```

A PDF extracting almost no characters is often scanned/image-only and should go through OCR first.
