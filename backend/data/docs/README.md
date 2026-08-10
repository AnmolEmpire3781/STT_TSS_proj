# Knowledge documents for Phase 1

Put your **English and/or Hindi** knowledge files in this folder, then run ingestion.

Supported by the current starter:

- `.pdf` — text-based PDFs only; scanned/image-only PDFs need OCR before ingestion
- `.docx` — paragraph text is extracted; complex tables/images are not extracted by the starter
- `.txt` — UTF-8 recommended
- `.md` — UTF-8 recommended
- `.csv` — UTF-8/UTF-8-BOM recommended; each row is converted to pipe-delimited text

Good content types:

- company policies
- employee/customer FAQs
- product manuals
- support/troubleshooting instructions
- warranty/refund/cancellation terms
- SOPs and process documentation
- product/catalog reference CSVs

Do not use the RAG folder as a replacement for transactional databases. Large/highly dynamic row-level datasets should later be queried through a database/tool connector.

The three `sample_*` files are only demo knowledge. Delete/replace them before using real business data.
