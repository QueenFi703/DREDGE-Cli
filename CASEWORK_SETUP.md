# Annual agency casework setup

Annual enterprise pricing is **quote per agency**, not a fixed Stripe price. The $19/month plan is unchanged. Requests are encrypted and recorded; this implementation does not automatically send email, create a contract, charge a customer or grant agency access. An administrator retrieves requests from the encrypted database as part of the manual contracting process.

Railway variables on DREDGE-Cli:

- `OPENAI_API_KEY`: reuse the existing Astra project API key. Enter privately in Railway; never commit it. Requests use `gpt-6-astra`, without a fallback model.
- `CASEWORK_ENCRYPTION_KEY`: a stable Fernet key generated with `python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'`. Store securely and back it up separately from the database. Replacing or losing it makes existing encrypted records unreadable; rotation needs a migration.
- `CASEWORK_MEMBERS_JSON`: trusted OAuth IDs mapped to agency and role, e.g. `{"github:174653334":{"agency":"your-agency-id","role":"supervisor"}}`. Use your actual agency ID and authorized staff. `caseworker` sees owned cases; `supervisor` sees cases within their agency. Studio operator, reviewer and admin roles grant no case access.
- `CASEWORK_REAL_DATA_ENABLED=true`: set only after the agency approves data classification, consent, retention/deletion procedures, backup security and permitted staff. Default off.
- `CASEWORK_AI_DATA_ENABLED=true`: separately approve case-text transfer to OpenAI after verifying project controls, applicable agreements and policy. Default off. A per-request confirmation is also required.

Existing persistent `STUDIO_DB_PATH` stores encrypted case titles, original documents, extracted text, quote requests and AI outputs. Audit events contain object IDs and actions only. Case files are not stored in the legacy Studio runs or public static directories. Uploads support UTF-8 TXT, unencrypted PDFs, JPEG and PNG, 5 MB, 100 PDF pages, 100,000 characters and 20 megapixels. Local OCR is attempted for photos and text-poor PDF pages. Archive is retention, not erasure. Permanent deletion and cryptographic key rotation require an administrator-run procedure; do not promise them in a contract until that procedure is established.

Private AI drafts send at most 5 selected files and the question to OpenAI, with no tools. Client explanations are excluded by default. A separate per-request opt-in includes the latest five explanations for that case, including explanations from any assigned client; selected evidence plus explanations must fit 50,000 characters. Public research is a separate web-enabled request with no case binding or attachment; users must enter a generic question and confirm no client data. That confirmation is not a reliable automated PII detector. Do not enable live web research for protected client information.

Responses use `store:false`, foreground requests and no automatic retry. This does not guarantee zero retention: provider abuse monitoring and caching controls still apply. Verify the reused project's actual data controls and agreements. No HIPAA, CJIS or other compliance certification is claimed.

Each agency is limited to 20 AI attempts per rolling 24 hours, output capped at 2,400 tokens and web tool calls capped at 2. Failed calls count. Tokens are reported from the provider, but this is not an exact dollar-spend cap. Set project-level spending limits in OpenAI; reuse means both apps share the project's budget and rate limits.

AI drafts require a caseworker to validate evidence; the tool does not make benefits, eligibility or adverse-action decisions. Client-visible drafts require a different reviewer from the author; legal drafts additionally require qualified-legal-review attestation. This attestation does not independently verify a professional license. Pilot only after agency review, restore testing, retention and incident procedures are established.

## Client portal and DREDGE Case Law

`/client` uses authenticated, explicitly assigned case access rather than staff roles. Staff must verify identity before assigning the exact OAuth account ID. Clients see their own submissions and explanations; released drafts are shared with assigned clients of that case. Submission receipts include SHA-256 hashes. Original TXT, PDF, JPEG and PNG files are encrypted in the persistent database. Photos and scanned PDFs receive local OCR; extracted text must be checked and confirmed by authorized staff before inference. Redaction creates a separate exact-match text copy, not a sanitized original PDF/image. Review every copy before disclosure. Activity shows recorded document access and AI transfer events, not a complete external disclosure ledger.

Discernment compares evidence with client explanations and requests neutral clarification; it must not score credibility, fraud or eligibility. Including the latest five client explanations requires the separate opt-in for each request. Evidence plus opted-in explanations is limited to 50,000 characters. Public case-law research takes a generic question and jurisdiction with no case attachments. Citations and subsequent treatment need qualified verification; there is no integrated citator.

DREDGE Case Law offers per-case quote requests to ColeWorld inc. / Cultivating Faith. Only configured provider supervisor IDs can prepare estimates; `LEGAL_PREPARATION_PROVIDER_IDS` is an application config set, defaulting to the existing ColeWorld owner. Estimates do not collect payment, include attorney services or establish representation. Per-case checkout, professional-license verification, image redaction and retention/deletion interfaces remain separate implementation work. Existing monthly Stripe billing is unchanged.

## Local OCR

Railway runtime installs Tesseract, English language data and Poppler via Dockerfile. No external OCR service or AI provider receives files during extraction. Private Linux memory-backed temporary files are removed after processing. One OCR operation per server process is permitted; concurrent submissions retain originals with a `busy` status and can retry. Each operation has a 30-second subprocess budget, at most 10 text-poor PDF pages, and PDF rasters limited to 3,000 pixels on their longest edge. Blank pages, handwriting, layout and language differences may require staff correction. This release targets printed English.

Uploads preserve exact original bytes and receipt hashes; encrypted records include draft extraction text and its status. The staff interface offers original download, text correction, verification and retry. AI requests reject OCR-dependent evidence until staff confirms checked text. Verification records the staff ID and timestamp; a text revision guard rejects stale edits. Verified text may contain up to 100,000 characters; only this endpoint receives a bounded larger JSON request allowance, including escaped Unicode. Client-originated originals share verified text metadata with staff while preserving original download bytes. Existing stored images and empty scanned PDFs can be retried from the staff interface. Extraction failure never implies an empty valid case record.

### Provider execution receipts and a bounded connectivity check

Casework retains actual provider response/request IDs, returned model, status and
reported usage. Sanitized web-search call IDs, states, action types and validated
public HTTPS source URLs are recorded separately from answer citations. It never
stores raw reasoning or provider error bodies. Incomplete and failed requests have
receipts but no accepted draft; missing usage or billing remains unknown. Models &
tools only reports recorded web success after a completed response with an actual
completed search-call receipt. A receipt does not validate legal authority or facts.

In public research, “Load my research execution history” reloads the signed-in
member's latest 20 public attempts (including pending/failed); other users cannot
read those receipts via that route. Existing agency permissions remain required.

The optional one-shot public-web test uses a fixed generic Missouri payment-history
question, standard (`default`) service tier, low reasoning/context, 600 output
tokens and at most one web tool call. No case fields/history are sent. Its $3.50
reservation is a planning allowance, **not a provider-enforced spending cap**.
A transaction and unique agency/profile index prevent another attempt, including
when the first call times out or the process restarts. There are no retries. Obtain
appropriate spend approval before clicking it. Ordinary research remains separately
consented and is not covered by this one-shot spending reservation. No paid test is
performed by the automated test suite; mocked-provider tests are not live evidence.
