# Annual agency casework setup

Annual enterprise pricing is **quote per agency**, not a fixed Stripe price. The $19/month plan is unchanged. Requests are encrypted and recorded; this implementation does not automatically send email, create a contract, charge a customer or grant agency access. An administrator retrieves requests from the encrypted database as part of the manual contracting process.

Railway variables on DREDGE-Cli:

- `OPENAI_API_KEY`: reuse the existing Astra project API key. Enter privately in Railway; never commit it. Requests use `gpt-6-astra`, without a fallback model.
- `CASEWORK_ENCRYPTION_KEY`: a stable Fernet key generated with `python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'`. Store securely and back it up separately from the database. Replacing or losing it makes existing encrypted records unreadable; rotation needs a migration.
- `CASEWORK_MEMBERS_JSON`: trusted OAuth IDs mapped to agency and role, e.g. `{"github:174653334":{"agency":"your-agency-id","role":"supervisor"}}`. Use your actual agency ID and authorized staff. `caseworker` sees owned cases; `supervisor` sees cases within their agency. Studio operator, reviewer and admin roles grant no case access.
- `CASEWORK_REAL_DATA_ENABLED=true`: set only after the agency approves data classification, consent, retention/deletion procedures, backup security and permitted staff. Default off.
- `CASEWORK_AI_DATA_ENABLED=true`: separately approve case-text transfer to OpenAI after verifying project controls, applicable agreements and policy. Default off. A per-request confirmation is also required.

Existing persistent `STUDIO_DB_PATH` stores encrypted case titles, original documents, extracted text, quote requests and AI outputs. Audit events contain object IDs and actions only. Case files are not stored in the legacy Studio runs or public static directories. Uploads support UTF-8 TXT and unencrypted text PDFs, 5 MB, 100 pages, 100,000 characters. Scans need OCR first. Archive is retention, not erasure. Permanent deletion and cryptographic key rotation require an administrator-run procedure; do not promise them in a contract until that procedure is established.

Analysis sends at most 5 selected files / 50,000 characters to OpenAI, with no tools. Public research is a separate web-enabled request with no case binding or attachment; users must enter a generic question and confirm no client data. That confirmation is not a reliable automated PII detector. Do not enable live web research for protected client information.

Responses use `store:false`, foreground requests and no automatic retry. This does not guarantee zero retention: provider abuse monitoring and caching controls still apply. Verify the reused project's actual data controls and agreements. No HIPAA, CJIS or other compliance certification is claimed.

Each agency is limited to 20 AI attempts per rolling 24 hours, output capped at 2,400 tokens and web tool calls capped at 2. Failed calls count. Tokens are reported from the provider, but this is not an exact dollar-spend cap. Set project-level spending limits in OpenAI; reuse means both apps share the project's budget and rate limits.

AI drafts require a caseworker to validate evidence; the tool does not make benefits, eligibility or adverse-action decisions. There is no enforced independent AI draft sign-off in this release. Pilot only after agency review, restore testing, retention and incident procedures are established.
