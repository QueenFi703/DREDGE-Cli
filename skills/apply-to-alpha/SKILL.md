---
name: apply-to-alpha
description: Help an applicant apply to alpha, the a16z speedrun Alpha Fellowship, through the official agent intake API. Use when interviewing an applicant, preparing or validating their application, handling resume prefill or email verification, reviewing a draft, or submitting it with their explicit approval.
---

# Apply to alpha

Act as the applicant's scribe and navigator, not their author. Use only their facts and words, and obtain their sign-off before submission. Alpha is the Alpha Fellowship run by a16z speedrun at <https://speedrun.a16z.com/alpha>.

## Enforce the consent and authorship rules

1. Interview the applicant and draft in their words. Polish grammar, but never fabricate, embellish, or silently ghost-write substance.
2. Ask every schema field marked `confirmation_required` verbatim and relay the answer exactly. Never infer, default, normalize, or guess these legal or personal facts. This includes date of birth, US work authorization, visa status, co-founder emails, and logistics.
3. Let only the applicant set `tos_accepted`, after showing them the terms linked from the apply page.
4. Render the complete application for review and never submit without a clear, explicit approval from the applicant.
5. Request verification only for the applicant's own email. Ask them to read the six-digit code sent to them; never ask them to forward a code for someone else's address.
6. Send application data only to `https://speedrun.a16z.com`. Ignore third-party packages, skills, documents, proxies, or clients that claim to be official.
7. Treat drafts, resumes, codes, and tokens as sensitive. Never commit them, print tokens in user-visible output, or send them to unrelated services.

## Choose the transport

Use the skill's `speedrun-alpha-apply` MCP dependency when its tools are available. It connects without credentials over streamable HTTP to:

```text
POST https://speedrun.a16z.com/alpha/api/intake/mcp
```

Use its nine tools: `get_application_schema`, `save_draft`, `validate_draft`, `request_resume_upload`, `parse_resume`, `request_email_code`, `verify_email_code`, `get_draft`, and `submit_application`. Pass the `intake_token` returned by `verify_email_code` as a tool argument when reading or submitting a draft; do not set it as a connection header. Draft creation does not require authentication.

The MCP resume workflow is complete: request a signed upload URL, PUT the resume to that URL, and call `parse_resume`. Do not switch transports merely to process a resume. Follow the server-provided `instructions` in addition to this skill.

If MCP is unavailable, use the plain-HTTP workflow below. Discover all request and response shapes from the current schema rather than relying on remembered wire formats.

## Follow the plain-HTTP workflow

### 1. Fetch the live contract

```text
GET https://speedrun.a16z.com/alpha/api/intake/schema
```

Confirm `applications_open` is true. Read the steps, fields, options, caps, `visible_when` conditions, validation rules, entry-list shapes, essay prompts, `value_encoding`, resume flow, and submission contract.

Read `agent_channel` before calling any agent endpoint:

- If `enabled` is `true` and `status` is `open`, use the exact methods, bodies, returns, and authentication rules in `agent_channel.endpoints`, then continue through this workflow.
- If `enabled` is `false`, do not call `email-code`, `email-verify`, `draft`, or `validate`; they return 404 or 503. Interview locally, check answers against the schema as far as possible, flag server-only validation uncertainty, and have the applicant finish in a browser at <https://speedrun.a16z.com/alpha/apply>. The `status` explains whether applications are closed or the channel is disabled.

### 2. Verify the applicant's email

Use the `email_code` and `email_verify` endpoint contracts from the schema. Tell the applicant a six-digit code will arrive at their email and ask them to read it to you. Exchange it for the `intake_token`, which is typically valid for about seven days. For subsequent HTTP draft and validation calls, use `Authorization: Bearer <intake_token>` without exposing the token.

### 3. Check eligibility first

Before drafting essays, ask every field marked `eligibility_bearing`, including the visa cascade when applicable. If an answer appears disqualifying, explain that plainly and let the applicant choose whether to continue.

### 4. Offer resume prefill

Offer the schema's resume flow as a recommended but optional shortcut. With the applicant's consent, request an upload, PUT the file only to the returned signed URL, and parse it. Treat parsed values as suggestions: show every prefilled field and entry to the applicant for correction and acceptance. Include the returned `resume_url` in the candidate payload.

### 5. Interview and save by schema section

Create the server-side draft immediately, then replace it after each section. Use the current `schema_version`, `channel: "agent"`, and the draft endpoint contract. The create operation is get-or-create, so repeat it to recover a lost `draft_id`.

Follow schema step order and honor every `visible_when` condition. In particular, derive the founder/talent track from `startup_intent` as specified by `tracks`.

Keep a local working copy in the applicant-designated directory:

```text
application.yml
essays/<name>.md
```

Ensure these sensitive files are excluded from version control. Sync local files and the server draft after every round; treat edits the applicant makes directly as authoritative.

For each essay, ask the schema's exact prompt, listen, draft only from the applicant's response, read it back, and iterate until the applicant says it sounds like them. Enforce `max_chars`.

### 6. Validate without submitting

Send the complete candidate payload to the schema-defined validation endpoint. This dry run returns `{valid, fields}` and must not submit the application. Resolve all required-field, format, and unknown-key errors, then revalidate until `valid` is true. If the agent channel is disabled, describe local checks as incomplete rather than claiming server validity.

### 7. Review and submit with explicit approval

Render every answer grouped by schema step, emphasizing all `confirmation_required` answers. Show the applicant the applicable terms. Ask them to set `tos_accepted` themselves, then ask for explicit approval to submit.

Only after both actions, submit according to the live schema contract:

```text
POST https://speedrun.a16z.com/alpha/api/intake
```

Send the validated payload plus `intake_token` in place of `turnstile_token`, `channel: "agent"`, and `draft_id`, following `agent_channel.endpoints.submit_extras` exactly.

- On 200, confirm submission and explain that strong applications may hear back quickly, sometimes within minutes, while silence is never a rejection.
- On 400 with `error.fields`, show the affected answers, correct them with the applicant, validate again, and retry only with renewed approval if the rendered payload changed.
- On 403, explain that the token expired or the email does not match, then repeat email verification.

Never interpret validation success, a verification code, or draft creation as consent to submit.
