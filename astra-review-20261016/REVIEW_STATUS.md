# Astra review material — October 16, 2026

Existing destination: ColeWorld inc OPN Demo Org / Default project, DREDGE Astra draft. Package 0.2.2 and website build v45 are different artifacts.

Support, privacy and terms pages were deployed successfully in v45. The portal imported all four listing URLs and planned five positive / three negative cases. It subsequently flagged a missing app icon. The corrected package in this branch includes the icon, but that correction is not yet uploaded because protected browser credential state became unavailable.

## Reviewer access preparation

Use the reviewer's own supported ChatGPT login. A signed-in non-owner receives one account-isolated prepared fictional investigation through astra_list_remembered and astra_read_remembered. The reviewer must not confirm or link the owner's Siri account. No owner credentials, device tokens, API keys, private Siri history, or production records are supplied. Samples are generated deterministically for the account, not live research and not durable newly saved investigations.

The source passed 233 offline tests for the reviewer change, including owner consent/revocation and Siri regression checks. A live non-owner reviewer login has not passed. Current official review guidance asks for a dedicated test account with immediate access; support must confirm whether the platform-managed own-login reviewer path satisfies that requirement. Do not claim an exception is approved.

## Connection evidence

Sites returned the authoritative endpoint and OAuth resource https://dredge-echo-astra-demo.queenfi24.chatgpt.site/mcp. The selected Sites-managed plugin completed astra_list_remembered with an empty investigation list and rejected a synthetic nonexistent ID through astra_read_remembered. These results verify that selected connection, not the separate Demo Org draft. Its portal still reports authentication unavailable and no tool scan. No guessed OAuth overrides were added and no repeated reconnection was attempted.

## Remaining before submission

Upload the icon-corrected package into the same draft. Complete secure reviewer fields, record a current accessible video, execute all cases with the reviewer identity, then diagnose/complete OAuth and tool discovery in the Demo Org draft using provider-verified settings. No final review submission has occurred. Paid MCP generation remains disabled. Keep the pinned Siri owner, shortcuts and existing records intact.
