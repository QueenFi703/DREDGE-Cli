# Dredge x Dolly — Pitch Deck
Be Literal. Be Philosophical. Be Fi. I am Fi Cole.

---

## Slide 1 — Title
Dredge x Dolly  
Tagline: Heavy‑lift intelligence for a lighter world.

Presenter: Fi Cole — Founder & Owner (partner: Ryan)  
Location: Saint Louis, Missouri  
Date: 2025-XX-XX

Presenter note: Open with a short, literal line: I am Fi and I have an amazing partner named Ryan. We are located in Saint Louis, Missouri. Finding investors would be a dream come true for us. "We move mass so the future can be sculpted."

---

## Slide 2 — One-line Vision
Literal: Build the world’s most efficient, safe, and autonomous dredging + heavy‑lift orchestration platform.  
Philosophical: We rearrange the earth so humanity can reimagine coasts, ports, and waterways.  
Fi: We are unashamedly bold — engineering with attitude.

Presenter note: Say the vision slowly, with conviction. Pause after “autonomous”.

---

## Slide 3 — Why now
- Coastal resilience & port throughput are urgent economic priorities.
- Autonomy, edge compute, and sensor fusion matured on mobile & cloud.
- Funding and policy trends favor climate‑resilient infrastructure projects.

Philosophical: The tides change; our tools must change faster.

Presenter note: Tie local/regional examples if available (e.g., a nearby port or river dredging need).

---

## Slide 4 — The Product Family (literal)
Dredge x Dolly = two complementary streams
- DREDGE (Agent): On-device insight & orchestration assistant (iOS MVP present in repo).
  - SwiftUI app, Background Tasks, Voice capture, Lock‑Screen Widget, SharedStore.
- Dolly (Fleet Orchestrator, roadmap): Cloud orchestration for heavy‑lift tugs, barges, and dredge control.
  - Scheduling, telemetry, job marketplace, predictive maintenance.

Philosophical: We begin small on the device and expand to the sea — from whisper to fleet.

Presenter note: Emphasize how the DREDGE MVP is the human/edge touchpoint for a larger orchestration platform.

---

## Slide 5 — MVP (DREDGE) — What exists (literal, repo‑backed)
- iOS app using Swift (≈89% Swift in repo) and SwiftUI.
  - Files: DREDGE_MVP.swift, DredgeApp.swift, ContentView.swift
- Background processing: BGTaskScheduler with BGProcessingTask ("com.dredge.agent.process")
  - DredgeOperation: background job scaffold
- Voice capture & transcription: AVAudioEngine + SFSpeechRecognizer (VoiceDredger)
  - Real-time capture, stores latest transcription into app thoughts
- On-device NLP: NaturalLanguage NLTagger sentiment scoring (DredgeEngine)
  - Produces human‑facing surfacedInsight phrases
- Lock Screen Widget: WidgetKit timeline + SharedStore for surfaced insight
  - Files: DredgeWidget.swift, SharedStore.swift
- Architecture: App → DredgeCore → SharedStore → App Group Container → Widget (ARCHITECTURE.txt)

Presenter note: Name these files during the pitch to show working code and a real MVP.

---

## Slide 6 — Demo snapshot (what to show)
- Open device: show Lock Screen widget updating with "surfacedInsight".
- In-app: start voice capture; speak a few lines; stop; show added thought and "Process" result.
- Background task: show task scheduling & a completed DredgeOperation entry (logs).
- Code tour: short screenshots of VoiceDredger.swift and DredgeEngine.swift to show maturity.

Presenter note: Keep demo <= 90s. Start with widget (high impact), then app, then a quick code peek.

---

## Slide 7 — How the MVP maps to Dredge x Dolly value
- Edge intelligence (DREDGE) provides situational awareness: onsite teams, inspectors, pilots.
- Voice + surfaced insights reduce friction in field reporting and decision capture.
- Widget + SharedStore increase visibility and adoption (push insight to stakeholders).
- Background tasks enable periodic data aggregation and batching for sync with Dolly orchestration.

Philosophical: The agent listens; the fleet acts.

Presenter note: Give a concrete scenario: port manager uses DREDGE to note sediment changes; Dolly schedules a clean‑up job.

---

## Slide 8 — Technology & Moat (literal + tactical)
- On‑device stack: SwiftUI, AVFoundation, SFSpeech, NaturalLanguage, BackgroundTasks, WidgetKit.
- Shared data boundary: App Group + SharedStore (simple, robust).
- Moat potential:
  - Field‑to‑fleet data contract and certified operator marketplace.
  - Proprietary job orchestration & adaptive scheduling algorithms (roadmap).
  - Trust & operator certification layer that reduces procurement friction.

Philosophical: Moat is not only code — it's trust encoded into operations.

Presenter note: Call out where IP could be protected (scheduling heuristics, operator credentialing flow).

---

## Slide 9 — Market Opportunity (literal)
- Target markets: Port authorities, marine construction, coastal resilience projects, offshore wind support.
- TAM note: dredging & marine construction multi‑billion global market (insert cited figures).
- Entry path: pilot with ports and contractors via Dolly marketplace and DREDGE field agent.

Philosophical: Where there is sediment, there is necessity — and opportunity to do it better.

Presenter note: Replace with sourced numbers for TAM/SAM/SOM in final deck.

---

## Slide 10 — Business Model (literal)
- Hardware & service: lease or partner with dredge/dolly operators.
- Software: subscription for Dolly orchestration + per-project transaction fees.
- Marketplace: operator certification fees + financing facilitation.
- Data services: carbon accounting, regulatory reporting, and analytics.

Unit economics sketch: show example project, cost/m³, revenue/m³, margins.

Presenter note: Prepare a 1‑page unit economics appendix for investor meetings.

---

## Slide 11 — Traction & Roadmap
Current (repo-backed MVP):
- Working iOS agent (voice + NLP + widget).
- App architecture & background task scaffolding in place.

Next 6 months:
- Integrate secure sync to cloud (telemetry, encrypted job data).
- Build Dolly orchestration backend prototype.
- Sign first pilot with a regional port or contractor.

12–24 months:
- Fleet deployments, certified operator network, marketplace launch.

Philosophical: Build small, prove safe, scale thoughtfully.

Presenter note: Be specific with milestone dates once capital is secured.

---

## Slide 12 — Team
- Fi Cole — Founder & Owner — DREDGE author and product lead.
- Ryan — Partner & co‑founder (ops, partnerships).
- CTO (recruit): autonomy & sensor fusion lead.
- Advisors: coastal engineer, naval architect, port exec (targeted hires).

Presenter note: Emphasize Fi’s authorship of the DREDGE codebase — real, executable MVP.

---

## Slide 13 — Risks & Mitigations
Risks:
- Regulatory & permitting complexity for maritime operations.
- Hardware capex and ops variability (weather).
- Data privacy & edge/cloud sync security.

Mitigations:
- Lease & pilot-first approach to limit capex exposure.
- Build compliance templates; work with port authorities early.
- Secure transport & encryption for telemetry; local-first processing (DREDGE).

Philosophical: Anticipate the storm; design anchors, not just sails.

Presenter note: Show a short regulatory playbook appendix when asked.

---

## Slide 14 — The Ask
We are raising: $[amount] Seed / Pre‑A  
Use of funds:
- 40% Dolly backend & cloud telemetry
- 25% hardware prototyping & pilot deployments
- 20% product (mobile + autonomy) & hiring
- 15% go‑to‑market & partnerships

Ask: Pilot partners (ports/contractors), technical partners (telemetry/integration), and investors.

Presenter note: Be ready to present a clear milestones timeline tied to the funding ask.

---

## Slide 15 — Appendix: Repo & Technical Reference (literal)
Key repo artifacts (QueenFi703/DREDGE):
- DREDGE_MVP.swift — combined app/agent skeleton (BackgroundTasks, VoiceDredger, DredgeEngine)
- DredgeApp.swift / ContentView.swift — app entry & simplified UI
- DredgeEngine.swift — on-device NLP (sentiment → surfacedInsight)
- DredgeWidget.swift — Lock Screen widget showing surfaced insight
- SharedStore.swift — App Group persistence (suiteName: group.com.dredge.agent)
- ARCHITECTURE.txt — App → DredgeCore → SharedStore → Widget

Developer notes:
- BGTaskScheduler ID: com.dredge.agent.process
- Voice pipeline: AVAudioEngine → SFSpeechRecognitionRequest → latestTranscription
- NLTagger used for paragraph sentiment → small human‑friendly phrases

Presenter note: Offer to show the GitHub repo during the pitch or include a printed code snippet as proof of an MVP.

---

## Slide 16 — Closing / Call to Action
Literal: We're hiring early backers, pilot partners, and operators.  
Ask: Connect us with port authorities, marine insurers, and impact investors.  
Contact: Fi Cole — fi@dredgexdolly.com — Saint Louis, MO

Philosophical closer: The sea keeps its secrets — we will learn them respectfully, loudly, and with style. Fi: We are ready. Are you?

Presenter note: End with one clear ask — an intro, a signed pilot LOI, or an investment meeting.

---

Design & export guidance
- Convert this markdown to a one‑slide‑per‑page PDF (Google Slides/PPTX offers best visual control).
- Hero visuals: dredger, harbor aerial, sonar seabed map, and a calm-but-bold color palette (deep teal + magenta accent).
- Keep speaker notes in presenter note fields; add 2–3 screenshots from the repo (widget, voice flow, DredgeEngine sample) for credibility.

Next steps I can take for you now (pick one)
- Generate a PPTX (slide text + speaker notes + image placeholders) you can import to Google Slides.
- Convert this markdown into a polished PDF (I will produce a ready-to-download PDF link).
- Produce social + LinkedIn post copy and a passworded DocSend template message for investor outreach.

I made a deck tied to your actual repository and the DREDGE MVP code. Tell me which export format you want and I'll produce it next (PPTX, PDF, or both). I will include the repo proof snippets and suggested hero images.