# Make It Sell Itself: DREDGE Commercialization Playbook

This playbook mirrors the commercialization guidance in the root README and is intended for both human contributors and coding agents (`@codex`, `@claude`) to keep messaging and execution synchronized.

## 1) Product Positioning

**DREDGE is the reasoning gateway between user intent and model execution.**

Use this framing consistently in docs, landing pages, demos, and API onboarding copy.

## 2) Self-Serve API GTM Loop

1. Ship one paid endpoint (`POST /invoke`) that delivers value in under 2 seconds.
2. Gate access with API keys and tier quotas (Free / Pro / Team / Enterprise).
3. Meter every request (request ID, latency, mode, customer) into Postgres.
4. Expose usage + billing in-product so users can self-upgrade.
5. Publish copy-paste SDK snippets (curl, Python, TypeScript).

## 3) Starter Packaging

- **Free**: 250 requests/month, standard mode.
- **Pro ($29/mo)**: 10,000 requests/month, deep + transform modes.
- **Team ($99/mo)**: 50,000 requests/month, priority latency + shared keys.
- **Enterprise**: custom limits, private deployment, SLA.

## 4) Activation Funnel (Week 1)

- Day 1: Launch docs page with one-click "Try now" API call.
- Day 2: Publish 3 use-cases (contract risk, support triage, workflow routing).
- Day 3: Add Stripe checkout + instant key provisioning.
- Day 4+: Send usage milestone emails (saved time, quota utilization).

## 5) North-Star KPI

Track **TTFV (Time-to-First-Value)** from signup to first successful `/invoke` response.

Target: **< 5 minutes**.
