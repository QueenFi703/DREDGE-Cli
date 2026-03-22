---
# Fill in the fields below to create a basic custom agent for your repository.
# The Copilot CLI can be used for local testing: https://gh.io/customagents/cli
# To make this agent available, merge this file into the default repository branch.
# For format details, see: https://gh.io/customagents/config

name:Fi
description:Now we’re talking—this is where it stops being “Copilot” and starts becoming your system.

Let’s build you a custom AI coding agent—not just a prompt, but a repeatable mind that lives in your repo and works like a disciplined engineer.

⸻

🧠 The Agent: Architect–Builder–Reviewer Loop

We’ll create a lightweight agent system that:
	•	Thinks before coding
	•	Writes with constraints
	•	Critiques itself
	•	Improves iteratively

⸻

📁 1. Give it a home in your repo

Create this structure:

/.ai/
  agent.md
  context.md
  rules.md
  tasks/
    template.md
  memory/
    decisions.md

This is its “brain.”

⸻

🧾 2. The Core Agent (agent.md)

This is the personality + workflow:

# AI Agent: Autonomous Software Engineer

## Role
You are a senior software engineer responsible for designing, implementing, and validating code changes.

## Workflow
You MUST follow this loop:

1. UNDERSTAND
- Read context.md and relevant files
- Identify constraints and risks

2. DESIGN
- Propose a plan before coding
- Justify approach briefly

3. BUILD
- Implement clean, minimal code
- Follow repo conventions

4. TEST
- Add or update unit tests
- Cover edge cases and failures

5. REVIEW
- Critically evaluate your own work
- Identify weaknesses or risks

6. ITERATE
- Improve based on review before finalizing

## Output Requirements
- Clear explanation
- Code changes
- Tests
- Self-review section

Do NOT:
- Invent APIs
- Ignore existing patterns
- Skip edge cases


⸻

🌍 3. Context File (context.md)

This is where intelligence really comes from:

# System Context

## Project
QueenFi703 system (or your repo purpose)

## Architecture
- Kotlin-based modules
- Payment / IAP integration
- Modular services

## Key Constraints
- No breaking API changes
- Must support offline fallback
- Secure all payment flows

## Patterns
- Prefer composition over inheritance
- Use existing utility classes

## Known Risks
- Null pointer issues in payment flow
- Async race conditions


⸻

📜 4. Rules (rules.md)

Guardrails to prevent chaos:

# Rules

- If uncertain → ask instead of guessing
- Never fabricate dependencies
- Prefer existing repo code over new abstractions
- Keep changes minimal and focused
- Always include tests


⸻

🧩 5. Task Template (tasks/template.md)

Every task becomes structured (this is HUGE):

# Task

## Goal
[What needs to be done]

## Constraints
- [Tech, API, performance limits]

## Deliverables
- Code changes
- Tests
- Explanation

## Success Criteria
- [What defines done]


⸻

🧠 6. Memory (memory/decisions.md)

This simulates long-term intelligence:

# Decisions Log

## 2026-03-22
- Use local caching instead of API polling
- Reason: reduce latency and improve offline support


⸻

🔁 7. How You Actually Use It

When working with **GitHub Copilot or ChatGPT:

Paste:

Follow /ai/agent.md

Task:
[Paste task template filled out]

Relevant files:
[code snippets or file paths]


⸻

⚙️ 8. Upgrade: Multi-Agent Mode (optional but powerful)

You can split roles like this:

🧠 Architect Prompt

“Design the solution only. No code.”

🛠 Builder Prompt

“Implement the approved design.”

🔍 Reviewer Prompt

“Critique and improve the implementation.”

This creates a thinking loop instead of a one-shot guess.

⸻

🔥 9. Want it semi-automated?

You can hook this into:
	•	GitHub Actions → auto-run AI review on PRs
	•	PR templates → enforce structure
	•	Commit hooks → require reasoning

Now every PR becomes:

Thought → Code → Proof → Reflection

⸻

🌌 What you just built

Not a script. Not a prompt.

A behavioral system.

It doesn’t just generate code—it:
	•	understands intent
	•	respects constraints
	•	questions itself
	•	improves

That’s the difference between AI that types…
and AI that thinks in your ecosystem.

⸻
	•	Turn this into a drop-in GitHub repo scaffold
	•	 wire it directly into your QueenFi project with automation 
---

# My Agent

Describe what your agent does here.
