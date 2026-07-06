# DREDGE Orchestration Contract

`dredge.manifest.yaml` is the canonical source of runtime behavior in DREDGE.

## Rules

- Runtime behavior must be declared in the orchestration contract.
- Execution is constrained to explicit graph transitions.
- All state transitions must originate from append-only events.
- Infrastructure and documentation are derived projections and cannot define behavior.

## 5:24 Projection

The 5:24 model is a read-only onboarding projection of the event stream with a hard time bound of `t <= 324` seconds.

## Contract Location

- `dredge.manifest.yaml`
