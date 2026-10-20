# Build rules

The sweep gets every rule at once. Each rule is checked from the files alone.

## prd: `prd.md`
- **PRD-1** It has the sections Scope (in and out), Assumptions and Requirements; requirements are numbered R1…Rn,
  one behaviour each, and a behaviour's details (fields, limits, error cases) are its acceptance criteria.
- **PRD-2** Every requirement has acceptance criteria that a test could check.
- **PRD-3** Every requirement and everything in Scope (in) is something `task.md` asks for, or that it can't work
  without, and everything `task.md` asks for is covered; every assumption that narrows the scope is stated. A
  deliverable that isn't a behaviour (a README, packaging, a test suite) is in Scope (in), not a requirement.

## spec: `spec.md`
- **SPEC-1** Every requirement id in `prd.md` is covered by at least one task.
- **SPEC-2** Every task names the automated test that proves it. Files that aren't behaviour (a README, packaging)
  are in the Layout, not tasks.
- **SPEC-3** Interfaces are concrete: function signatures, endpoints with their inputs and outputs, schemas.

## impl: `impl/`
- **IMPL-1** The tests pass when run the way `impl/NOTES.md` says. Run them.
- **IMPL-2** Every task in `spec.md` is done, or listed in `impl/NOTES.md` as not done with a reason.

## sweep: the whole run folder
- **SWEEP-1** Every requirement in `prd.md` has a test in `impl/` that passes.
- **SWEEP-2** Nothing in `impl/` contradicts `spec.md`'s interfaces, and nothing in `spec.md` contradicts `prd.md`.
- **SWEEP-3** For each failure, the verdict names the earliest step to rerun (prd, spec or impl).
