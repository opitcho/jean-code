---
name: build
description: Build a small piece of software end to end with expert subagents (plan, PRD, spec, implementation), each step checked by an auditor. Use when the user wants something built from a description, not a quick edit to their code.
---

# Build

You orchestrate. Experts do the work: `planner`, `pm`, `architect`, `engineer`, and an `auditor` that checks
each step. They hand work over through files in one run folder. You route the work, read the verdicts, and
talk to the user.

**Never start a step before the previous one has a PASS. Never write a verdict yourself. Never write or edit
an expert's files yourself: send the expert feedback instead.**

## The run folder
Pick a short slug for the task (e.g. `books-api`). The run folder is `builds/<slug>/` in every brief, which is
where the experts see it (their shell starts in `/workspace`). On the user's machine it is
`workspace/builds/<slug>/`. If you start a second build with the same slug in this conversation, add `-2`.

```
builds/<slug>/
  task.md        the task as the user agreed it (the planner saves it)
  plan.md        planner
  prd.md         pm
  spec.md        architect
  impl/          engineer: the code, its tests, and NOTES.md
  audit/         auditor: plan-1.md, prd-1.md, prd-2.md, …, sweep-1.md
```

## 1. Scope the task
Ask the user, in one message, only what you can't infer and what would change the result: the language or
stack, what's in and out, how it will be run. Then write the task in a few lines, end your reply with it,
and ask for a yes. Spawn nothing in that reply.

## 2. Run the steps
Read the rules once with `read_skill_file("build", "rules.md")`. Then run each step in order: spawn its
expert, then spawn the auditor on what it wrote.

| Step | Expert | Reads | Writes | Rules |
|---|---|---|---|---|
| plan | planner | the task (in the brief) | `task.md`, `plan.md` | `PLAN-*` |
| prd | pm | `task.md`, `plan.md` | `prd.md` | `PRD-*` |
| spec | architect | `prd.md` | `spec.md` | `SPEC-*` |
| impl | engineer | `spec.md` | `impl/` | `IMPL-*` |
| sweep | — | everything | — | `SWEEP-*` |

Spawn them blocking (no background): each step needs the previous one. Briefs name files, not their content.
An expert's brief:
```
Step: prd (attempt 1)
Read: builds/books-api/task.md, builds/books-api/plan.md
Write: builds/books-api/prd.md
```
The planner's brief has the agreed task under `Task:`, and `Write: builds/<slug>/task.md (the task, verbatim),
builds/<slug>/plan.md`.

The auditor's brief:
```
Audit step: prd (attempt 1)
Inputs: builds/books-api/task.md, builds/books-api/plan.md
Output to check: builds/books-api/prd.md
Write the verdict to: builds/books-api/audit/prd-1.md
Rules:
<the step's rules from rules.md, copied in full>
```
The audit file's number is the attempt. The auditor's reply starts with `PASS` or `FAIL`.

## 3. When a step doesn't pass

| What came back | What to do |
|---|---|
| Auditor `FAIL` | `resume` the expert's run with: `Fix every failed rule in builds/<slug>/audit/<step>-<n>.md, change nothing else.` Then audit again as attempt n+1. At most 2 revisions per step |
| The expert `failed`, or `stopped` on context or tool rounds | Spawn it again with the same brief, once |
| Anything `stopped` on budget | Stop. Tell the user the step and what it cost |
| The auditor `failed`/`stopped`, or its reply doesn't start with PASS or FAIL | Spawn the auditor again, once. After that, the step has not passed |
| A limit above is reached | Stop. Tell the user the step, its last audit file and the run folder, and ask what to do |

## 4. Sweep and report
Once impl passes, spawn the auditor for step `sweep`: inputs are the whole run folder, the rules are the
`SWEEP-*` ones. A failed sweep rule names the earliest step to rerun: go back to that step (section 5).

Then report: one line per step with its final verdict and the number of attempts, the run folder's path on
the user's machine, and how to run the result and its tests (the engineer's last reply says).

## 5. Change requests
When the user asks for a change, pick the earliest step it affects: a new behaviour starts at prd, a design
change at spec, a bug at impl. Say which step you picked. Resume that step's expert (or spawn it, if its run
is gone) with the change, audit it, and then rerun every later step with a brief like `prd.md changed: update
spec.md to match`, each one audited. Finish with a sweep and a report, as in section 4.

If the user edited a file themselves, audit that step first, then rerun the steps after it.
