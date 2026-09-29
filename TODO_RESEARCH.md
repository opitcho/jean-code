# Research evals: TODO

Use cases to build evals for the `research` subagent (`src/research.py`). Each run leaves a folder (`brief.md`,
`facts.jsonl`, `pages/`, `draft.md`, `agents/`), so a grader can score any run after the fact. Where a case needs
control over what the web says, serve fixed pages in place of `pages.read_page` and `tools.web_search`, as the tests
do.

Observations cited below come from run `workspace/research/20260929-113342-191733` (world models for sim-to-real);
`.experiments/research_draft_process_audit.py` reproduces them.

## Asked for

- [ ] **Changing narrative.** When the facts change while the report is being written, the narrative changes too.
  - Observed: the Summary's verdict was written after round 1 (94 of 145 facts) and kept its wording ("real but
    narrow") through two more rounds.
  - Test: serve pages so round 1 supports conclusion A, and a later round's pages clearly overturn it. Also: rerun
    the final write from all facts with no draft, and compare the verdicts.
  - Metric: the final Summary/Verdict states the later conclusion; no section still argues A without saying it was
    overturned.

- [ ] **Dealing with contradictory facts.** Two sources disagree (numbers, claims, or an old figure a newer one
  replaces).
  - Test: plant pairs of pages that contradict each other, some of equal standing and some where one is clearly
    newer or primary.
  - Metric: both sides are cited and the conflict is named in the body or Caveats; neither side is silently dropped;
    where one source is newer or primary, the report says which it prefers and why.

- [ ] **Accurate citations.** Each cited sentence says what its facts support, and each fact's claim says what its
  quote supports.
  - Observed: only quotes are checked. Claims can add to their quote (F75's claim adds "from the AgiBot Genie
    Team"), and nothing checks a sentence against the fact it cites.
  - Metric: for every cited sentence, a judge rates whether its cited quotes support it (not the claims). Numbers are
    compared exactly: every number in a sentence appears in a cited quote. Also: the rate of claims that go beyond
    their quote.

## Other ideas

- [ ] **Knowledge from memory.** The report states only what the facts say.
  - Observed: 15 paragraphs outside Caveats cite nothing, including the verdict bullet and "most industrial and
    household assembly — sits outside the confirmed horizon".
  - Test: a made-up topic (a fictional library or company) served from fixed pages whose facts contradict what a
    model would guess.
  - Metric: no uncited claims; nothing in the report that isn't on a served page.

- [ ] **Leading briefs.** The lead's briefs for later rounds are neutral, and at least one looks for evidence against
  the current draft.
  - Observed: the round-3 brief asked for "canonical, quotable, quantitative statements that learned dynamics models
    become unreliable", naming papers the lead remembered.
  - Metric: a judge rates each later-round brief as neutral or leading, and checks whether any brief asks for
    evidence against the draft.

- [ ] **Reading past the first window.** Researchers find evidence in results and limitations sections, not only in
  abstracts and introductions.
  - Observed: 26 of the 38 pages longer than one window were read only from the start.
  - Test: plant the key caveat far into a long page.
  - Metric: the caveat is recorded and reaches the report.

- [ ] **Search snippets used as sources.** Quotes come from what was fetched, not from search snippets.
  - Observed: F4 was copied from a search snippet; its sentence starts at character 9,096 and only 0–8000 was
    fetched.
  - Metric: the share of ⚠ facts whose quote appears in an earlier `web_search` result but not in the fetched text.

- [ ] **Quote-check accuracy.** A regression set of real quote and page pairs, false negatives and false positives.
  - Test: start from the `tests/fixtures/` excerpts, and add sabotaged quotes like the ones in
    `.experiments/research_quote_check_fixes.py`.
  - Metric: the share of ⚠ facts that are actually on the page (false negatives), and the share of sabotaged quotes
    that verify (false positives).

- [ ] **Using ⚠ facts.** ⚠ facts are cited only when nothing better exists, and are always named in Caveats.
  - Metric: every ⚠ cited in the body appears in Caveats; unused facts aren't mostly ⚠ just because the quote check
    failed.

- [ ] **Coverage of the brief.** Every in-scope item and every Deliverable item in the brief has cited facts, or is
  listed as a gap in Caveats.
  - Metric: per item, covered / listed as a gap / missing.

- [ ] **Questions with no good answer.** The evidence doesn't exist, or the question rests on a false premise.
  - Metric: the report says so, and doesn't pad with weak or tangential facts.

- [ ] **Hedges kept.** Qualifiers in sources ("author-reported", "in simulation", "on the hard subset") survive into
  the report.
  - Metric: a judge compares each cited sentence's hedging with its quote's; count the hedges dropped.

- [ ] **Duplicated work.** The same fact recorded twice, or the same ground covered by two researchers.
  - Observed: F86/F91 and F101/F104 have identical quotes; F106/F113 too.
  - Metric: duplicate quotes per run, and pages fetched by more than one researcher.

- [ ] **Stopping early.** A run cut off at half its budget still leaves a report that is coherent and cited, even if
  thin.
  - Metric: same graders as above, on a run stopped early; the Summary doesn't claim coverage it lacks.

- [ ] **Cost.** Facts recorded, verified and cited per dollar, to compare prompts and models, alongside the quality
  metrics above.
