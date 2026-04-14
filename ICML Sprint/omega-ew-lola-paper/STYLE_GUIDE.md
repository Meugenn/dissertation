# EW-LOLA Paper Style Guide for Future Agents

Use this file when editing the EW-LOLA paper materials.

## 0. Canonical Files And Venue Rules

- The canonical submission file is now `main.tex`, not `paper-draft.md`.
- Keep `paper-draft.md` and `MASTER_REPORT.md` as planning and long-form reference documents, but treat `main.tex` as the source of truth for the paper itself.
- The paper currently targets the official ICML 2026 LaTeX template via `icml2026.sty` and `icml2026.bst`.
- Keep the draft in anonymous review mode by default with `\usepackage{icml2026}` unless there is an explicit decision to switch to camera-ready mode.
- Respect ICML formatting constraints: 10 point Times-style font via the template, two-column layout, figure captions below figures, table captions above tables, and no manual spacing compression.
- Keep the main body within the ICML limit for submissions. References and appendices may extend beyond that limit, but they must remain in the same compiled PDF.
- Do not replace the official style files with custom local variants.

## 1. Voice

- Write like a careful mathematics and ML paper, not a blog post.
- Use plain declarative sentences. Lead with the claim, then support it.
- Keep the tone sober. If a result is inherited, say it is inherited. If a proof is pending, say it is pending.
- Use UK English unless the submission template or venue instructions force a change.

## 2. What To Optimise For

- Precision over flourish.
- Concrete theorem-to-source alignment.
- Honest scope control.
- Readability for technically literate workshop reviewers.

## 3. Structural Rules

- Keep the paper centred on one claim: EW and LOLA compose without interference.
- Distinguish clearly between:
  - inherited material from Giannou, the dissertation, and the technical report;
  - the new composition theorem for this paper;
  - planned versus completed empirical reruns.
- Prefer section openings that say what the section proves or argues.
- Keep proofs in the main body as proof sketches unless the algebra is truly one line.
- Push long derivations, measure-theoretic detail, and implementation clutter to appendices.

## 4. Sentence-Level Guidance

- Prefer short to medium sentences with one main claim each.
- Use commas and full stops more than em dashes.
- Avoid rhetorical questions.
- Avoid suspense setups and dramatic reveals.
- Do not write as if you are teaching a novice audience unless the sentence really needs a definition.

## 5. Tropes To Avoid

The `tropes.md` file is binding. In practice this means:

- Do not use phrases like "delve", "quietly", "robust" as padding, "serves as", "here's the thing", or "it is worth noting".
- Do not write in the "not X, but Y" pattern unless there is a real logical contrast and it is used sparingly.
- Do not stack tricolons or anaphora for effect.
- Do not manufacture drama with fragments like "The result? Strong."
- Do not inflate stakes beyond what the theorem or experiment supports.
- Do not invent labels such as "variance paradox", "spectral trap", or similar shorthand unless they are real terms in the literature.

## 6. Preferred Paper Style

- Name the theorem, give the condition, state the consequence.
- When citing prior work, say exactly what it proved and what it did not prove.
- When discussing experiments, separate setup, prediction, and observed result.
- When discussing Lean, say exactly what is formalised and exactly what still has `sorry`.
- When describing novelty, keep the claim narrow and checkable.

## 7. Notation And Formatting

- Keep notation consistent with Giannou et al. and the technical report whenever possible.
- Use straight ASCII punctuation in markdown drafts.
- Avoid unnecessary Greek symbols in prose if the ASCII spelling is just as clear.
- Reserve displayed equations for definitions, theorem statements, and indispensable update rules.
- Avoid bold-first bullets. If a list is needed, keep the bullet text plain.

## 8. Source Hygiene

- Do not imply that a theorem is proved in the workshop paper if the current draft only has a proof sketch.
- Do not imply that experiments were rerun unless they were rerun in this workspace or you have the exact output in hand.
- When a claim comes from the technical report, the dissertation, Kim et al. (2021), or Foerster et al. (2018), keep that provenance visible.
- If future edits change theorem statements, re-check them against the source PDFs before polishing prose.

## 9. A Good Default Paragraph Shape

One useful pattern for this project:

1. First sentence: state the point of the paragraph.
2. Second and third sentences: give the technical reason or cite the source theorem.
3. Final sentence: explain why the point matters for this paper.

If the paragraph starts sounding like marketing copy, rewrite it.
