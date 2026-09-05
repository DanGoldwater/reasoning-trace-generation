---
marp: true
theme: gaia
paginate: true
---

# Supervised Fine Tuning Data Pipeline

---

Goals:

- Inference with structured output and reasoning traces
- Filter by correctness of structure
- Filter by correctness of answer
- Filter by hallucination / reasoning quality

---

Basic approach:

- Lean hard on programmatic checks (ty, basedpyright, ruff)
- Plenty of unit tests
- integration tests which _do not_ mock ollama.
- Utilise pydantic framework instead of string manipulation


---

Basic architecture

![h:450](figures/architecture.svg)

---

Hallucination filtering

![h:450](figures/filtering.svg)

---

![h:450](figures/architecture2.svg)

---

Hallucination failure rate
![h:450](figures/hallucination.svg)

---

Problems

- Many of the answers are constrained to a very small range of categorical answers. This means a correctness gate is unlikely to provide the protection we might expect it to.
- Reasoning traces from the Qwen model show a _lot_ of hallucinations.
- Often found that the model would hit token limit without supplying an answer. Increasing token limit didn't seem to help - started looping.
- Hallucination filtering difficult without source material.
- With 4b model, only 3/10 questions passed (without ljudge). Mostly on token limits / spiralling. 


---

How would we improve this?

- Constrained optimisation over hyperparameters:
  - Max tokens
  - System prompt (good candidate for GEPA)
  - Inclusion / exclusion of supporting information

Could optimise for:

- Time
- Lval score of reasoning traces
- Number of accepted reasoning traces
