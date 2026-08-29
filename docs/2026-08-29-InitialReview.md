# Initial Review: PaddleOCR-VL Document Extraction POC

- **Date:** 2026-08-29
- **Reviewer:** Saurabh (with automated code review)
- **Audience:** the intern who built this POC. This document explains what is wrong, why it is wrong, and gives you a step-by-step checklist to fix it. Every task is something you do yourself. Ask questions when stuck, but read the linked docs first.

---

## 1. Summary

The POC demonstrates the right idea (OCR a document, extract fields against a schema, keep provenance), but the repository has three structural problems and several code bugs:

1. **Files were copied from the PaddleOCR repository into this repo.** About 2.1 MB of upstream files (training configs, `setup.py`, `.gitignore`, and the upstream README) are committed here, and none of them are used by our code.
2. **The most important step of the pipeline is missing from the repo.** The scripts read `output/invoice_res.json` and `output/invoice.md`, but nothing in the repo creates those files. The PaddleOCR-VL call was run by hand and never committed, so nobody else can reproduce the pipeline.
3. **The repo has no standard Python project structure.** There is no dependency file, no tests, no package layout, and five scripts that run code at import time.

The checklist in Section 7 fixes all of this. Work through it top to bottom. Every task is one atomic git commit with a given commit message, and each ends with a verification step so you can prove to yourself it worked.

---

## 2. What the review found (with evidence)

### 2.1 Copied upstream files (should never have been committed)

| File / directory | What it is | Evidence it is unused |
|---|---|---|
| `configs/` (except `extraction_schema.json`) | 157 files (2.1 MB) from PaddleOCR's `configs/` tree: 155 training YAMLs (det/rec/cls/kie/table), a config-generator script, and a `.gitkeep` | No script references any of them. They configure *training* of the classic PP-OCR models and are irrelevant to PaddleOCR-VL *inference* |
| `setup.py` | Copied from PaddleOCR (it still has the PaddlePaddle copyright header). It contains an empty `setup()` call | This repo is not a package, and an empty `setup()` does nothing |
| `.gitignore` | PaddleOCR's own `.gitignore` | It ignores `deploy/android_demo/`, `test_tipc/web/` and other paths that do not exist in this repo |
| `README.md`, lines 92–415 | The entire upstream PaddleOCR README (banners, badges, release notes, citations) concatenated after our 91-line POC README | Upstream branding presented as ours; it buries the actual POC documentation |

**Why this matters:** committing someone else's files into your repo (called "vendoring") means you now silently own them. They go stale, they bloat every clone, they confuse readers about what this repo actually is, and they carry a license you must now track. If you need another project's code, you either install it as a dependency or reference it as a git submodule (Section 4 explains which to use here).

### 2.2 The pipeline is not reproducible

`run_pipeline.py`, `schema_extract.py`, `provenance.py`, and `provenance_bbox.py` all read files from `output/`, which is gitignored. The README says "First run PaddleOCR-VL on the input document and generate the output files" — but the command or code that does this was never committed. A pipeline whose first stage exists only in your shell history is not a pipeline. Rule of thumb: **if a fresh clone plus the README cannot reproduce your result, the work is not done.**

### 2.3 PP-DocLayoutV2 was never used

The task asked for PaddleOCR-VL **and** PP-DocLayoutV2. There is no trace of layout-detection work in the repo. The good news (Section 3) is that PP-DocLayoutV2 is already *inside* the PaddleOCR-VL pipeline, so you get it almost for free once the pipeline is invoked properly.

### 2.4 Code bugs (all verified by running/reading the code)

| # | Location | Bug |
|---|---|---|
| B1 | `run_pipeline.py:26` | The loop iterates over schema fields but indexes a hardcoded dict with `patterns[field]`. Adding any new field to `configs/extraction_schema.json` crashes with `KeyError`. `schema_extract.py:23` already uses the safe `patterns.get(field)` — the two copies diverged |
| B2 | `run_pipeline.py:20,35` | The `total_amount` regex is `(.+)` here but `([\d,]+)` in `schema_extract.py`. With input like `Total Amount: $52,000`, `float("$52000")` raises `ValueError` (verified at runtime) |
| B3 | `create_invoice.py:31` | `img.save(r"samples\invoice.png")` is a Windows path. On Linux, backslash is a normal filename character, so it creates a stray file literally named `samples\invoice.png` and never regenerates `samples/invoice.png` |
| B4 | all four extraction scripts | The `patterns` dict is copy-pasted into `run_pipeline.py`, `schema_extract.py`, `provenance.py`, and `provenance_bbox.py`, and has already diverged (see B2). One source of truth, four stale copies |
| B5 | all five scripts | Everything runs at import time: no `main()` function, no `if __name__ == "__main__":` guard. Importing any of these modules (for example, to reuse the patterns) executes the whole pipeline as a side effect |
| B6 | `run_pipeline.py:29` | Missing fields are written as bare `null` while found fields are `{"value": ..., "provenance": ...}` objects. Consumers doing `result[field]["value"]` crash on the `null`. Also, the `INPUT` constant on line 5 is never used |

You will fix all six in Phases 3 and 4 of the checklist.

---

## 3. Background: what these two components actually are

Read this section carefully — it corrects a misunderstanding in the original task description and will save you from integrating something that is already integrated.

### 3.1 PaddleOCR-VL

- **What it is:** a document-parsing pipeline built around PaddleOCR-VL-0.9B, a small vision-language model (a NaViT-style visual encoder plus the ERNIE-4.5-0.3B language model). It parses a document image or PDF into structured elements: text, tables, formulas, charts, in reading order, across 109 languages.
- **Where it lives:** inside the main PaddleOCR repository, https://github.com/PaddlePaddle/PaddleOCR (there is no separate PaddleOCR-VL repo). The pipeline class is `paddleocr/_pipelines/paddleocr_vl.py`.
- **How you install it:** `pip install "paddleocr[doc-parser]"` (in this project we use Poetry, so: `poetry add "paddleocr[doc-parser]"`), plus the PaddlePaddle framework `>= 3.2.1`. GPU (Compute Capability >= 7.0, CUDA >= 11.8) is the primary target; CPU works but is slow for the 0.9B model.
- **Docs:** https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html

### 3.2 PP-DocLayoutV2

- **What it is:** a layout-analysis model. It detects 25 kinds of regions on a page (text, table, image, formulas, headers, footers, seals, charts, ...) *and* predicts their reading order. Architecturally it is an RT-DETR-based detector followed by a small pointer network that orders the detected boxes.
- **Where it lives:** the runnable model ships inside the `paddleocr` pip package (weights auto-download from Hugging Face: https://huggingface.co/PaddlePaddle/PP-DocLayoutV2). **Note:** the PaddleDetection link in the original task (`PaddleDetection/tree/develop/configs/doclayout`) returns 404 — that directory does not exist. PaddleDetection is the training framework behind these models, but you do not need it (or its configs) for inference. If we ever fine-tune layout detection on our own documents, the documented route is through PaddleX, which wraps PaddleDetection internally.
- **Docs:** https://www.paddleocr.ai/main/en/version3.x/module_usage/layout_analysis.html

### 3.3 The key fact: they are already integrated

The PaddleOCR-VL pipeline is two stages, and **stage 1 *is* PP-DocLayoutV2**:

```
document image/PDF
      |
      v
Stage 1: PP-DocLayoutV2         -> detects regions + reading order, crops each region
      |
      v
Stage 2: PaddleOCR-VL-0.9B (VLM) -> recognizes each crop (text/table/formula/chart)
      |
      v
merged result in reading order  -> parsing_res_list (JSON), markdown export
```

So "build a pipeline using PaddleOCR-VL and PP-DocLayoutV2" does not mean wiring two packages together yourself. It means invoking `PaddleOCRVL()` (which uses PP-DocLayoutV2 internally) and, where useful, also calling the layout module directly for layout-only analysis or debugging:

```python
# The full pipeline (uses PP-DocLayoutV2 internally as stage 1):
from paddleocr import PaddleOCRVL
pipeline = PaddleOCRVL()
output = pipeline.predict("samples/invoice.png")
for res in output:
    res.save_to_json(save_path="output/")      # parsing_res_list with block_bbox etc.
    res.save_to_markdown(save_path="output/")  # markdown for text-level extraction

# The layout model alone (useful to visualize/debug stage 1):
from paddleocr import LayoutDetection
layout = LayoutDetection(model_name="PP-DocLayoutV2")
for res in layout.predict("samples/invoice.png", batch_size=1):
    res.save_to_img(save_path="output/")       # page image with drawn boxes
    res.save_to_json(save_path="output/")
```

The JSON result's `parsing_res_list` contains one entry per detected block with `block_label`, `block_content`, and `block_bbox` — exactly the keys your provenance scripts already consume.

---

## 4. Third-party code: dependency vs. submodule (and why we vendor neither)

Three ways to use someone else's code, from best to worst for our case:

1. **A pip/Poetry dependency (what we will do).** `paddleocr` is published on PyPI. Declaring it in `pyproject.toml` gives us pinned versions, clean upgrades, and zero foreign files in our repo. This is the right choice whenever the upstream project publishes a package — and PaddleOCR does.
2. **A git submodule under `third_party/` (only if we need upstream *source*).** A submodule is a pointer (a commit hash) to another repository, checked out into a subdirectory. The files are not copied into our history; `git clone --recurse-submodules` fetches them. Use this only when you need to read or build upstream source that is not shipped in the pip package, for example if we later fine-tune PP-DocLayout models and want the PaddleX/PaddleDetection source pinned next to our training scripts. Until that day, we do not add any submodule: an unused submodule is clutter with extra ceremony.
3. **Copying files into the repo (what happened, called vendoring).** Never do this without an explicit decision and a recorded license. It is what this review is cleaning up.

**Decision for this repo:** consume `paddleocr[doc-parser]` (and `paddlepaddle`) as Poetry dependencies. Delete every copied upstream file. Create `third_party/` only when a concrete need for upstream source exists, and then only as submodules with a `third_party/README.md` explaining what each one is for and which commit is pinned. Since our own code lives in its own package directory, the day a submodule is added nothing needs restructuring.

---

## 5. Target repository structure

We follow https://github.com/saurabheights/cookie-cutter-python (our standard template): Poetry with a PEP 621 `[project]` table, a flat package directory (here `document_extraction/` — a valid-identifier form of the project name, instead of a `src/` layout), `tests/`, and ruff + mypy + pytest + pre-commit configured in `pyproject.toml`. Since this repo is a POC and not an installable package, we keep `package-mode = false` in the Poetry config and do not publish anything.

Tools glossary (first-time readers):

- **Poetry** manages dependencies and the virtualenv from `pyproject.toml` (https://python-poetry.org).
- **ruff** is a linter and code formatter; **mypy** is a static type checker.
- **pre-commit** runs those checks automatically on every `git commit` (https://pre-commit.com).
- **Taskfile** is a command shortcut file for the go-task runner (https://taskfile.dev); optional convenience.
- **GitHub Actions** runs lint and tests in the cloud on every push; the workflow lives in `.github/workflows/`.
- **Conventional Commits** is the commit-message convention we use (https://www.conventionalcommits.org).

```
PaddleOCR-VL-Document-Extraction/
├── .github/workflows/ci.yml         # lint + tests on push/PR
├── .gitignore                       # OUR ignores: output/, .venv/, caches, __pycache__/
├── .pre-commit-config.yaml          # ruff + ruff-format hooks
├── README.md                        # POC readme ONLY (~90 lines), links to upstream
├── Taskfile.yml                     # task lint / format / test / typecheck
├── pyproject.toml                   # deps: paddleocr[doc-parser], paddlepaddle, pillow
├── poetry.lock
├── configs/
│   └── extraction_schema.json       # the ONLY file that survives from configs/
├── docs/
│   └── 2026-08-29-InitialReview.md  # this document (already committed by the reviewer)
├── samples/
│   ├── invoice.png
│   └── invoice.txt
├── tests/
│   ├── fixtures/                    # committed OCR outputs used as test data
│   ├── test_extraction.py           # schema extraction against a fixture markdown
│   └── test_provenance.py           # provenance matching against a fixture JSON
└── document_extraction/             # our package (flat layout, per the template)
    ├── __init__.py
    ├── patterns.py                  # THE single field->regex mapping (fixes B4)
    ├── ocr.py                       # runs PaddleOCRVL, saves json+markdown (fixes 2.2)
    ├── layout.py                    # optional: PP-DocLayoutV2 alone, for debugging
    ├── extract.py                   # schema-driven extraction (fixes B1, B2)
    ├── provenance.py                # bbox provenance from parsing_res_list (fixes B6)
    ├── pipeline.py                  # main(): ocr -> extract -> provenance, __main__ guard
    └── create_sample.py             # invoice generator (fixes B3), __main__ guard
```

Notes for a first-time reader:

- **Why a package directory instead of loose scripts?** So code can be imported without side effects (`from document_extraction.patterns import PATTERNS`) and tested. Every module gets a `main()` plus `if __name__ == "__main__": main()` guard; no code runs at import time.
- **Why `package-mode = false`?** Poetry then manages only dependencies and the virtualenv; it will not try to build/install the project as a distributable package. That is exactly what a POC needs.
- **`output/` stays gitignored** — generated artifacts never get committed. What changes is that the code to *generate* them is now in the repo. Test fixtures under `tests/fixtures/` ARE committed: they are input data for tests, not generated artifacts.

---

## 6. The full pipeline, end to end

After the migration, one command reproduces everything from a fresh clone:

```bash
poetry install
poetry run python -m document_extraction.pipeline samples/invoice.png
```

which runs these stages. To keep responsibilities clear, each module exposes one function, and `pipeline.py` chains them in memory; only `ocr.py` and `pipeline.py` write files:

1. **`ocr.py` — Document parsing.** `parse_document(image_path: Path, out_dir: Path) -> dict` runs `PaddleOCRVL().predict(...)`, calls `save_to_json()` and `save_to_markdown()` into `out_dir`, and returns the parsed result. Internally this is PP-DocLayoutV2 (layout + reading order) followed by the 0.9B VLM (recognition).
2. **`extract.py` — Schema-driven extraction.** `extract_fields(markdown: str, schema: dict) -> dict` looks up each schema field's regex in `patterns.PATTERNS` with `.get()` (unknown fields produce `{"value": None, "provenance": None}` plus a logged warning, never a crash) and coerces `"number"` fields. The `total_amount` pattern must tolerate a currency symbol: `r"Total Amount:\s*\$?\s*([\d,.]+)"`.
3. **`provenance.py` — Grounding.** `attach_provenance(fields: dict, parsing_res_list: list) -> dict` finds, for each extracted field, the block whose `block_content` matches, and attaches `page`, `source_text`, `block_bbox`, `block_id`, `block_label`. Found and missing fields share one output shape: `{"value": ..., "provenance": ...}`.
4. **`layout.py` (optional but instructive).** Run `LayoutDetection(model_name="PP-DocLayoutV2")` on the same image and save the visualization with `save_to_img()`. Compare its boxes with the bboxes in your provenance output — they come from the same stage-1 model, which is the most concrete way to *see* that PP-DocLayoutV2 is already inside your pipeline.

Later extensions (not in this checklist): multi-page PDFs (`page` is hardcoded to 1 today), fuzzy matching instead of regex-only extraction, and accuracy evaluation against labeled documents.

---

## 7. THE CHECKLIST — one task, one commit

Rules before you start:

- **Every task below is one atomic git commit.** Atomic means: the commit does exactly one thing, its message says what that one thing is, and the repository still works after it (nothing imports a file that no longer exists, and tests that exist stay green). If you notice you are typing "and" into a commit message, you are probably squashing two tasks into one.
- Commit messages follow Conventional Commits (`chore:`, `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `ci:` — see https://www.conventionalcommits.org). The exact message to use is given for each task.
- Stage only the files the task names (`git add <file>...`), never `git add -A`.
- **Branch and PR model:** create one branch, `git checkout -b cleanup/repo-structure`. After finishing Phase 1, push it and open ONE pull request on GitHub (after `git push -u origin cleanup/repo-structure`, GitHub shows a "Compare & pull request" button). At the end of each later phase, push again — the same PR updates automatically. If you have never opened a PR, read https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request first.
- This review document is already committed to `main` by the reviewer; you do not commit it.

### Phase 0 — Understand before touching (no commits)

- [ ] **T0.1** Read Section 3 of this document, then the PaddleOCR-VL pipeline docs (https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html) end to end.
- [ ] **T0.2** Read what a git submodule is (https://git-scm.com/book/en/v2/Git-Tools-Submodules), then write 3–5 sentences in your own words in a local notes file: why vendoring the PaddleOCR files was wrong, and why for this repo a pip dependency beats a submodule. You will paste this into the PR description when you open the PR after Phase 1.
- [ ] **T0.3** In the PaddleOCR GitHub repo, open `paddleocr/_pipelines/paddleocr_vl.py` and identify where stage 1 (layout) and stage 2 (VLM recognition) happen. Hint: read the class's `predict` method and search the file for `layout` and for the VLM/recognition call. Note the line numbers in your notes file for the PR description.

### Phase 1 — Remove the copied upstream files (4 commits)

- [ ] **T1.1** Run `du -sh .` and note the size. Delete every file under `configs/` except `configs/extraction_schema.json`.
      Verify: `find configs -type f` lists exactly one file; `du -sh .` shrank by about 2 MB versus your noted value.
      Commit: `chore: remove vendored PaddleOCR training configs`
- [ ] **T1.2** Delete `setup.py` (it is upstream's, and its empty `setup()` does nothing).
      Verify: `git grep -l "PaddlePaddle Authors" -- ':!docs/'` returns nothing (the `':!docs/'` part excludes this review document, which mentions that string).
      Commit: `chore: remove upstream setup.py`
- [ ] **T1.3** Truncate `README.md` after line 91 (the upstream banner `<div align="center">` starts at line 92; keep only the POC content above it). Add one line linking to https://github.com/PaddlePaddle/PaddleOCR as the upstream project.
      Verify: `wc -l README.md` is under 100 and the file no longer contains `<div`.
      Commit: `docs: remove concatenated upstream README, link to PaddleOCR instead`
- [ ] **T1.4** Replace `.gitignore` with one written for THIS repo: `output/`, `__pycache__/`, `*.py[cod]`, `.venv/`, `.cache/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.idea/`, `.vscode/`.
      Verify: `git check-ignore output/x` matches; no rule mentions `android_demo` or `test_tipc`.
      Commit: `chore: replace upstream .gitignore with project-specific one`
      Then push the branch and open the PR (see the branch and PR model above); paste your T0.2/T0.3 notes into its description.

### Phase 2 — Standard project skeleton, per cookie-cutter-python (5 commits)

The template repo stores files with `{{cookiecutter.project_slug}}` placeholders, so you cannot copy them verbatim. Where a task says "from the template", first generate a real project once in a scratch directory and take files from there:

```bash
cd /tmp && pipx run cookiecutter gh:saurabheights/cookie-cutter-python
# answer the prompts with throwaway values; a real project directory appears
```

- [ ] **T2.1** Create `pyproject.toml`. Start from this content (adapted from the template for a non-installable POC), not from the raw template file:

      ```toml
      [project]
      name = "paddleocr-vl-document-extraction"
      version = "0.1.0"
      description = "PaddleOCR-VL + PP-DocLayoutV2 document extraction POC"
      requires-python = ">=3.10"
      dependencies = []

      [tool.poetry]
      package-mode = false

      [tool.pytest.ini_options]
      addopts = "--cov=document_extraction"
      cache_dir = ".cache/pytest"

      [tool.mypy]
      cache_dir = ".cache/mypy"

      [tool.ruff]
      line-length = 120
      cache-dir = ".cache/ruff"
      ```

      Verify: `poetry check` passes.
      Commit: `chore: add pyproject.toml per cookie-cutter-python template`
- [ ] **T2.2** Add the runtime dependencies. First check for a GPU: run `nvidia-smi` — if it prints a table with a CUDA version >= 11.8, you have a usable GPU; if it errors or is not found, use the CPU wheel.
      CPU: `poetry add paddlepaddle "paddleocr[doc-parser]" pillow`
      GPU: `poetry source add --priority explicit paddle https://www.paddlepaddle.org.cn/packages/stable/cu126/`, then `poetry add --source paddle paddlepaddle-gpu`, then `poetry add "paddleocr[doc-parser]" pillow` (the GPU wheel is not on PyPI, which is why the extra package source is needed; check `poetry source add --help` if the syntax differs in your Poetry version).
      Stage both `pyproject.toml` and `poetry.lock`.
      Verify: `poetry run python -c "import paddleocr; print(paddleocr.__version__)"` prints a 3.x version.
      Commit: `chore: add paddleocr[doc-parser], paddlepaddle, and pillow dependencies`
- [ ] **T2.3** Add dev dependencies as a PEP 735 `[dependency-groups]` table. Poetry 2.x group flags do not manage this table, so edit `pyproject.toml` by hand — add:

      ```toml
      [dependency-groups]
      dev = ["pytest>=8", "pytest-cov>=6", "ruff>=0.8", "mypy>=1.14", "pre-commit>=4"]
      ```

      then run `poetry lock` and `poetry install` (no flags).
      Verify: `poetry run pytest --version` and `poetry run ruff --version` work.
      Commit: `chore: add dev dependency group (pytest, ruff, mypy, pre-commit)`
- [ ] **T2.4** Copy `.pre-commit-config.yaml` from your generated scratch project, and create this minimal `Taskfile.yml` (the template's Taskfile uses Docker targets we do not need; running `task` commands requires the go-task binary from https://taskfile.dev/installation/ and is optional):

      ```yaml
      version: '3'
      tasks:
        lint:      { cmds: ["poetry run ruff check ."] }
        format:    { cmds: ["poetry run ruff format ."] }
        test:      { cmds: ["poetry run pytest"] }
        typecheck: { cmds: ["poetry run mypy document_extraction/"] }
      ```

      Run `poetry run pre-commit install`.
      Verify: `poetry run pre-commit run --files .pre-commit-config.yaml Taskfile.yml` passes. Do NOT run it with `--all-files` yet — the hooks would reformat the five legacy scripts, which Phase 3 is about to replace anyway. From now on the hooks run automatically on whatever you stage.
      Commit: `chore: add pre-commit hooks and Taskfile`
- [ ] **T2.5** Create the package: `document_extraction/__init__.py` (empty file).
      Verify: `poetry run python -c "import document_extraction"` succeeds.
      Commit: `chore: create document_extraction package skeleton`

### Phase 3 — Move and fix the existing code (4 commits)

The old scripts keep working until each one's replacement lands; each replacement commit deletes the script it replaces, so the repo never has two competing copies. These commits use `fix:` where behavior changes and `refactor:` where it does not.

- [ ] **T3.1** Create `document_extraction/patterns.py` containing the ONE `PATTERNS` dict (the single source of truth that fixes B4). Use `r"Total Amount:\s*\$?\s*([\d,.]+)"` for `total_amount` so a currency symbol cannot crash the number conversion (fixes B2).
      Verify: `poetry run python -c "from document_extraction.patterns import PATTERNS; print(sorted(PATTERNS))"` lists the five fields.
      Commit: `feat: add shared field-pattern module`
- [ ] **T3.2** Move `create_invoice.py` to `document_extraction/create_sample.py`: fix the save path to a portable one built with `pathlib` (`Path("samples") / "invoice.png"`) (fixes B3); wrap everything in `main()` with an `if __name__ == "__main__": main()` guard (B5). Delete `create_invoice.py` in this same commit.
      Verify: `poetry run python -m document_extraction.create_sample` updates `samples/invoice.png` (check its timestamp with `ls -l samples/`); then `poetry run python -c "import document_extraction.create_sample"` prints nothing and changes no timestamps.
      Commit: `fix: portable save path in invoice generator, moved into package`
- [ ] **T3.3** Rewrite `schema_extract.py` as `document_extraction/extract.py` with the `extract_fields()` interface from Section 6: import `PATTERNS`, look fields up with `.get(field)`, emit `{"value": None, "provenance": None}` for missing/unknown fields, log a warning for schema fields that have no pattern (fixes B1, B6). `main()` + guard. Delete `schema_extract.py` in this same commit.
      Verify: `poetry run python -c "import document_extraction.extract"` produces no output and no files.
      Commit: `fix: schema extraction tolerates unknown fields, replaces schema_extract.py`
- [ ] **T3.4** Merge `provenance.py` and `provenance_bbox.py` into `document_extraction/provenance.py` with the `attach_provenance()` interface from Section 6 (they do the same job at two levels of detail — keep the bbox version). `main()` + guard. Delete both old scripts in this same commit.
      Verify: import is side-effect free. If you still have the `output/invoice_res.json` you generated by hand for the original POC, run the module against it and compare with your old output; if you deleted it, defer this comparison until after T4.1 regenerates the file.
      Commit: `refactor: merge provenance scripts into document_extraction.provenance`

### Phase 4 — Make the pipeline reproducible: the missing OCR stage (4 commits)

- [ ] **T4.1** Write `document_extraction/ocr.py` with the `parse_document()` interface from Section 6: `main()` takes an input path argument (use `argparse`, the standard-library command-line parser), runs `PaddleOCRVL().predict(...)`, saves JSON and markdown to `output/`. This is the step that was missing from the repo.
      Verify: `poetry run python -m document_extraction.ocr samples/invoice.png` creates `output/invoice_res.json` and `output/invoice.md`.
      Commit: `feat: add PaddleOCR-VL invocation stage`
- [ ] **T4.2** Write `document_extraction/pipeline.py`: `main()` chains `parse_document` -> `extract_fields` -> `attach_provenance` in memory and writes `output/final_result.json`. Delete `run_pipeline.py` in this same commit (this is its replacement, with B1/B2/B6 fixed by construction).
      Verify: test from a fresh clone in a new directory:
      `git clone /path/to/this/repo /tmp/fresh-check && cd /tmp/fresh-check && git checkout cleanup/repo-structure && poetry install && poetry run python -m document_extraction.pipeline samples/invoice.png`
      Confirm `output/final_result.json` has all five fields with values and bboxes. Paste the JSON into the PR description.
      Commit: `feat: add end-to-end pipeline entrypoint, replacing run_pipeline.py`
- [ ] **T4.3** Write `document_extraction/layout.py`: run `LayoutDetection(model_name="PP-DocLayoutV2")` on the input and save the box visualization with `save_to_img()` to `output/`.
      Verify: compare the drawn boxes against the `block_bbox` values in your provenance output; write 2–3 sentences in the PR about what PP-DocLayoutV2 contributed.
      Commit: `feat: add PP-DocLayoutV2 layout visualization tool`
- [ ] **T4.4** Update `README.md`: how to install (the T2.2 commands), how to run (one command), and the expected output.
      Verify: follow the README yourself in a fresh shell (or hand it to someone else) and confirm the commands work exactly as written.
      Commit: `docs: rewrite README with reproducible install and run instructions`

### Phase 5 — Tests (2 commits)

- [ ] **T5.1** Save a real `output/invoice_res.json` and `output/invoice.md` as small fixtures under `tests/fixtures/` (fixtures ARE committed — they are test data, unlike `output/`). Add `tests/test_extraction.py`: extraction over the fixture markdown returns the five expected values; a schema field with no pattern yields `{"value": None, "provenance": None}` and does not raise; `total_amount` parses from `52,000` and from `$52,000` (the T3.1 pattern handles both).
      Verify: `poetry run pytest` is green.
      Commit: `test: add extraction tests with OCR output fixtures`
- [ ] **T5.2** Add `tests/test_provenance.py`: each field's provenance points at a block whose `block_content` contains the value; a field absent from the document yields the null-shaped object.
      Verify: `poetry run pytest` green; `poetry run ruff check .` and `poetry run mypy document_extraction/` pass.
      Commit: `test: add provenance tests`

### Phase 6 — CI and wrap-up (1 commit)

- [ ] **T6.1** Add `.github/workflows/ci.yml` running ruff + pytest on push/PR (start from your generated scratch project's workflow; a single Python version is fine, and skip the GPU-dependent OCR stage — CI only tests extraction and provenance against fixtures).
      Verify: CI is green on the PR.
      Commit: `ci: add lint and test workflow`
- [ ] **T6.2** (no commit) In the PR description, write a short retrospective: what you would do differently, and one thing you learned about layout analysis vs. recognition. Then request review.

**Definition of done:** a colleague with no context can clone the repo and get `output/final_result.json` with provenance by following the README commands; CI is green; `git grep -l "PaddlePaddle Authors" -- ':!docs/'` is empty; every module is importable without side effects.

---

## 8. References

- PaddleOCR repository (hosts PaddleOCR-VL): https://github.com/PaddlePaddle/PaddleOCR
- PaddleOCR-VL pipeline usage: https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html
- PP-DocLayoutV2 model card: https://huggingface.co/PaddlePaddle/PP-DocLayoutV2
- Layout analysis module usage: https://www.paddleocr.ai/main/en/version3.x/module_usage/layout_analysis.html
- PaddleOCR-VL technical report: https://arxiv.org/abs/2510.14528
- Project template: https://github.com/saurabheights/cookie-cutter-python
- Git submodules: https://git-scm.com/book/en/v2/Git-Tools-Submodules
- Creating a pull request: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
- Fine-tuning layout models (only if ever needed): https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/module_usage/tutorials/ocr_modules/layout_detection.en.md
