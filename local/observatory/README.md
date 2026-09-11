# Dumè GPT — Corsica well-being observation assistant

A calm, institutional web front end for decision-makers in Corsican
municipalities (mayors, town-hall secretaries). One question in, one composed
answer out, with a staged progress indicator while it works.

It wraps the existing **RAPTOR v10** pipeline (`rag_v10_raptor_subq.py`) — **no
RAG source file is modified**. Model / pipeline selection exists but is
**hidden**: ordinary visitors always get the production configuration.

```
local/observatory/
├─ server.py            FastAPI host (routes + SSE)
├─ pipeline_manager.py  loads v10 (+ v11), model/provider shims, progress, commune data
├─ ui/index.html        the single-page interface (self-contained, English)
├─ run_observatory.ps1  launcher
└─ README.md
```

## Run

```powershell
.\local\observatory\run_observatory.ps1
```

or directly:

```powershell
.\.venv\Scripts\python.exe .\local\observatory\server.py
```

Open <http://127.0.0.1:8600/>. The RAPTOR pipeline loads in a background thread
(~20–60 s); the page shows *Preparing data…* until `/api/health` reports ready.

### Requirements

* `CLAUDE_API_KEY` and `MISTRAL_API_KEY` in the repo-root `.env` (the keys
  `rag_v10_raptor_subq.py` already uses). `OPENAI_API_KEY` is optional — only
  needed if you switch the support model to an OpenAI one in the console.
* The `.venv` environment (FastAPI, uvicorn, anthropic, openai — already installed).
* `chroma_portrait/` with `portrait_verbatims` + `raptor_summaries` (already present).
* `df_mean_by_commune.csv` at the repo root — feeds the sidebar indicators
  (68 communes). If absent, the sidebar simply hides the indicator panel.

> **Mistral is out of the default path.** The workspace behind `MISTRAL_API_KEY`
> has no access to `mistral-large-*` (`tier_not_allowed`) and no completion quota
> on the rest (`429`). Swapping keys doesn't help — it's the account plan. So the
> **default** now runs the three roles on OpenAI + Anthropic (all keys working):
> decomposition `gpt-4o-mini`, answering `claude-haiku-4-5-20251001`, synthesis
> `gpt-4o`. The exact thesis config (Mistral Large ×2 + Claude Haiku) is one
> click away in the console — **"Reference pipeline (thesis)"** — and will work
> again the moment a Mistral plan is active.

## The interface

* **Chat thread** — your question as a green bubble, the answer as flowing serif
  text under the *Dumè GPT* avatar, with a timestamp.
* **Staged progress** — *Understanding your question → Consulting interviews &
  survey data (point i of n) → Drafting the answer*, plus an elapsed counter.
* **Follow-up chips** — the sub-questions the pipeline generated, clickable.
* **Sidebar** — pick any of the **360 Corsican communes**: shows its overall
  OppChoVec rank as text, and the embedded **OppChoVec dashboard** (see below)
  as a live choropleth map, both updating together. The selector is **context
  only** — it never changes the answer; the question reaches the pipeline
  verbatim and v10's own commune detector decides what it's about.
* **Recent conversations** — stored in this browser's `localStorage`; click one
  to reopen it. `+` (top right) starts a new conversation, which also clears
  conversation memory (see below).
* **Conversation memory** — follow-ups work ("what about Bastia?" after asking
  about Ajaccio). The browser sends its last 3 exchanges back with each new
  question; one small LLM call (reusing the decomposer model, no extra
  console setting) folds them into a standalone question *before* it reaches
  the RAPTOR pipeline — v10 itself never sees the conversation and stays
  single-turn/stateless. No server-side session or persistence: it's exactly
  what's on screen, sent once. Invisible to ordinary visitors; the `?dev=`
  inspector shows the original → standalone rewrite for each answer.
* Answers are steered to **English** regardless of the corpus language.

## The OppChoVec dashboard (`local/observatory/dashboard/`)

A separate, static, no-backend Leaflet dashboard — 11 tabs (OppChoVec/Opp/
Cho/Vec choropleths, LISA spatial clustering, CAH hierarchical clustering,
correlations, Parangons/archetypes, Data Viz, Entreprises) — copied wholesale
from `C:\These\visu_oppchovec_propre` (its own repo,
`github.com/Ghinc/oppchovec_visu`) and mounted at `/dashboard/`. Its own
`script.js` (~208KB) is untouched; only two small additive files integrate it:

* **`embed.js`** — `?embed=1` hides the dashboard's own sidebar/tab bar for a
  compact map-only view (used by the sidebar iframe) with an "Open full
  dashboard ↗" link back to the unmodified 11-tab experience; `?commune=<name>`
  preselects a commune via the dashboard's own `afficherCommune()` global.
* A banner in its stub **"DumèGPT" chat tab** (originally a fake typing
  simulation, never wired to a backend) now links back to the real chat here
  (`target=_top`) — the tab's original DOM was left intact so `script.js`'s
  existing event listeners don't throw.

The large route-network GeoJSON files (~29MB) and the `.xlsx` exports are
gitignored inside `dashboard/` (mirrors that project's own policy) but present
on disk from the copy; `Commune_Corse.geojson` (264KB, the commune boundaries
every map needs) is the one GeoJSON kept in git. Open `/dashboard/` directly
for the full, unembedded dashboard.

## The hidden model console

Add `?dev=<code>` to the URL, e.g. <http://127.0.0.1:8600/?dev=corse2026>.

The code is `OBS_DEV_CODE` (default `corse2026`; override with
`run_observatory.ps1 -DevCode …` or the env var). Without the correct code the
console is invisible **and** the server ignores any model config sent to
`/api/ask` — a crafted request can't reach it.

The v10 pipeline has **three LLM roles**, each set independently in the console
(any of Mistral / Claude / OpenAI × its models):

| Role | What it does | Default | Thesis reference |
|---|---|---|---|
| **Decomposition** | question → sub-questions | `openai / gpt-4o-mini` | `mistral / mistral-large-latest` |
| **Answering** | answers each sub-question from retrieved text | `claude / claude-haiku-4-5-20251001` | same |
| **Synthesis + scoring** | writes the final answer, scores the dimension | `openai / gpt-4o` | `mistral / mistral-large-latest` |

Plus: pipeline version (`v10`, `v11` if it loads), output language (`en` / `fr`),
`k` (1–20), sub-questions (1–8), temperature (blank = the pipeline's tuned values).

Two preset buttons:

* **Default (low-cost)** — the table's *Default* column. ~6 ¢/question; the
  answerer (Claude Haiku, ~35 k input tokens/question) is the cost driver, not
  the synthesis model.
* **Reference pipeline (thesis)** — restores the exact config behind every
  evaluation run so far (Mistral Large decomposition + Claude Haiku answering +
  Mistral Large synthesis). The "Applied" line shows **Reference pipeline** when
  that's active. Needs a working Mistral plan.

For the article's headline numbers, the closest working stand-in for Mistral
Large on the synthesis role is **Claude Sonnet 5** or `gpt-4o` (richer output
from Sonnet); `gpt-4o-mini` is fine for decomposition.

After each answer the console shows a **Last run** inspector: the three roles
used, per-step wall-time and token counts, the dimension score, the
sub-questions with their intermediate answers, and the raw sources JSON. Choices
persist in `localStorage`.

> 5-family models (Sonnet 5 / Opus 5) reject a fixed `temperature`; the wrapper
> drops it for those automatically.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OBS_PORT` | `8600` | HTTP port |
| `OBS_HOST` | `127.0.0.1` | bind address (keep local) |
| `OBS_DEV_CODE` | `corse2026` | unlock code for the model console |
| `OBS_ENABLE_V11` | `1` | also load the v11 agentic pipeline |
| `OBS_FORCE_CPU_EMBED` | `1` | pin embeddings to CPU (avoid fighting llama-server for the GPU) |

## API

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | the UI |
| `/dashboard/` | GET | OppChoVec dashboard (static; `?embed=1`, `?commune=<name>`) |
| `/api/health` | GET | `{ready, error, v11}` |
| `/api/examples` | GET | starter questions |
| `/api/communes` | GET | all 360 Corsican communes |
| `/api/commune?name=&view=` | GET | one commune's sidebar profile (`view` = `objective` / `subjective`) |
| `/api/config?code=` | GET | dev unlock + choice lists (code-gated) |
| `/api/ask` | POST | `{question, config?}` → SSE: `progress` / `result` / `error` |

Queries are serialised (the pipeline uses module-global state); a second
concurrent request gets a friendly "answering another question" error rather
than corrupting a run.
