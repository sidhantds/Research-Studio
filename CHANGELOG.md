# Changelog

> **Research Studio turns a question into a reviewable research case:** editable plans,
> claim-level evidence, conflict detection, approval-gated actions and an exportable audit trail.
> Chat is the interface; the auditable research case is the product.

All changes to Research Studio, from the original command-line agent to the current web app.
Newest first. All work happened on 2026-11-18. Live code is on `main` at
https://github.com/sidhantds/Research-Studio.

For where the product is heading, and what is built versus planned, see
[Direction](#direction-from-chat-to-research-case) at the end.

---

## 11. Intent routing, one answer schema, follow-ups in the same session — *not yet committed*

**Intent router** (`agent.py`)
- Before anything runs, each request gets a `Route`: `response_type` (code, explanation, research,
  comparison, fact_check, clarification), `context_mode` (answer_from_session, research_further,
  new_research), `needs_tools` and a reason.
- Keyword rules decide the clear cases with **no model call**: all six example requests in the spec
  route correctly by rule. Only unclear requests (e.g. "Who founded it?") use one fast
  structured call.
- A first question that needs no research ("Generate hello world in Java") is answered directly:
  one call, no tools, no sources.

**One answer envelope** for every response type
- `Answer`: response_type, title, direct_answer, details, code (`CodeAnswer`: language, filename,
  code, run_instructions, expected_output, explanation), comparison, verdict, sources (`Source`:
  id, title, url, domain, quote, supports), follow_ups, context_mode, new_searches.
- **One card layout** for all types:
  - a header with the type badge, the language or topic, Copy and Save
  - the direct answer: a code block with run commands, a verdict, or a comparison table
  - Details, only for research, comparison and fact check
  - Evidence: each source with the passage it was cited for
  - status and Next steps
- Differences from the spec:
  - **`url` is a plain string** keeping only http(s) values, instead of `HttpUrl`, so one bad link
    can't fail the whole answer.
  - **`comparison` uses typed rows** (option, aspect, value), because Gemini's structured output
    can't fill free-form dicts.
- **Every answer is cleaned up before it's shown or stored:**
  - sources are renumbered 1..n and the `[n]` markers rewritten to match
  - sources the session never saw are dropped
  - a source's quote is kept only if it appears word for word in the retrieved text

**Follow-up modes** (above the composer)
- **Use current research** (default): answers from the session's evidence. It never searches. If
  new facts are needed, it says so and offers **Research further**.
- **Research further:** up to 2 new searches in the same session. A new topic is confirmed before
  starting a separate case.
- **New research:** a separate research case.
- After any search, the mode returns to *Use current research*, so each search is a deliberate
  choice.
- The old "Search automatically on follow-ups" setting is gone, because this replaces it.

**One active session**
- Follow-ups keep the session id (`thread_id`) and bump `version`. Each answer is added as a new
  message (kept from section 9, where answers stack like a chat) and only new sources are added.
- The status line shows how each answer was produced: *Answered from existing research · 0 new
  searches*, *Researched further · 2 new searches · 3 new sources*, or *Answered directly · no
  research needed*.
- Each run carries a `RunContext` (thread_id, response type, depth, sources, source rules, search
  budget) as LangChain run metadata. AgentExecutor has no checkpointer, so continuity lives in the
  app's session store, keyed by that `thread_id`.

**Structured evidence** (`evidence.py`, replaces the 6,000-character cap)
- Each lookup is split into one item per source: claim (the most relevant sentence), quote, URL,
  domain, retrieval time and a relevance score. No model calls are needed.
- A follow-up gets only:
  - the original question
  - the current answer (and code)
  - the 8 strongest items for that follow-up
  - the follow-up itself
- No raw pages, no full chat history. In testing, a code follow-up sent 451 characters of context.
- Raw lookups are still kept for the audit trail and for checking quotes word for word, but they
  aren't sent to the model.
- The run trace records how much context each run sent.

**Tested**, live:
1. **"Generate hello world in Java":** a code card with no searches.
2. **"Add exception handling":** same session, v2, 0 searches.
3. **"Latest Java features?" in *Use current research*:** it did **not** search and showed the
   notice.
4. **Research further:** 2 searches, 3 new sources, v3.
5. **"Compare Java 21 and 17":** a comparison table, 1 search.
6. ***New research*:** a second, separate session.

---

## 10. Research cases: claim-level evidence, review and audit — *not yet committed*

**Claims with quoted evidence**
- After every answer, a **separate** model call splits it into claims (up to 8). For each claim
  it quotes the passage that supports it and gives a verdict: *Supported*, *Partially
  supported*, *Unsupported* or *Insufficient evidence*, with a one-line reason.
- **The code checks every quote word for word** against the text the tools actually returned.
  If a claim marked Supported has a quote that isn't there, it's downgraded to Insufficient
  evidence. So the model can't vouch for its own claims unchecked.
- A passage that contradicts a claim is kept as a **conflict** (also checked word for word).
- A **Claims** panel under each answer shows verdicts, quotes and source links.

**Review queue**
- Claims that aren't fully supported, or that conflict with another source, wait for
  **Accept** / **Reject**. Rejected claims stay visible, struck through, and Undo works.
- The case status is **Needs review** until the queue is empty.

**Case header and case file**
- Above the thread: status · sources · claims · conflicts · version (e.g.
  "Needs review · 4 sources · 9 claims · 1 conflict · v3").
- The **Case file** has six tabs:
  - **Review:** the queue of claims waiting for you.
  - **Claims:** every claim, grouped by question.
  - **Conflicts:** the supporting and the contradicting passage, side by side.
  - **Plan:** every plan version.
  - **Evidence:** every lookup and what it returned.
  - **Trace:** for each run, the time, step, status, lookups, tokens, model, prompt version and
    source rules.

**Plans kept as part of the case**
- Each plan step comes with the agent's reason for it ("Why: …").
- Steps can be moved up or down, marked **Required** or **Optional**, edited, added or removed.
- Each step records who wrote it: *drafted by agent*, *edited by you* or *added by you*.
- Every approved plan is saved as a numbered version on the case and linked to the answer it
  produced.

**Source rules** (Settings)
- **Only use these domains** / **Never use these domains**, enforced in code on every search
  result. `gov` matches every .gov site, and lookalike domains such as `fakenasa.gov` are
  rejected. Wikipedia switches off automatically when the rules exclude it.
- **Recency:** past year, month or week, passed to the search engines.
- The active rules show in the sidebar and are recorded on every answer and run.

**Follow-ups ask before searching**
- When the gathered evidence can't answer a follow-up, the thread shows what's missing and
  offers **Request more evidence** (up to 2 searches) or **Answer from what we have**. A
  different topic offers **Research it**.
- Settings has **Search automatically on follow-ups** for the old behaviour.

**Audit export**
- **Export → Download audit report (.md)** contains:
  - the case status, model and prompt version
  - every answer with a claims table: claim, verdict, quoted evidence, source, review status
  - conflicts
  - appendices: plan versions, lookups and run trace
  - a "How to read this" note
- `PROMPT_VERSION` in `agent.py` is recorded in every run and export.

**Also**
- **Runs & quality** adds three metrics: *Claims supported*, *Claims checked* and *Prompt
  version*. Its table gains a Claims column.
- The Library shows claim and to-review counts for each case.
- The welcome screen says what's different: every claim is checked against its passage.
- **Fixed:** points that already had inline citations showed them twice (e.g. "¹³ ¹³").

**Tested** (live model calls and real searches):
- **New research:** 5 claims, all with word-for-word quotes.
- **Follow-up needing new evidence:** it asked first; "Request more evidence" made 1 search
  and produced 2 supported claims from nasa.gov.
- **Allowlist `nasa.gov`:** all 4 cited sources were nasa.gov.
- **Plan:** reorder and Optional kept, stored as v1 and linked to its answer.
- **Accept** cleared the review queue.
- All pages load with no errors.

---

## 9. Research threads and research-first answers — `735f187`

**Threads that keep their history**
- Every question and answer is its own message, added **below** the earlier ones.
  Earlier messages are never edited or replaced.
- One assistant bubble per answer, even when a follow-up has to search more.
- A different-topic question is researched in the same thread instead of replacing the page.
  Plan and clarify cards appear under the thread, not in place of it.
- The page scrolls to the newest message automatically.
- Only **New research** in the sidebar starts an empty thread.

**Real research, minimal citations**
- The agent searches before stating facts, uses more than one source for key claims,
  prefers authoritative sources, and says when sources disagree.
- Citations are small superscript links (¹ ²) inside the sentences, with one short source line
  under each answer (e.g. "¹ nasa.gov · ² wikipedia.org").
- The large badges, sources box and "Check sources" button were removed.
- Follow-ups are answered from the research already gathered. If that isn't enough, the agent
  searches automatically (at most 2 searches).
- "Related" questions are small chips. Clicking one asks it as a follow-up.

**Search fixes**
- Web search tries several engines in turn (Yahoo, DuckDuckGo, Mojeek, Brave, Google), each
  with a short timeout, so one slow engine no longer fails the whole lookup.
- Wikipedia results now include the article URL, so they can be cited.

**Simpler sidebar**
- The sidebar shows only: New research, Research, Library, Sources, Depth and Settings.
- Approvals appears only when a file is waiting.
- Research mode, usage, cost settings, model info, Runs & quality and Clear history moved
  into **Settings**.

---

## 8. Fix for stale code on Streamlit Cloud — `b9d0f37`, `9be54dc`

- After a deploy, Cloud could run new page code against old helper modules, which caused an
  `AttributeError` on `session_state.runs`.
- `app.py` now reloads `tools`, `agent` and `studio` whenever their files change.

## 7. Follow-ups stay in the same research session — `26e9797`

- Follow-up questions reuse the session's evidence (capped at 6,000 characters) instead of
  starting over, which saves tokens and time.
- Each answer shows how it was produced: answered from existing research, or researched further
  with N new searches.
- A search budget is enforced even when the model calls tools in parallel.
- Answers given without searching can't cite sources the session never saw.
- The "How it works" box was removed from the welcome screen.
- **Fix:** clicking Approve on a plan could make the plan vanish and loop. Slow work now runs
  as a "job" on its own rerun, and every control is locked until it finishes.

## 6. Dark theme — `f802a45`

- Dark palette set in `.streamlit/config.toml` (background `#101214`, accent `#4C8DF6`).
- The chat bar is centered on the welcome screen.

## 5. Sidebar redesign, modes and quality page — `38c38ff`

- Research modes: Ask first, Review plan, Auto-research.
- Source choice (Web, Wikipedia, your own documents) and depth (Quick, Standard, Deep).
- Token usage and estimated cost.
- New **Runs & quality** page, which tracks success rate, average time, cost, answer
  format, sources grounded and plan followed.

## 4. Ready for Streamlit Cloud — `62fe90c`

- Saved files are kept per session, so users don't see each other's files.
- `requirements.txt` trimmed and pinned.
- `.env.example` added; secrets stay out of git.
- The secret goes in Cloud's Secrets box as TOML with a quoted value:
  `GOOGLE_API_KEY = "..."`.

## 3. README — `29f3463`

- Setup, running locally, deploying, configuration and troubleshooting.

## 2. First version of the web app — `eeb415c`

- Streamlit UI over the LangChain + Gemini research agent from `main.py`.
- Single-column "Research Studio" layout with a library of past research and
  approval before any file is saved.
- **Fixes along the way:**
  - search timeouts (Startpage) that made every answer the same
  - Wikipedia crashes on empty replies
  - Windows Smart App Control blocking pandas, worked around by not using widgets that need it
  - occasional unparseable model replies, now handled by a lenient parse plus a repair pass

## 1. Starting point

- `main.py`: a command-line research agent using LangChain, Gemini, DuckDuckGo and Wikipedia.
  It still works as a CLI.

---

## Files at a glance

| File | What it does |
|---|---|
| `app.py` | Entry point: page setup, sidebar, navigation, module reload |
| `app_pages/research.py` | The chat thread and composer |
| `app_pages/library.py` | Past research sessions |
| `app_pages/approvals.py` | Approve or skip file saves |
| `app_pages/runs.py` | Runs & quality metrics |
| `studio.py` | Shared logic: state, research runs, follow-ups, rendering |
| `agent.py` | Intent router, `Answer` schema, prompts, agent builder, claim checker, reply parsing |
| `evidence.py` | Compresses lookups into evidence items; builds the compact follow-up context |
| `tools.py` | Web search, Wikipedia and document search tools |
| `main.py` | Original command-line version |

---

## Direction: from chat to research case

"Chat, search, cite, follow up" is not new: Perplexity and Deep Research already do it well.
Research Studio should not compete on chat polish, number of searches or report length.
Its difference is a **controlled research workflow whose output a person can audit, reuse and
act on**. The main object is a **research case** (plan, claims, evidence, decisions, versions,
approvals), and chat is one way to work on it.

### Where we are today

| Capability | Status | What exists now |
|---|---|---|
| Editable plan before research | ✅ Built | Reorder, edit, add or remove steps; Required/Optional; agent's reason per step; who changed what; versions kept on the case and linked to answers. |
| Claim-level evidence | ✅ Built | Each claim has a word-for-word quote, checked in code against the retrieved text. |
| Claim confidence labels | ✅ Built | Supported, Partially supported, Unsupported, Insufficient evidence, with a reason. |
| Conflict handling | ✅ Built | Contradicting passages kept per claim; Conflicts tab shows both sides. |
| Approval-gated actions | ✅ Built | File saves wait in the Approvals queue until approved or skipped. |
| Reusable evidence base | ✅ Built | Follow-ups reuse the case's evidence and ask before searching for more. |
| Quality scorecard | ✅ Built | Runs & quality: format, grounding, plan followed, claims supported, cost, latency. |
| Run trace | ✅ Built | Per run: timestamp, step, status, lookups, tokens, model, prompt version, source rules. |
| Source allowlist / denylist | ✅ Built | Enforced on every search result in code. |
| Recency controls | 🟡 Partial | Passed to search engines as a date filter. Wikipedia isn't date-filtered, and the app doesn't read publication dates itself. |
| Human review queue | ✅ Built | Accept / Reject / Undo on flagged claims; case shows Needs review until done. |
| Export with audit appendix | 🟡 Partial | Markdown audit report with claims, quotes, plans, lookups and trace. No PDF yet. |

### Planned next

Items 1–8 of the first roadmap are built (section 10). What's left:

1. **PDF export** of the audit report.
2. **Publication dates.** Read each source's date so recency is checked, not just requested.
3. **Edit a plan after it runs.** Re-run with a changed plan as v2 and compare the answers.
4. **"Blocked" plan steps**, and the agent explaining changes it makes to a plan mid-run.
5. **High-impact flag.** Let the user mark claims as material so they always go to review,
   even when supported.
6. **Persistence.** Cases live in the browser session today; saving them needs storage and sign-in.

### Focus

Start with **technical due diligence** (investors, analysts, engineers checking a technology or
vendor). It has a clear deliverable, measurable evidence quality, and a real reason for plan
approval and human review.

### What we won't claim

Not "100% accurate". The goal is: *every material claim is traceable to inspectable evidence,
and unsupported claims are explicitly flagged.*
