# Research Studio

**Research Studio turns a question into a reviewable research case:** each claim is checked against the passage it came from, unsupported claims are flagged for review, plans are editable and versioned, any file save waits for your approval, and the whole case exports as an audit report.

ChatGPT and Perplexity answer questions. Research Studio keeps a research case you can audit: the plan, the claims, the evidence behind each one, the conflicts, your review decisions and a trace of every run. You work on the case through chat.

Built with [LangChain](https://www.langchain.com/), Google Gemini and [Streamlit](https://streamlit.io/).

## Features

### Evidence you can check

- **Claim-level verification.** After each answer, a separate model call splits it into claims (up to 8). It quotes the passage behind each claim and gives a verdict: *Supported*, *Partially supported*, *Unsupported* or *Insufficient evidence*. The app then checks every quote **word for word** against the text the search tools actually returned. A claim whose quote isn't there is downgraded, so the model never grades its own work unchecked.
- **Conflicts.** When one source contradicts a claim, both passages are shown side by side.
- **Review queue.** Claims that aren't fully supported, or that conflict, wait for you to **Accept** or **Reject**. The case shows *Needs review* until the queue is clear.
- **Evidence on every answer.** Each cited source appears under the answer with the passage it was cited for. A quote is shown only if it appears word for word in the retrieved text, and sources the session never retrieved are dropped.
- **Source rules** (Settings). *Only use these domains*, *Never use these domains* and *Recency* (past year, month or week). The domain rules are enforced in code on every search result, not just suggested to the model.

### Answers shaped to the question

- **Intent routing.** Each request is classified before anything runs: code, explanation, research, comparison, fact check or clarification. Keyword rules handle the clear cases with no model call; only ambiguous requests use one fast structured call.
- **No research when none is needed.** "Generate hello world in Java" is answered directly: one call, no searches, no sources.
- **One answer card for every type.** A header with the type, Copy and Save; then the direct answer:
  - code, with filename, run commands and expected output
  - a verdict, for fact checks
  - a comparison table, for comparisons

  Then details (research, comparisons and fact checks only), evidence, a status line and next-step suggestions.

### Follow-ups that don't waste tokens

- **Follow-up mode**, above the chat bar:
  - **Use current research** (default) answers from what the session already found. It never searches. If new facts are needed, it says so and offers to research further.
  - **Research further** adds up to 2 new searches in the same session.
  - **New research** starts a separate research case.

  After any search, the mode returns to *Use current research*, so searching is always a deliberate choice.
- **One session per case.** Follow-ups keep the same session id (`thread_id`) and bump its version. Each answer is added below the earlier ones, and only new sources are added. Every answer says how it was produced, e.g. *Answered from existing research · 0 new searches*.
- **Structured evidence.** Each lookup is compressed into small items: the key sentence, a word-for-word quote, URL, retrieval time and a relevance score. A follow-up gets the original question, the current answer (and code), the 8 most relevant items and the question itself. Raw pages and the full chat history are never re-sent.

### The research case

- **Case header:** status · sources · claims · conflicts · version.
- **Case file**, in six tabs:
  - **Review:** claims waiting for you to accept or reject
  - **Claims:** every claim, with its verdict
  - **Conflicts:** the supporting and contradicting passages side by side
  - **Plan:** every plan version
  - **Evidence:** each evidence item, plus the raw lookups
  - **Trace:** for each run, the time, step, lookups, context size, tokens, model, prompt version and source rules
- **Plans** (Settings → Before researching → *Review plan*). Each step comes with the agent's reason for it. You can reorder, edit, add or remove steps, and mark them *Required* or *Optional*. Each step records whether the agent drafted it or you edited or added it. Approved plans are kept as numbered versions, each linked to the answer it produced. *Ask first* also asks up to two clarifying questions.
- **Audit export.** A Markdown report with:
  - case status, model and prompt version
  - every answer with a claims table (claim, verdict, quoted evidence, source, review)
  - conflicts
  - appendices: plan versions, lookups and run trace

### Also

- **Sources and depth** (sidebar): Web, Wikipedia and uploaded Documents (PDF, TXT, Markdown); Quick, Standard or Deep.
- **Live progress:** one line shows what the agent is doing (`✓ Web search · Reading Wikipedia...`); expand it to see each lookup.
- **Approval before saving:** the agent can only *request* to save a file; nothing is written until you approve it.
- **Library:** every research case, with its claim and review counts, plus files you've saved. Saves stay private to each visitor's browser session.
- **Runs & quality:** every run with timing, tokens and automatic checks: valid format, sources grounded, plan followed, claims supported, and follow-ups answered without new searches. Optional cost estimate.
- **Reliable search:** several search engines are tried in turn, each with a short timeout; repeat lookups are cached.
- **Prompt-injection guard:** text returned by tools is treated as reference material, never as instructions.
- **CLI mode:** `main.py` runs the agent in the terminal and prints the answer as JSON.

## Getting started

### Prerequisites

- Python 3.11+
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

### Install

```bash
git clone https://github.com/sidhantds/Research-Studio.git
cd Research-Studio

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Configure

Copy the example environment file and add your key:

```bash
cp .env.example .env
```

```env
GOOGLE_API_KEY=your-google-api-key-here
```

### Run

```bash
python -m streamlit run app.py
```

Then open http://localhost:8501.

To use the terminal version instead:

```bash
python main.py
```

## How it works

```text
Question → Route → Answer directly, or research → Check claims → Case
                                                                   ↓
Follow-up → Route → Use current research | Research further (≤2) | New research
```

1. **Route.** `route_request` in `agent.py` returns a `Route` with `response_type`, `context_mode`, `needs_tools` and a reason. Keyword rules come first; an LLM call is the fallback.
2. **Plan (optional).** In *Review plan* mode, Gemini drafts sub-questions with a reason for each, as structured output (`make_plan`).
3. **Research.** A LangChain tool-calling agent runs with:
   - Web search (Yahoo, DuckDuckGo, Mojeek, Brave and Google, tried in turn), filtered by your source rules
   - Wikipedia
   - Your uploaded documents
   - A save request that goes to the approval queue

   Each run carries a `RunContext` (thread id, response type, depth, sources, rules, search budget) as LangChain run metadata.
4. **Answer.** Every answer is parsed into one Pydantic `Answer` envelope: `response_type`, `title`, `direct_answer`, `details`, `code`, `comparison`, `verdict`, `sources`, `follow_ups`, `context_mode`, `new_searches`. If the output isn't valid JSON, the parser extracts the JSON from the reply. If that fails, it asks the model once to convert its answer.
5. **Verify.** `check_claims` extracts claims and quotes evidence; the app confirms each quote in code and downgrades any it can't find.
6. **Store.** Lookups are kept raw (for the audit trail and quote checks) and also compressed into evidence items by `evidence.py`. Follow-ups use the compressed items.

Tool failures, such as a search timeout, come back to the agent as messages instead of exceptions, so it can try another source.

## Project structure

```text
├── app.py                  # Entry point: page config, navigation, sidebar, settings
├── app_pages/
│   ├── research.py         # The research session: chat, follow-up modes, jobs
│   ├── library.py          # Research cases and saved files
│   ├── approvals.py        # Pending file-save requests
│   └── runs.py             # Run history, usage and quality checks
├── studio.py               # Sessions, research runs, claim checks, answer card, case file, export
├── agent.py                # Intent router, Answer schema, prompts, agent, planner, claim checker, parser
├── evidence.py             # Evidence items and the compact follow-up context
├── tools.py                # Web search, Wikipedia, documents, source rules, save tool
├── main.py                 # Command-line interface
├── CHANGELOG.md            # What changed, and what's planned
├── .streamlit/config.toml  # Dark theme
├── .env.example            # Environment variable template
└── requirements.txt
```

## Configuration

| What | Where |
|---|---|
| Gemini model | `MODEL_NAME` in `agent.py` |
| Prompt version (recorded in traces and exports) | `PROMPT_VERSION` in `agent.py` |
| Routing keyword rules | `_RULES`, `_CODE_VERB`, `_CODE_NOUN`, `_FRESH` in `agent.py` |
| Search engines and timeouts | `_ENGINES`, `_ENGINE_TIMEOUT` in `tools.py` |
| Depth presets (lookups, step and time limits) | `DEPTHS` in `studio.py` |
| "Research further" search budget | `FURTHER_MAX_LOOKUPS` in `studio.py` |
| Evidence items sent with a follow-up | `PACK_ITEMS` in `evidence.py` |
| Source rules and recency | Settings in the app |
| Default token prices for cost estimates | `PRICE_INPUT_PER_M` and `PRICE_OUTPUT_PER_M` environment variables or secrets |
| Theme colors and font | `.streamlit/config.toml` |

## Deploy to Streamlit Community Cloud

1. Push the repo to GitHub, then go to [share.streamlit.io](https://share.streamlit.io) and click **Create app**.
2. Set **Main file path** to `app.py`.
3. Under **Advanced settings**, choose Python **3.11** and add your key under **Secrets**:

   ```toml
   GOOGLE_API_KEY = "your-google-api-key"
   ```

   Streamlit Cloud also exposes top-level secrets as environment variables, so no code changes are needed.
4. Click **Deploy**.

Every visitor's research runs on your API key. To control who can use it, make the app private under **Settings → Sharing** and invite people by email.

## Limitations

- Research cases live in the browser session; they're gone when the session ends. Export the audit report to keep one.
- Recency is passed to the search engines as a date filter. The app doesn't read publication dates itself, and Wikipedia isn't date-filtered.
- The audit export is Markdown only (no PDF yet).
- Claim checks add one model call per answer (code answers skip it).
- Verification makes each claim traceable to inspectable evidence. It doesn't guarantee the claim is true.

## Troubleshooting

**`streamlit` command is blocked or not found (Windows)**
Run it through Python instead: `python -m streamlit run app.py`.

**`DLL load failed ... An Application Control policy has blocked this file`**
Windows Smart App Control can block some compiled packages, such as pandas. The app avoids features that need pandas, so it runs with the policy on. If you add features that use pandas, you may need to adjust the policy.

**Search keeps failing**
Some networks block certain search engines. The app tries several in turn; reorder or trim the `_ENGINES` list in `tools.py` to put the ones that work for you first. Wikipedia still works when web search fails.

**Every search returns "No results from allowed sources"**
Your *Only use these domains* rule may be too narrow. Clear it in Settings, or add more domains.

**"No `GOOGLE_API_KEY` found"**
Make sure `.env` is in the project root and restart the app.
