# Research Studio

A research agent that answers questions with real research: it searches the web, Wikipedia and your own documents, cross-checks what it finds, and answers in plain prose with small inline citations to the sources it actually read. Follow-up questions continue the same conversation, reusing what's already been found.

Built with [LangChain](https://www.langchain.com/), Google Gemini and [Streamlit](https://streamlit.io/).

## Features

- **Research first**: the agent searches before stating facts, uses more than one source for important claims, prefers authoritative sources, and says when sources disagree or evidence is thin.
- **Inline citations**: claims carry small superscript citations (¹ ²) linked to their sources, with one compact source line under each answer. Each cited source is checked against what the tools actually returned, and the app flags any that weren't.
- **Chat-style threads**: each follow-up gets its own answer below the earlier ones. Follow-ups are answered from the evidence the conversation already has (0 searches). When that isn't enough, the agent searches for what's missing (up to 2 searches). A question on a different topic offers to start a new research session.
- **Sources and depth** (sidebar): pick Web, Wikipedia and/or uploaded Documents (PDF, TXT, Markdown), and Quick, Standard or Deep. Depth sets how many lookups the agent makes and a step and time limit.
- **Live progress**: a one-line progress strip shows what the agent is doing (`✓ Web search · Reading Wikipedia...`). Expand it to see each lookup; every session keeps its research steps.
- **Optional planning** (Settings → Before researching): *Auto-research* by default. *Review plan* lets you edit and approve the research plan first; *Ask first* also asks up to two clarifying questions.
- **Reliable search**: several search engines are tried in turn, each with a short timeout, and repeat lookups are cached.
- **Lower token use**: follow-ups send a compact evidence pack (recent answers, numbered sources, trimmed snippets) instead of the whole conversation.
- **Approval before saving**: the agent can only *request* to save a file; an **Approvals** link appears when something is waiting.
- **Library**: every research session, plus files you've saved. Saves are private to each visitor's session; download anything you want to keep.
- **Runs & quality** (Settings): every run with timing, tokens, searches and automatic checks (valid format, sources grounded, plan followed, follow-ups answered without new searches), plus an optional cost estimate.
- **Prompt-injection guard**: text returned by tools is treated as reference material, never as instructions.
- **CLI mode**: `main.py` runs the same agent in the terminal and prints JSON.

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
Ask  →  Review plan  →  Watch research  →  Read and act on the report
```

1. **Plan.** In *Review plan* mode, Gemini drafts sub-questions as structured output (`make_plan` in `agent.py`).
2. **Research.** A LangChain tool-calling agent runs with these tools:
   - Web search: DuckDuckGo, Bing, Mojeek and Yahoo, tried in turn
   - Wikipedia
   - A save request that goes to the approval queue
3. **Stream.** A LangChain callback handler turns each tool call into a live progress update in the UI.
4. **Answer.** Every answer, whether code, explanation, research, comparison or fact check, is parsed into one Pydantic `Answer` envelope (`response_type`, `title`, `direct_answer`, `details`, `code`, `comparison`, `verdict`, `sources`, `follow_ups`). It renders as one standard card. If the model's output isn't valid JSON, the parser pulls the JSON out of the reply. If that fails, it asks the model once to convert its answer into the schema.

Tool failures, such as a search timeout, are returned to the agent as messages instead of exceptions, so it can fall back to another source.

## Project structure

```text
├── app.py                  # Entry point: page config, navigation, sidebar
├── app_pages/
│   ├── research.py         # Ask, review plan, watch progress, read report
│   ├── library.py          # Past reports and saved files
│   ├── approvals.py        # Pending file-save requests
│   └── runs.py             # Run history, usage and quality checks
├── studio.py               # Shared state, live progress handler, UI components
├── agent.py                # LLM, prompt, response schema, planner, parser
├── tools.py                # Web search, Wikipedia and save tools
├── main.py                 # Command-line interface
├── .streamlit/config.toml  # Light theme
├── .env.example            # Environment variable template
└── requirements.txt
```

## Configuration

| What | Where |
|---|---|
| Gemini model | `MODEL_NAME` in `agent.py` |
| Search engines and timeouts | `_ENGINES`, `_ENGINE_TIMEOUT` in `tools.py` |
| Depth presets (lookups, step and time limits) | `DEPTHS` in `studio.py` |
| Follow-up search budget and evidence pack size | `FURTHER_MAX_LOOKUPS`, `EVIDENCE_PACK_CHARS` in `studio.py` |
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

## Troubleshooting

**`streamlit` command is blocked or not found (Windows)**
Run it through Python instead: `python -m streamlit run app.py`.

**`DLL load failed ... An Application Control policy has blocked this file`**
Windows Smart App Control can block some compiled packages, such as pandas. The app avoids features that need pandas, so it runs with the policy on. If you add features that use pandas, you may need to adjust the policy.

**Search keeps failing**
Some networks block certain search engines. The app tries several in turn; reorder or trim the `_ENGINES` list in `tools.py` to put the ones that work for you first. Wikipedia still works when web search fails.

**"No `GOOGLE_API_KEY` found"**
Make sure `.env` is in the project root and restart the app.
