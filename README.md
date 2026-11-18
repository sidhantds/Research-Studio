# Research Studio

A research agent that turns a question into a sourced report. It drafts a research plan for you to review, searches the web and Wikipedia while you watch, and returns a report with key findings, inline citations and the raw evidence behind them.

Built with [LangChain](https://www.langchain.com/), Google Gemini and [Streamlit](https://streamlit.io/).

## Features

- **Three research modes** (sidebar):
  - **Ask first**: the agent asks up to two clarifying questions if your request is ambiguous, then proposes a plan.
  - **Review plan**: the agent breaks your question into sub-questions, each with its sources, for you to edit and approve.
  - **Auto-research**: plans and runs without stopping.
- **Choose your sources**: web search, Wikipedia, and your own uploaded documents (PDF, TXT, Markdown).
- **Depth**: Quick, Standard or Deep. Each sets how many lookups the agent makes, how long the report is, and a step and time limit for the run.
- **Live progress**: a one-line progress strip shows what the agent is doing (`✓ Web search · Reading Wikipedia...`). Expand it to see each tool call.
- **Structured reports** with:
  - a one-sentence answer
  - key findings with `[1]`-style citations linked to their sources
  - a detailed report and a numbered source list
  - the raw tool output, collapsed by default
- **Act on findings**: each finding has **Explore**, **Check sources** and **Ask follow-up** buttons.
- **Approval before saving**: the agent can only *request* to save a file. Nothing is saved until you approve it on the **Approvals** page.
- **Library**: every report from the session, plus files you've saved. Saves are private to each visitor's session; download anything you want to keep.
- **Runs & quality**: every run with timing, tokens and three automatic checks:
  - valid report format
  - sources grounded (each cited source appears in what the tools returned)
  - plan followed
- **Usage tracking**: token counts from every model call, and a cost estimate once you enter prices under **Settings**.
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
4. **Report.** The agent's answer is parsed into a Pydantic `ResearchResponse` (topic, answer, key findings with source IDs, summary, sources). If the model's output isn't valid JSON, the parser pulls the JSON out of the reply. If that fails, it asks the model once to convert its answer into the schema.

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
| Search engines | `backend=` in `tools.py` |
| Depth presets (lookups, step and time limits) | `DEPTHS` in `studio.py` |
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
Some networks block certain search engines. Edit the `backend=` list in `tools.py`. The agent still answers from Wikipedia when web search fails.

**"No `GOOGLE_API_KEY` found"**
Make sure `.env` is in the project root and restart the app.
