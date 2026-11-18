"""Command-line entry point. For the web UI run: streamlit run app.py"""

from agent import build_agent_executor, run_research

if __name__ == "__main__":
    query = input("How can I help you today?\n> ").strip()
    if not query:
        raise ValueError("Please enter a query.")

    result = run_research(build_agent_executor(), query)

    if result.response:
        print(result.response.model_dump_json(indent=2))
    else:
        print(f"Error parsing response: {result.error}")
        print(f"Raw response: {result.raw_text}")
