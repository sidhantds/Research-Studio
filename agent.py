import json
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"


class Finding(BaseModel):
    text: str
    source_ids: list[int] = []  # 1-based positions in `sources`


class ResearchResponse(BaseModel):
    topic: str
    answer: str = ""  # one-sentence direct answer
    key_findings: list[Finding] = []
    summary: str  # detailed report, 2-4 paragraphs
    sources: list[str]
    tools_used: list[str]


@dataclass
class ToolStep:
    tool: str
    tool_input: str
    output: str


@dataclass
class AgentResult:
    response: ResearchResponse | None
    raw_text: str
    steps: list[ToolStep] = field(default_factory=list)
    error: str | None = None


parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that summarizes topics and provides sources and tools used for the summary.
            Answer the query and use necessary tools to provide a comprehensive summary.
            Fill the fields like this:
            - answer: one plain sentence that directly answers the question.
            - key_findings: 3 to 5 short findings. For each, list source_ids: the 1-based
              positions in `sources` that support it.
            - summary: a detailed report in 2 to 4 paragraphs separated by blank lines.
            - sources: full URLs wherever possible.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{query}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
).partial(format_instructions=parser.get_format_instructions())

TOOLS = [search_tool, wiki_tool, save_tool]


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=MODEL_NAME)


def build_agent_executor(llm=None, tools=None) -> AgentExecutor:
    llm = llm or get_llm()
    tools = tools or TOOLS
    agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True,
    )


# ---------- Research planning ----------

PLAN_TOOLS = ["Web", "Wikipedia", "Both"]
_TOOL_INSTRUCTION = {
    "Web": "web search",
    "Wikipedia": "wikipedia",
    "Both": "both web search and wikipedia",
}


class PlanStep(BaseModel):
    question: str
    tool: str  # one of PLAN_TOOLS
    expected: str = ""  # what this step should find, in a few words


class ResearchPlan(BaseModel):
    steps: list[PlanStep]


def make_plan(llm, query: str) -> list[dict]:
    """Ask the LLM to break a request into 2-4 sub-questions, each with a source."""
    planner = llm.with_structured_output(ResearchPlan)
    try:
        plan = planner.invoke(
            [
                (
                    "system",
                    "Break the user's research request into 2 to 4 short, focused sub-questions. "
                    f"For each, pick a source from {PLAN_TOOLS}: 'Web' for recent or specific facts, "
                    "'Wikipedia' for background and definitions, 'Both' when it needs both. "
                    "In 'expected', say in under 8 words what the step should find.",
                ),
                ("human", query),
            ]
        )
        steps = [
            {
                "question": s.question.strip(),
                "tool": s.tool if s.tool in PLAN_TOOLS else "Web",
                "expected": s.expected.strip(),
            }
            for s in (plan.steps if plan else [])
            if s.question.strip()
        ]
    except Exception:
        steps = []
    return steps or [{"question": query, "tool": "Web", "expected": ""}]


def plan_to_instructions(steps: list[dict] | None) -> str:
    if not steps:
        return ""
    lines = "\n".join(
        f"{i}. {s['question']} (use {_TOOL_INSTRUCTION.get(s['tool'], 'web search')})"
        for i, s in enumerate(steps, 1)
    )
    return f"\n\nFollow this research plan, approved by the user:\n{lines}"


def _extract_text(raw_output) -> str:
    """Gemini may return a plain string or a list of content blocks."""
    if isinstance(raw_output, str):
        return raw_output
    if isinstance(raw_output, list):
        text_blocks = [
            block["text"]
            for block in raw_output
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if not text_blocks:
            raise ValueError("Agent response did not contain any text blocks.")
        return "\n".join(text_blocks)
    raise TypeError(
        f"Expected agent output to be text or text blocks, got {type(raw_output).__name__}."
    )


def to_chat_history(turns: list[tuple[str, str]]):
    """Convert (role, text) pairs into LangChain messages."""
    return [
        HumanMessage(content=text) if role == "user" else AIMessage(content=text)
        for role, text in turns
    ]


def run_research(
    executor: AgentExecutor,
    query: str,
    chat_history: list[tuple[str, str]] | None = None,
    callbacks: list | None = None,
) -> AgentResult:
    payload = {"query": query}
    if chat_history:
        payload["chat_history"] = to_chat_history(chat_history)

    raw_response = executor.invoke(payload, config={"callbacks": callbacks or []})

    steps = [
        ToolStep(
            tool=getattr(action, "tool", "tool"),
            tool_input=str(getattr(action, "tool_input", "")),
            output=str(observation),
        )
        for action, observation in raw_response.get("intermediate_steps", [])
    ]

    raw_output = raw_response.get("output", "")
    try:
        output_text = _extract_text(raw_output)
    except (TypeError, ValueError) as e:
        return AgentResult(None, str(raw_output), steps, error=str(e))

    try:
        return AgentResult(parse_response(output_text), output_text, steps)
    except Exception as e:  # model didn't follow the JSON format
        return AgentResult(None, output_text, steps, error=str(e))


def parse_response(text: str) -> ResearchResponse:
    """Strict parse, then lenient JSON extraction, then one LLM repair pass."""
    try:
        return parser.parse(text)
    except Exception as strict_error:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                data = json.loads(match.group(0))
                # Some replies wrap the answer in the schema itself: {"properties": {...}}
                if isinstance(data.get("properties"), dict) and "topic" in data["properties"]:
                    data = data["properties"]
                return ResearchResponse.model_validate(data)
            except Exception:
                pass
        if not text.strip():
            raise strict_error
        repaired = get_llm().with_structured_output(ResearchResponse).invoke(
            "Convert this research answer into the required structure. Keep every fact and "
            f"source; don't add new ones.\n\n{text}"
        )
        if repaired is None:
            raise strict_error
        return repaired