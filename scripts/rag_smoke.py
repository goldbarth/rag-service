"""Manual smoke check for the tool-augmented RAG path. Needs a valid .env.

This is the only check that can say whether SYSTEM_PROMPT works. Every test in
tests/test_rag.py fakes the ToolCompleter, so they prove that a prompt is
passed on, never that the model obeys it. A model that answers from its weights
instead of calling the tool passes all of them.

Two separate questions per case, and the run only answers the first by itself:
  called    - did the model use the tool at all (rounds carry the names)
  grounded  - does the answer rest on what came back, with a doc_id#section

The second one needs a human reading the output. That is why the cases are
built so that the corpus and the model's own knowledge disagree: an ungrounded
answer is then visibly wrong rather than merely unsourced.

Usage:
    uv run python scripts/rag_smoke.py

Spends tokens: one run per case, each up to max_rounds provider calls.
"""

from typing import Any

from harness.api.dependencies import get_llm_client, get_llm_config
from harness.api.routers.rag import SYSTEM_PROMPT
from harness.core.interfaces import SectionHit, ToolSpec
from harness.core.tools import build_section_search_tool
from harness.infrastructure.retrieval.stub import StubCorpus

# Deliberately not the stub corpus from infrastructure: these sections state
# something the model cannot know and would not guess. If an answer contains
# the invented facts, the tool result reached it; if it contains the real
# world, the prompt failed and the model answered from memory.
_SECTIONS = (
    SectionHit(
        doc_id="handbook",
        section="release-cadence",
        text=(
            "The Bellweather harness ships a release every 43 days, "
            "a cadence chosen so a sweep always spans two releases."
        ),
    ),
    SectionHit(
        doc_id="handbook",
        section="judge-policy",
        text=(
            "A Bellweather verdict of unclear is reported separately and "
            "never counted as a miss."
        ),
    ),
    SectionHit(
        doc_id="handbook",
        section="storage",
        text="Bellweather run artifacts are kept for exactly 19 months.",
    ),
)

CASES = {
    "in-corpus": "How often does the Bellweather harness ship a release?",
    "in-corpus-detail": "How long are Bellweather run artifacts kept?",
    # The search still returns all three sections here, because StubCorpus
    # scores on substrings and "bellweather" sits in every one of them. None of
    # them answers the question, so this case asks whether the model stays with
    # what it got instead of filling the gap from its own weights.
    "hits-but-no-answer": "Which release does the Bellweather judge policy ship?",
    # No section scores at all here, so the handler returns an empty string.
    # That is the one input the prompt has an explicit rule for, and nothing
    # else in this file reaches it.
    "no-hits": "Which database engine powers deployment?",
}


if __name__ == "__main__":
    llm = get_llm_client()
    config = get_llm_config()
    tools: list[ToolSpec[Any]] = [build_section_search_tool(StubCorpus(_SECTIONS))]

    print(f"model: {config.model_name}")
    print(f"prompt:\n{SYSTEM_PROMPT}\n")

    for label, question in CASES.items():
        result = llm.complete_with_tools(
            system_prompt=SYSTEM_PROMPT,
            user_message=question,
            config=config,
            tools=tools,
        )

        called = [name for round_ in result.rounds for name in round_.tool_names]
        # "called" is decidable from the rounds, so the run says it outright
        # instead of leaving it in a list the reader has to notice is empty. A
        # model that stopped calling the tool then reads as FAILED, not as one
        # more line of output. "grounded" stays for the human below.
        verdict = "ok" if called else "FAILED, answered without the tool"

        print(f"{label}: {question}")
        print(f"  called: {verdict}")
        print(f"  stop_reason: {result.stop_reason}")
        print(f"  rounds: {len(result.rounds)}  tool calls: {called or 'none'}")
        print(f"  answer: {result.text}")
        for index, round_ in enumerate(result.rounds):
            print(f"  usage[{index}]: {round_.usage}")
        print()
