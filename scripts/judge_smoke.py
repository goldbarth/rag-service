"""Manual smoke check for the LLM judge. Needs a valid .env."""

from harness.api.dependencies import get_llm_client, get_llm_config
from harness.core.evaluation import judge_answer

QUESTION = "What is the capital of France?"
EXPECTED = "Paris is the capital of France."

CASES = {
    "match": "The capital of France is Paris.",
    "contradiction": "Lyon is the capital of France.",
}

if __name__ == "__main__":
    llm = get_llm_client()
    # temperature unset on purpose, see LlmConfig.
    config = get_llm_config()

    for label, actual in CASES.items():
        result = judge_answer(QUESTION, EXPECTED, actual, llm, config)
        print(f"{label}: {result.verdict.verdict} - {result.verdict.reasoning}")
        print(f"  usage: {result.usage}")
