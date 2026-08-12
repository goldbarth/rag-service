"""Manual smoke check against the real OpenAI API. Needs a valid .env."""

from rag_service.api.dependencies import get_llm_client

if __name__ == "__main__":
    answer = get_llm_client().complete(
        system_prompt="You're a helpful assistant. Keep your answers brief.",
        user_message="Tell me in one sentence what a retrieval-regression harness is.",
    )
    print(answer)
