from collections.abc import Sequence

from harness.core.interfaces import SectionHit


class StubCorpus:
    def __init__(self, sections: Sequence[SectionHit]) -> None:
        self._sections = list(sections)

    def find(self, query: str, top_k: int) -> list[SectionHit]:
        # Scores on substrings, not on words: a term of two or three letters
        # hits almost every section, and a term carrying punctuation ("api?")
        # hits none. Good enough to exercise the tool loop, and worth knowing
        # when writing a query that is supposed to return nothing. Real
        # retrieval replaces this in phase 3.
        terms = query.lower().split()

        scored: list[tuple[int, SectionHit]] = []
        for section in self._sections:
            haystack = f"{section.doc_id} {section.section} {section.text}".lower()
            score = sum(1 for term in terms if term in haystack)
            if score > 0:
                scored.append((score, section))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [section for _, section in scored[:top_k]]


_STUB_SECTIONS: tuple[SectionHit, ...] = (
    SectionHit(
        doc_id="example-doc",
        section="intro",
        text="This is a stub section used for local tool testing.",
    ),
    SectionHit(
        doc_id="openai",
        section="responses-api",
        text="The OpenAI Responses API can create completions and tool calls.",
    ),
)


def build_stub_corpus() -> StubCorpus:
    """Return a corpus over the built-in sample sections."""
    return StubCorpus(_STUB_SECTIONS)
