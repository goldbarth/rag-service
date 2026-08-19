from harness.core.interfaces import SectionHit
from harness.core.tools import SectionSearchParams, build_section_search_tool


class FakeSearch:
    """Stands in for any SectionSearch. The tool must not care which one."""

    def __init__(self, hits: list[SectionHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def find(self, query: str, top_k: int) -> list[SectionHit]:
        self.calls.append((query, top_k))
        return self.hits


def test_handler_formats_hits() -> None:
    search = FakeSearch([SectionHit("d", "s", "text")])
    tool = build_section_search_tool(search)

    assert tool.handler(SectionSearchParams(query="x", top_k=1)) == "d#s: text"


def test_handler_joins_multiple_hits_with_newlines() -> None:
    search = FakeSearch(
        [SectionHit("a", "one", "first"), SectionHit("b", "two", "second")]
    )
    tool = build_section_search_tool(search)

    result = tool.handler(SectionSearchParams(query="x", top_k=2))

    assert result == "a#one: first\nb#two: second"


def test_handler_returns_empty_string_without_hits() -> None:
    tool = build_section_search_tool(FakeSearch([]))

    assert tool.handler(SectionSearchParams(query="nothing", top_k=3)) == ""


def test_handler_passes_params_through_to_the_search() -> None:
    search = FakeSearch([])
    tool = build_section_search_tool(search)

    tool.handler(SectionSearchParams(query="responses api", top_k=2))

    assert search.calls == [("responses api", 2)]


def test_tool_spec_names_the_tool_and_its_params() -> None:
    # No description here on purpose: the factory leaves it None and the
    # docstring fallback lives in the adapter's _to_tool_param, so that is
    # where it gets asserted.
    tool = build_section_search_tool(FakeSearch([]))

    assert tool.name == "search_sections"
    assert tool.params is SectionSearchParams
    assert tool.description is None
