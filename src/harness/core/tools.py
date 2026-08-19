from pydantic import BaseModel, ConfigDict, Field

from harness.core.interfaces import SectionSearch, ToolSpec


class SectionSearchParams(BaseModel):
    """Search the corpus for sections matching a query."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Search terms, plain words.")
    top_k: int = Field(ge=1, le=10, description="How many sections to return.")


def build_section_search_tool(search: SectionSearch) -> ToolSpec[SectionSearchParams]:
    def handler(params: SectionSearchParams) -> str:
        hits = search.find(params.query, params.top_k)
        return "\n".join(f"{h.doc_id}#{h.section}: {h.text}" for h in hits)

    return ToolSpec(
        name="search_sections",
        params=SectionSearchParams,
        handler=handler,  # Closure captures `search`
    )
