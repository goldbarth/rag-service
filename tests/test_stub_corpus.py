import pytest
from pydantic import ValidationError

from harness.core.interfaces import SectionHit
from harness.core.tools import SectionSearchParams
from harness.infrastructure.retrieval.stub import StubCorpus, build_stub_corpus

_SECTIONS = (
    SectionHit(doc_id="alpha", section="intro", text="apples and pears"),
    SectionHit(doc_id="beta", section="body", text="apples only"),
    SectionHit(doc_id="gamma", section="body", text="nothing relevant"),
)


def test_find_returns_only_matching_sections() -> None:
    hits = StubCorpus(_SECTIONS).find("apples", top_k=10)

    assert [hit.doc_id for hit in hits] == ["alpha", "beta"]


def test_find_ranks_more_term_matches_first() -> None:
    hits = StubCorpus(_SECTIONS).find("apples pears", top_k=10)

    assert [hit.doc_id for hit in hits] == ["alpha", "beta"]


def test_find_matches_doc_id_and_section_too() -> None:
    hits = StubCorpus(_SECTIONS).find("gamma", top_k=10)

    assert [hit.doc_id for hit in hits] == ["gamma"]


def test_find_respects_top_k() -> None:
    hits = StubCorpus(_SECTIONS).find("apples", top_k=1)

    assert len(hits) == 1


def test_find_returns_nothing_without_a_match() -> None:
    assert StubCorpus(_SECTIONS).find("bananas", top_k=10) == []


def test_build_stub_corpus_is_searchable() -> None:
    hits = build_stub_corpus().find("responses", top_k=5)

    assert [hit.doc_id for hit in hits] == ["openai"]


def test_build_stub_corpus_returns_a_fresh_instance_per_call() -> None:
    # Not an isolation guarantee for the app: get_section_search caches, so
    # every request shares one corpus. This only pins that the caching decision
    # sits in the wiring, so a test can build its own without clearing a cache.
    assert build_stub_corpus() is not build_stub_corpus()


@pytest.mark.parametrize("top_k", [0, 11])
def test_top_k_outside_the_declared_range_is_rejected(top_k: int) -> None:
    # The bounds are part of the tool schema the model sees, so they must hold
    # on our side too.
    with pytest.raises(ValidationError):
        SectionSearchParams(query="apples", top_k=top_k)
