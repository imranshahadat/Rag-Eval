"""Unit tests for ContactExtractor (regex paths only, no LLM needed)."""
from memory.contact_extractor import ContactExtractor


def make_extractor() -> ContactExtractor:
    return ContactExtractor(llm=None)


def test_extract_email():
    contact = make_extractor().extract("You can reach me at John.Doe@Example.COM anytime")
    assert contact is not None
    assert contact.email == "john.doe@example.com"


def test_extract_phone():
    contact = make_extractor().extract("Call me at +92 300 1234567")
    assert contact is not None
    assert contact.phone == "+923001234567"


def test_extract_strong_name():
    contact = make_extractor().extract("Hello, my name is john doe")
    assert contact is not None
    assert contact.name == "John Doe"


def test_no_contact_returns_none():
    assert make_extractor().extract("What projects have you worked on?") is None


def test_weak_name_without_llm_returns_none():
    # "i am" without a strong pattern and no LLM fallback must not crash or invent a name
    contact = make_extractor().extract("i am looking for a developer")
    assert contact is None
