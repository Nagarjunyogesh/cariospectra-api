from app.chat import _sanitize_reply


def test_strips_hashes_and_rules():
    raw = (
        "Hi Nagarjun, here is a summary.\n"
        "\n"
        "### What the findings mean\n"
        "--\n"
        "- Caries (tooth decay) — a breakdown of enamel.\n"
        "---\n"
        "## What to do next\n"
        "- Book a dental appointment.\n"
    )
    out = _sanitize_reply(raw)
    assert "###" not in out
    assert "##" not in out
    assert "--" not in out
    assert "What the findings mean" in out
    assert "Book a dental appointment." in out
