from app.chat import _sanitize_reply, build_context


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


def test_context_includes_nearby_best_match():
    ctx = build_context(
        {"full_name": "Nagarjun"},
        [{"created_at": "2026-09-26", "verdict": "Caries Detected", "count": 2, "model": "caries_photo"}],
        {
            "include_emergency": True,
            "places": [
                {
                    "name": "City Dental",
                    "kind": "dentist",
                    "distance_km": 1.2,
                    "rating": 4.6,
                    "reviews": 80,
                    "why": "General dentist — diagnoses and treats cavities",
                }
            ],
        },
    )
    assert "NEARBY CARE" in ctx
    assert "Best match: City Dental" in ctx
    assert "1.2 km" in ctx
