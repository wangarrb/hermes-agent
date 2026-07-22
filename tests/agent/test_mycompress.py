import pytest

from agent.mycompress import load_mycompress_focus, parse_mycompress_args


def test_parse_mycompress_keep_rounds_and_focus():
    assert parse_mycompress_args("-n 3 retain decisions") == (3, "retain decisions")
    assert parse_mycompress_args("retain decisions --keep-last-rounds 2") == (
        2,
        "retain decisions",
    )
    assert parse_mycompress_args("only this topic") == (None, "only this topic")


@pytest.mark.parametrize("raw", ["-n 0", "--keep-last-rounds 0", "-n 2 -n 3"])
def test_parse_mycompress_rejects_invalid_keep_rounds(raw):
    with pytest.raises(ValueError):
        parse_mycompress_args(raw)


def test_load_mycompress_focus_uses_skill_body_not_frontmatter(monkeypatch):
    import agent.skill_commands

    monkeypatch.setattr(
        agent.skill_commands,
        "_load_skill_payload",
        lambda *_args, **_kwargs: (
            {"content": "---\nname: mycompress\nno_slash: true\n---\n\nKeep decisions."},
            None,
            "mycompress",
        ),
    )

    assert load_mycompress_focus("database schema", runtime_note="cli") == (
        "Keep decisions.\n\nUser focus: database schema\n\n[Runtime: cli]"
    )
