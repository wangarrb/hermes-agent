import sys


def test_chat_subcommand_accepts_named_custom_provider(monkeypatch):
    import hermes_cli.main as main_mod

    captured = {}

    def fake_cmd_chat(args):
        captured["provider"] = args.provider
        captured["query"] = args.query

    monkeypatch.setattr(main_mod, "cmd_chat", fake_cmd_chat)
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "chat", "--provider", "cch", "-m", "gpt-5.4", "-q", "hello"],
    )

    main_mod.main()

    assert captured == {
        "provider": "cch",
        "query": "hello",
    }
