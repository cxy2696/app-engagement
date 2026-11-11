from engagegraph.cli import main


def test_version_command(capsys):
    assert main(["version"]) == 0
    assert "engagegraph" in capsys.readouterr().out


def test_parser_requires_command(capsys):
    try:
        main([])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("expected argparse to exit")
