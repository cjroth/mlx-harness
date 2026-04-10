def pytest_addoption(parser):
    parser.addoption(
        "--e2e",
        action="store_true",
        default=False,
        help="Run end-to-end tests (requires model and Docker)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--e2e"):
        skip = __import__("pytest").mark.skip(reason="need --e2e option to run")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip)
