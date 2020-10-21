import pytest

def pytest_addoption(parser):
    print(dir(parser))
    parser.addoption(
        "--branch", action="store", default=None, help="branch name",
    )


@pytest.fixture
def branch(request):
    return request.config.getoption("--branch")
