import pytest
from scarlet.testing import api


@pytest.mark.usefixtures("branch")
class TestRegressions(object):
    def test_set1(self, branch):
        api.deblend_and_measure(1, branch=branch, save_records=True)

    def test_set2(self, branch):
        api.deblend_and_measure(2, branch=branch, save_records=True)
        pass

    def test_set3(self, branch):
        api.deblend_and_measure(3, branch=branch, save_records=True, save_residuals=True)
        pass
