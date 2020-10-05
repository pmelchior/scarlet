from scarlet.testing import api


class TestRegressions(object):
    def test_set1(self):
        api.deblend_and_measure(1, save_records=True)

    def test_set2(self):
        api.deblend_and_measure(2, save_records=True)

    def test_set3(self):
        api.deblend_and_measure(3, save_records=True, save_residuals=True)
