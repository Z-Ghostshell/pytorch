# Fixture suite for TestPruneSkipsAtCollection in test/test_testing.py.
import unittest

import pytest

from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import lazy_skip_if, run_tests, TestCase


class TestPruneFixture(TestCase):
    def test_runs(self):
        pass

    @unittest.skip("static mark")
    def test_static_mark(self):
        pass

    @lazy_skip_if(lambda: True, "runtime condition")
    def test_lazy_skip(self):
        pass

    def test_body_skip(self):
        self.skipTest("decided in the body")


class TestSetupDependentSkip(TestCase):
    ready = False

    def setUp(self):
        super().setUp()
        type(self).ready = True

    @lazy_skip_if(lambda: not TestSetupDependentSkip.ready, "initialized by setUp")
    def test_runs_after_setup(self):
        self.assertTrue(self.ready)


class TestOverriddenSetup(TestCase):
    def setUp(self):
        # This class does not use TestCase.setUp's slow/fast policy.
        self.ready = True

    def test_runs(self):
        self.assertTrue(self.ready)


class TestPruneFixtureDevice(TestCase):
    @onlyCUDA
    def test_cuda_only(self, device):
        pass

    def test_any_device(self, device):
        pass


instantiate_device_type_tests(TestPruneFixtureDevice, globals(), only_for="cpu")


@pytest.mark.skip(reason="pytest mark")
def test_pytest_mark():
    pass


if __name__ == "__main__":
    run_tests()
