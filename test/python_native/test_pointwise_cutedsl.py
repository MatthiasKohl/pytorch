# Owner(s): ["module: dsl-native-ops"]

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestPointwiseIndexAdmission(TestCase):
    @parametrize("case", ("small", "count", "rounded", "offset"))
    def test_32bit_admission(self, case: str) -> None:
        from torch._native.ops.pointwise.indexing import can_use_32bit_index_math

        size = 2**31 if case == "count" else 2**31 - 1 if case == "rounded" else 32
        tensor = (
            torch.empty_strided((2, 2), (2**31, 1), device="meta")
            if case == "offset"
            else torch.empty(size, device="meta")
        )
        self.assertEqual(
            can_use_32bit_index_math((tensor,), tensor.shape), case == "small"
        )


instantiate_parametrized_tests(TestPointwiseIndexAdmission)

if __name__ == "__main__":
    run_tests()
