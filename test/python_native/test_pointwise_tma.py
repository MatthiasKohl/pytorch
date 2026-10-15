# Owner(s): ["module: dsl-native-ops"]

from unittest.mock import patch

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


class TestPointwiseTma(TestCase):
    @onlyCUDA
    @skipIfNoCuteDSL
    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("layout", ("mixed", "transposed", "rank3"))
    def test_tma_pointwise(self, device: str, dtype: torch.dtype, layout: str) -> None:
        from torch._native.ops.pointwise import kernel

        if torch.cuda.get_device_capability(device) != (10, 7):
            self.skipTest("Rubin pointwise tuning")
        if layout == "rank3":
            x = torch.randn(32, 32, 64, device=device, dtype=dtype).permute(2, 0, 1)
            y = torch.randn(x.shape, device=device, dtype=dtype)
        else:
            x = torch.randn(256, 128, device=device, dtype=dtype).t()
            y = torch.randn(x.shape, device=device, dtype=dtype)
            if layout == "transposed":
                y = torch.randn(256, 128, device=device, dtype=dtype).t()
        kernel._TUNED_PLAN.clear()
        for alpha in (0.5, 1.5):
            with torch.backends.python_native.cutedsl.disabled():
                expected = torch.add(x, y, alpha=alpha)
            with patch.object(kernel, "_TUNING_ENABLED", True):
                actual = torch.add(x, y, alpha=alpha)
            with torch.backends.python_native.cutedsl.disabled():
                self.assertEqual(actual, expected)
        self.assertTrue(
            any(p.config.path == "tma2d" for p in kernel._TUNED_PLAN.values())
        )

    @onlyCUDA
    @skipIfNoCuteDSL
    @dtypes(torch.bfloat16, torch.float32)
    def test_tma_bool_output_and_capture(self, device: str, dtype: torch.dtype) -> None:
        from torch._native.ops.pointwise import kernel

        if torch.cuda.get_device_capability(device) != (10, 7):
            self.skipTest("Rubin pointwise tuning")
        x = torch.randn(256, 128, device=device, dtype=dtype).t()
        y = torch.randn(x.shape, device=device, dtype=dtype)
        x[0, :3] = torch.tensor(
            (float("nan"), float("inf"), -float("inf")), device=device, dtype=dtype
        )
        out = torch.empty(x.shape, device=device, dtype=torch.bool)
        with torch.backends.python_native.cutedsl.disabled():
            expected = torch.gt(x, y)
        kernel._TUNED_PLAN.clear()
        with patch.object(kernel, "_TUNING_ENABLED", True):
            torch.gt(x, y, out=out)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                torch.gt(x, y, out=out)
            graph.replay()
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(out, expected)
        self.assertTrue(
            any(p.config.path == "tma2d" for p in kernel._TUNED_PLAN.values())
        )


instantiate_device_type_tests(TestPointwiseTma, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
