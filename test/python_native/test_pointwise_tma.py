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
        with torch.backends.python_native.cutedsl.disabled():
            expected = torch.add(x, y, alpha=0.5)
        kernel._TUNED_PLAN.clear()
        with patch.object(kernel, "_TUNING_ENABLED", True):
            actual = torch.add(x, y, alpha=0.5)
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

    @onlyCUDA
    @skipIfNoCuteDSL
    def test_rubin_vector_policy(self, device: str) -> None:
        from torch._native.ops.pointwise import kernel

        if torch.cuda.get_device_capability(device) != (10, 7):
            self.skipTest("Rubin pointwise tuning")
        x = torch.randn(5 * 1024 * 1024, device=device, dtype=torch.bfloat16)
        with torch.backends.python_native.cutedsl.disabled():
            expected = torch.neg(x)
        kernel._TUNED_PLAN.clear()
        with patch.object(kernel, "_TUNING_ENABLED", True):
            actual = torch.neg(x)
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(actual, expected)
        config = next(iter(kernel._TUNED_PLAN.values())).config
        self.assertEqual((config.vec, config.elements_per_thread), (8, 8))

    @onlyCUDA
    @skipIfNoCuteDSL
    @dtypes(torch.bfloat16, torch.float32)
    @parametrize("transposed", (False, True))
    def test_runtime_constant_cache(
        self, device: str, dtype: torch.dtype, transposed: bool
    ) -> None:
        from torch._native.ops.pointwise import kernel

        if torch.cuda.get_device_capability(device) != (10, 7):
            self.skipTest("Rubin pointwise tuning")
        x = torch.randn(256, 128, device=device, dtype=dtype)
        if transposed:
            x = x.t()
        y = torch.randn_like(x)
        kernel._TUNED_PLAN.clear()
        for scalar in (0.5, 1.5):
            with torch.backends.python_native.cutedsl.disabled():
                expected = torch.add(x, y, alpha=scalar)
            with patch.object(kernel, "_TUNING_ENABLED", True):
                actual = torch.add(x, y, alpha=scalar)
            with torch.backends.python_native.cutedsl.disabled():
                self.assertEqual(actual, expected)
        self.assertEqual(len(kernel._TUNED_PLAN), 1)
        plan = next(iter(kernel._TUNED_PLAN.values()))
        self.assertEqual(plan.config.path, "tma2d" if transposed else "vector")

    @onlyCUDA
    @skipIfNoCuteDSL
    def test_unsupported_layout_and_overlap(self, device: str) -> None:
        from torch._native.ops.pointwise import kernel

        x = torch.randn(128, 128, device=device)
        self.assertFalse(kernel.can_run_tuned((x.t(),), x.shape, x.dtype, x))
        self.assertFalse(kernel.can_run_tuned((x[:, ::2],), (128, 64), x.dtype))
        self.assertFalse(kernel.can_run_tuned((x, x[:1]), x.shape, x.dtype))
        tail = torch.randn(2049, device=device)
        self.assertFalse(kernel.can_run_tuned((tail,), tail.shape, tail.dtype))
        cow = x._lazy_clone()
        self.assertFalse(kernel.can_run_tuned((x,), x.shape, x.dtype, cow))

    @onlyCUDA
    @skipIfNoCuteDSL
    def test_inplace_and_cow_input(self, device: str) -> None:
        from torch._native.ops.pointwise import kernel

        if torch.cuda.get_device_capability(device) != (10, 7):
            self.skipTest("Rubin pointwise tuning")
        x = torch.randn(128, 256, device=device)
        cow = x._lazy_clone()
        with torch.backends.python_native.cutedsl.disabled():
            expected = torch.neg(x)
        with patch.object(kernel, "_TUNING_ENABLED", True):
            actual = torch.neg(cow)
            x.neg_()
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(actual, expected)
            self.assertEqual(x, expected)
        self.assertTrue(torch._C._is_cow_tensor(cow))

    @onlyCUDA
    @skipIfNoCuteDSL
    def test_vector_kernel_reused_across_shapes(self, device: str) -> None:
        from torch._native.ops.pointwise import kernel

        if torch.cuda.get_device_capability(device) != (10, 7):
            self.skipTest("Rubin pointwise tuning")
        kernel._TUNED_PLAN.clear()
        kernel._KERNELS.clear()
        for size in (2048, 4096):
            x = torch.randn(size, device=device, dtype=torch.bfloat16)
            with torch.backends.python_native.cutedsl.disabled():
                expected = torch.neg(x)
            with patch.object(kernel, "_TUNING_ENABLED", True):
                actual = torch.neg(x)
            with torch.backends.python_native.cutedsl.disabled():
                self.assertEqual(actual, expected)
        self.assertEqual(len(kernel._TUNED_PLAN), 2)
        self.assertEqual(len(kernel._KERNELS), 1)


instantiate_device_type_tests(TestPointwiseTma, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
