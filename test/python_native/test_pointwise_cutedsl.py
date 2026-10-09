# Owner(s): ["module: dsl-native-ops"]

from unittest.mock import patch

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    largeTensorTest,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


class TestPointwiseIndexWidth(TestCase):
    @skipIfNoCuteDSL
    @parametrize("case", ("small", "count", "rounded", "offset"))
    def test_index_width(self, case: str) -> None:
        import cutlass

        from torch._native.ops.pointwise import kernel

        size = 2**31 if case == "count" else 2**31 - 1 if case == "rounded" else 32
        tensor = (
            torch.empty_strided((2, 2), (2**31, 1), device="meta")
            if case == "offset"
            else torch.empty(size, device="meta")
        )
        expected = cutlass.Int32 if case == "small" else cutlass.Int64
        self.assertIs(kernel._index_type((tensor,), tensor.shape, 256), expected)


class TestPointwiseCuTeDSL(TestCase):
    @onlyCUDA
    @skipIfNoCuteDSL
    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("layout", ("mixed", "transposed", "rank3"))
    def test_tma_pointwise(self, device: str, dtype: torch.dtype, layout: str) -> None:
        from torch._native.ops.pointwise import tuned

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
        tuned._PLAN.clear()
        with patch.object(tuned, "_ENABLED", True):
            actual = torch.add(x, y, alpha=0.5)
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(actual, expected)
        self.assertTrue(any(p.config.path == "tma2d" for p in tuned._PLAN.values()))

    @onlyCUDA
    @skipIfNoCuteDSL
    @dtypes(torch.bfloat16, torch.float32)
    def test_tma_bool_output_and_capture(self, device: str, dtype: torch.dtype) -> None:
        from torch._native.ops.pointwise import tuned

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
        tuned._PLAN.clear()
        with patch.object(tuned, "_ENABLED", True):
            torch.gt(x, y, out=out)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                torch.gt(x, y, out=out)
            graph.replay()
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(out, expected)
        self.assertTrue(any(p.config.path == "tma2d" for p in tuned._PLAN.values()))

    @onlyCUDA
    @skipIfNoCuteDSL
    @dtypes(torch.bfloat16, torch.float32)
    @parametrize("transposed", (False, True))
    def test_wrapped_scalar_cache(
        self, device: str, dtype: torch.dtype, transposed: bool
    ) -> None:
        from torch._native.ops.pointwise import tuned

        if torch.cuda.get_device_capability(device) != (10, 7):
            self.skipTest("Rubin pointwise tuning")
        x = torch.randn(256, 128, device=device, dtype=dtype)
        if transposed:
            x = x.t()
        tuned._PLAN.clear()
        for scalar in (0.5, 1.5):
            with torch.backends.python_native.cutedsl.disabled():
                expected = torch.add(x, scalar)
            with patch.object(tuned, "_ENABLED", True):
                actual = torch.add(x, scalar)
            with torch.backends.python_native.cutedsl.disabled():
                self.assertEqual(actual, expected)
        self.assertEqual(len(tuned._PLAN), 1)
        plan = next(iter(tuned._PLAN.values()))
        self.assertEqual(plan.config.path, "tma2d" if transposed else "vector")

    @onlyCUDA
    @skipIfNoCuteDSL
    def test_unsupported_layout_and_overlap(self, device: str) -> None:
        from torch._native.ops.pointwise import tuned

        x = torch.randn(128, 128, device=device)
        self.assertFalse(tuned.can_run((x.t(),), x.shape, x.dtype, x))
        self.assertFalse(tuned.can_run((x[:, ::2],), (128, 64), x.dtype))
        self.assertFalse(tuned.can_run((x, x[:1]), x.shape, x.dtype))
        tail = torch.randn(2049, device=device)
        self.assertFalse(tuned.can_run((tail,), tail.shape, tail.dtype))
        cow = x._lazy_clone()
        self.assertFalse(tuned.can_run((x,), x.shape, x.dtype, cow))

    @onlyCUDA
    @skipIfNoCuteDSL
    def test_inplace_and_cow_input(self, device: str) -> None:
        from torch._native.ops.pointwise import tuned

        if torch.cuda.get_device_capability(device) != (10, 7):
            self.skipTest("Rubin pointwise tuning")
        x = torch.randn(128, 256, device=device)
        cow = x._lazy_clone()
        with torch.backends.python_native.cutedsl.disabled():
            expected = torch.add(x, 0.5)
        with patch.object(tuned, "_ENABLED", True):
            actual = torch.add(cow, 0.5)
            x.add_(0.5)
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(actual, expected)
            self.assertEqual(x, expected)
        self.assertTrue(torch._C._is_cow_tensor(cow))

    @onlyCUDA
    @skipIfNoCuteDSL
    def test_rubin_vector_policy(self, device: str) -> None:
        from torch._native.ops.pointwise import tuned

        if torch.cuda.get_device_capability(device) != (10, 7):
            self.skipTest("Rubin pointwise tuning")
        x = torch.randn(5 * 1024 * 1024, device=device, dtype=torch.bfloat16)
        with torch.backends.python_native.cutedsl.disabled():
            expected = torch.add(x, 0.5)
        tuned._PLAN.clear()
        with patch.object(tuned, "_ENABLED", True):
            actual = torch.add(x, 0.5)
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(actual, expected)
        config = next(iter(tuned._PLAN.values())).config
        self.assertEqual((config.vec, config.elements_per_thread), (8, 8))

    @onlyCUDA
    @skipIfNoCuteDSL
    def test_vector_kernel_reused_across_shapes(self, device: str) -> None:
        from torch._native.ops.pointwise import tuned

        if torch.cuda.get_device_capability(device) != (10, 7):
            self.skipTest("Rubin pointwise tuning")
        tuned._PLAN.clear()
        tuned._KERNELS.clear()
        for size in (2048, 4096):
            x = torch.randn(size, device=device, dtype=torch.bfloat16)
            with torch.backends.python_native.cutedsl.disabled():
                expected = torch.add(x, 0.5)
            with patch.object(tuned, "_ENABLED", True):
                actual = torch.add(x, 0.5)
            with torch.backends.python_native.cutedsl.disabled():
                self.assertEqual(actual, expected)
        self.assertEqual(len(tuned._PLAN), 2)
        self.assertEqual(len(tuned._KERNELS), 1)

    @onlyCUDA
    @skipIfNoCuteDSL
    @parametrize("reflected", (False, True))
    def test_generic_scalar_capture(self, device: str, reflected: bool) -> None:
        from torch._native.ops.pointwise import tuned

        x = torch.randn(128, 256, device=device)
        out = torch.empty_like(x)
        with torch.backends.python_native.cutedsl.disabled():
            expected = torch.rsub(x, 0.5) if reflected else torch.add(x, 0.5)
        with patch.object(tuned, "_ENABLED", False):
            if reflected:
                out = torch.rsub(x, 0.5)
            else:
                torch.add(x, 0.5, out=out)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                if reflected:
                    out = torch.rsub(x, 0.5)
                else:
                    torch.add(x, 0.5, out=out)
            graph.replay()
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(out, expected)

    @onlyCUDA
    @skipIfNoCuteDSL
    @dtypes(torch.bfloat16, torch.float32)
    @parametrize("transposed", (False, True))
    def test_singleton_dimensions(
        self, device: str, dtype: torch.dtype, transposed: bool
    ) -> None:
        from torch._native.ops.pointwise import tuned

        x = torch.randn(128, 1, 256, device=device, dtype=dtype)
        if transposed:
            x = x.permute(2, 1, 0)
        y = torch.randn(x.shape, device=device, dtype=dtype)
        with torch.backends.python_native.cutedsl.disabled():
            expected = torch.add(x, y)
        with patch.object(tuned, "_ENABLED", True):
            actual = torch.add(x, y)
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(actual, expected)

    @onlyCUDA
    @skipIfNoCuteDSL
    @largeTensorTest("19GB", device="cuda")
    @parametrize("size", (2**31, 2**31 + 32))
    @parametrize("tuning", (False, True))
    @parametrize("boolean", (False, True))
    def test_large_index(
        self, device: str, size: int, tuning: bool, boolean: bool
    ) -> None:
        from torch._native.ops.pointwise import tuned

        cc = torch.cuda.get_device_capability(device)
        if cc[0] not in (9, 10) or tuning and cc != (10, 7):
            self.skipTest("pointwise kernels require SM90+ and tuning requires Rubin")
        samples = (0, 31, 2**31 - 33, 2**31 - 1, size - 1)
        with torch.backends.python_native.cutedsl.disabled():
            x = torch.ones(size, device=device, dtype=torch.bfloat16)
            y = torch.full_like(x, 0.5)
            out = torch.zeros(
                size, device=device, dtype=torch.bool if boolean else x.dtype
            )
            indices = torch.tensor(samples, device=device)
            x[indices[0]] = -1
            expected = x[indices] > y[indices] if boolean else x[indices] + y[indices]
        self.assertGreater(x.numel(), torch.iinfo(torch.int32).max)
        tuned._PLAN.clear()
        with patch.object(tuned, "_ENABLED", tuning):
            if boolean:
                torch.gt(x, y, out=out)
            else:
                torch.add(x, y, out=out)
        if tuning:
            config = next(iter(tuned._PLAN.values())).config
            self.assertEqual(config.path, "vector")
            self.assertEqual(config.index_bits, 64 if size > 2**31 else 32)
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(out[indices], expected)


instantiate_device_type_tests(TestPointwiseCuTeDSL, globals(), only_for="cuda")
instantiate_parametrized_tests(TestPointwiseIndexWidth)


if __name__ == "__main__":
    run_tests()
