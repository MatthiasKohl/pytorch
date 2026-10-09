# Owner(s): ["module: dsl-native-ops"]

import math
from collections.abc import Callable
from functools import partial
from typing import Any
from unittest import mock

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


@skipIfNoCuteDSL
class TestPointwiseOps(TestCase):
    def _served(self, fn: Callable[[], Any]) -> tuple[int, Any]:
        from torch._native.ops.pointwise import kernel

        with mock.patch.object(kernel, "run", wraps=kernel.run) as launch:
            result = fn()
        return launch.call_count, result

    @parametrize("operation", ("add", "sub", "mul", "div"))
    def test_scalar_operand(self, device: str, operation: str) -> None:
        x = torch.randn(1024, device=device)
        fn = partial(getattr(torch, operation), x, 2.0)
        count, got = self._served(fn)
        with torch.backends.python_native.cutedsl.disabled():
            ref = fn()
        self.assertEqual(got, ref)
        self.assertEqual(count, 1)
        b = x.to(torch.bfloat16)
        self.assertEqual((b * 2.0).dtype, torch.bfloat16)
        i = torch.arange(64, device=device, dtype=torch.int32)
        self.assertEqual((i * 2).dtype, torch.int32)
        self.assertEqual((i + 0.5).dtype, torch.float32)
        with self.assertRaises(RuntimeError):
            x + torch.randn(1024)

    def test_scalar_base_pow(self, device: str) -> None:
        x = torch.tensor([1.0, 2.0], device=device)
        self.assertEqual(2.0**x, torch.tensor([2.0, 4.0], device=device))
        self.assertEqual(torch.pow(2.0, x), torch.tensor([2.0, 4.0], device=device))
        self.assertEqual(torch.float_power(2.0, x), x.double().exp2())
        i = torch.tensor([1, 2], dtype=torch.int32, device=device)
        self.assertEqual(torch.ldexp(i, i), torch.tensor([2.0, 8.0], device=device))

    def test_large_scalar_is_not_baked(self, device: str) -> None:
        from torch._native.ops.pointwise import kernel

        x = torch.randn(1024, device=device)
        for value in (1e16, 1e20, 3.4e38, -1e20, 1e-20, 2.5):
            with torch.backends.python_native.cutedsl.disabled():
                ref = torch.nn.functional.leaky_relu(x, negative_slope=value)
            self.assertEqual(
                torch.nn.functional.leaky_relu(x, negative_slope=value), ref
            )
        count = len(kernel._KERNELS)
        for value in (3.0, 4.0, 5.0, 1e17, 1e18):
            torch.nn.functional.leaky_relu(x, negative_slope=value)
        self.assertEqual(len(kernel._KERNELS), count)

    @parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
    def test_nan_to_num_default_bounds(self, device: str, dtype: torch.dtype) -> None:
        vals = [float("nan"), float("inf"), float("-inf"), 1.5]
        x = torch.tensor(vals, device=device, dtype=dtype)
        with torch.backends.python_native.cutedsl.disabled():
            ref = torch.nan_to_num(x)
        self.assertEqual(torch.nan_to_num(x), ref)

    @parametrize("dtype", (torch.float16, torch.float32, torch.float64, torch.int32))
    @parametrize("shape", ((512,), (256, 2), (64, 8)))
    @parametrize("offset", (0, 1, 2, 7))
    def test_alignment_cache(
        self, device: str, dtype: torch.dtype, shape: tuple[int, ...], offset: int
    ) -> None:
        base = torch.arange(4096, device=device).to(dtype) + 1
        x = base[offset : offset + math.prod(shape)].view(shape)
        for fn in (torch.neg, torch.sigmoid, torch.square):
            with torch.backends.python_native.cutedsl.disabled():
                ref = fn(x)
            self.assertEqual(fn(x), ref)

    def test_empty_copy_destination(self, device: str) -> None:
        dst = torch.empty(0, 2, device=device, dtype=torch.float64)
        dst.copy_(torch.ones(1, device=device, dtype=torch.float64))
        x = torch.empty(0, 2, 2, device=device, dtype=torch.float64)
        with torch.backends.python_native.cutedsl.disabled():
            ref = torch.linalg.matrix_power(x, 0)
        self.assertEqual(torch.linalg.matrix_power(x, 0), ref)

    def test_scalar_first_operand(self, device: str) -> None:
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        i = torch.tensor([2, 4, 6], dtype=torch.int32, device=device)
        for fn in (
            lambda: 1.0 - x,
            lambda: torch.remainder(2, i),
            lambda: torch.xlogy(2.0, x),
            lambda: torch.bitwise_and(3, i),
        ):
            count, got = self._served(fn)
            self.assertGreaterEqual(count, 1)
            with torch.backends.python_native.cutedsl.disabled():
                self.assertEqual(got, fn())
        with self.assertRaises(RuntimeError):
            torch.tensor([1.0, 2.0, 3.0]) + x

    @parametrize(
        "layout", ("contiguous", "unaligned", "transposed", "strided", "scalar")
    )
    def test_fill_layout(self, device: str, layout: str) -> None:
        if layout == "contiguous":
            target = torch.empty(1024, device=device)
        elif layout == "unaligned":
            target = torch.empty(1025, device=device)[1:]
        elif layout == "transposed":
            target = torch.empty(32, 32, device=device).t()
        elif layout == "strided":
            target = torch.empty(2048, device=device)[::2]
        else:
            target = torch.empty((), device=device)
        count, _ = self._served(lambda: target.fill_(2.5))
        self.assertEqual(count, 1)
        self.assertEqual(target, torch.full_like(target, 2.5))

    @parametrize("dtype", (torch.float32, torch.float64, torch.int32, torch.int64))
    def test_range_factories(self, device: str, dtype: torch.dtype) -> None:
        for start, end, steps in ((0, 1, 5), (-1, 1, 9), (5, -5, 64), (2.5, 3.5, 17)):
            fn = partial(torch.linspace, start, end, steps, device=device, dtype=dtype)
            count, got = self._served(fn)
            self.assertEqual(count, 1)
            with torch.backends.python_native.cutedsl.disabled():
                self.assertEqual(got, fn())
        with torch.backends.python_native.cutedsl.disabled():
            ref = torch.arange(0, 10, 1, device=device, dtype=dtype)
        out = torch.empty_like(ref)
        count, got = self._served(lambda: torch.arange(0, 10, 1, out=out))
        self.assertEqual(count, 1)
        self.assertEqual(got, ref)

    @parametrize("eps", (None, 0.0, 0.001, 0.1, 0.6))
    def test_optional_scalar(self, device: str, eps: float | None) -> None:
        vals = [0.0, 1e-8, 0.5, 0.9999, 1.5, -0.5, float("nan")]
        x = torch.tensor(vals, device=device)
        with torch.backends.python_native.cutedsl.disabled():
            ref = torch.logit(x, eps)
        self.assertEqual(torch.logit(x, eps), ref, equal_nan=True)

    def test_lazy_metadata_declines(self, device: str) -> None:
        x = torch.randn(64, device=device, dtype=torch.float64)._neg_view()
        self.assertTrue(x.is_neg())
        for fn in (
            torch.sin,
            torch.relu,
            torch.nan_to_num,
            lambda t: t * 2.0,
            lambda t: t.clone(),
            lambda t: t.float(),
        ):
            with torch.backends.python_native.cutedsl.disabled():
                ref = fn(x)
            self.assertEqual(fn(x), ref)
        c = torch.randn(64, device=device, dtype=torch.complex64).conj()
        with torch.backends.python_native.cutedsl.disabled():
            ref = c.real
        self.assertEqual(c.real, ref)


instantiate_device_type_tests(TestPointwiseOps, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
