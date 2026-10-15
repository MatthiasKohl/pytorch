# Owner(s): ["module: dsl-native-ops"]

import contextlib
from collections.abc import Callable
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
class TestCuTeDSLRng(TestCase):
    def _served(self, fn: Callable[[], torch.Tensor]) -> tuple[int, torch.Tensor]:
        from torch._native.ops.rng import cutedsl_kernels

        with mock.patch.object(
            cutedsl_kernels, "fill_random", wraps=cutedsl_kernels.fill_random
        ) as launch:
            result = fn()
        return launch.call_count, result

    def _both(
        self, device: str, fn: Callable[[], torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        gen = torch.cuda.default_generators[torch.device(device).index]
        gen.manual_seed(7)
        before = gen.get_offset()
        ours = fn()
        ours_delta = gen.get_offset() - before
        gen.manual_seed(7)
        before = gen.get_offset()
        with torch.backends.python_native.cutedsl.disabled():
            aten = fn()
        return ours, aten, ours_delta, gen.get_offset() - before

    @parametrize("size", (4, 256, 1000, 4096, 100_000, 1 << 20, (1 << 20) + 7))
    @parametrize(
        "distribution", ("uniform", "uniform_scaled", "normal", "normal_scaled")
    )
    def test_values_and_offset(self, device: str, size: int, distribution: str) -> None:
        def sample() -> torch.Tensor:
            target = torch.empty(size, device=device)
            if distribution == "uniform":
                return target.uniform_()
            if distribution == "uniform_scaled":
                return target.uniform_(-2.0, 2.0)
            if distribution == "normal":
                return target.normal_()
            return target.normal_(10.0, 2.5)

        ours, aten, ours_delta, aten_delta = self._both(device, sample)
        self.assertEqual(ours, aten, atol=0, rtol=0)
        self.assertEqual(ours_delta, aten_delta)

    @parametrize("constructor", ("rand", "randn", "rand_like", "randn_like"))
    def test_constructor_route(self, device: str, constructor: str) -> None:
        def sample() -> torch.Tensor:
            fn = getattr(torch, constructor)
            if constructor.endswith("_like"):
                return fn(torch.empty(4096, device=device))
            return fn(4096, device=device)

        torch.cuda.manual_seed(11)
        count, ours = self._served(sample)
        self.assertEqual(count, 1)
        torch.cuda.manual_seed(11)
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(ours, sample(), atol=0, rtol=0)

    def test_graph_capture(self, device: str) -> None:
        def capture_replays(disabled: bool) -> list[torch.Tensor]:
            ctx = (
                torch.backends.python_native.cutedsl.disabled()
                if disabled
                else contextlib.nullcontext()
            )
            with ctx:
                torch.cuda.manual_seed(99)
                x = torch.empty(4096, device=device)
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        x.uniform_()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    x.uniform_()
                results = []
                for _ in range(3):
                    graph.replay()
                    torch.cuda.synchronize()
                    results.append(x.clone())
                return results

        ours = capture_replays(False)
        self.assertNotEqual(ours[0], ours[1])
        for got, ref in zip(ours, capture_replays(True)):
            self.assertEqual(got, ref, atol=0, rtol=0)

    def test_unsupported_inputs(self, device: str) -> None:
        gen = torch.Generator(device=device).manual_seed(5)
        for fn in (
            lambda: torch.empty(64, device=device, dtype=torch.float64).uniform_(),
            lambda: torch.empty(64, device=device, dtype=torch.float16).normal_(),
            lambda: torch.empty(64, device=device).uniform_(0, 1, generator=gen),
            lambda: torch.empty(8, 8, device=device).t().uniform_(),
            lambda: torch.empty(0, device=device).uniform_(),
        ):
            count, _ = self._served(fn)
            self.assertEqual(count, 0)
        for args in ((), (2.0, 2.0)):
            count, _ = self._served(
                lambda: torch.empty(64, device=device).uniform_(*args)
            )
            self.assertEqual(count, 1)

    def test_compile_count_independent_of_size(self, device: str) -> None:
        from torch._native.ops.rng import cutedsl_kernels

        compile_kernel: Any = cutedsl_kernels._compile
        before = compile_kernel.cache_info().misses
        for size in (1 << 10, 1 << 16, 1 << 20, 1 << 24):
            torch.empty(size, device=device).uniform_()
        self.assertLessEqual(compile_kernel.cache_info().misses - before, 1)


instantiate_device_type_tests(TestCuTeDSLRng, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
