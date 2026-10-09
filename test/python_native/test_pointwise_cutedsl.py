# Owner(s): ["module: dsl-native-ops"]

import torch
from torch.testing._internal.common_device_type import (
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
    @parametrize("reflected", (False, True))
    def test_generic_scalar_capture(self, device: str, reflected: bool) -> None:
        x = torch.randn(128, 256, device=device)
        out = torch.empty_like(x)
        with torch.backends.python_native.cutedsl.disabled():
            expected = torch.rsub(x, 0.5) if reflected else torch.add(x, 0.5)
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
    @largeTensorTest("19GB", device="cuda")
    @parametrize("size", (2**31, 2**31 + 32))
    @parametrize("boolean", (False, True))
    def test_large_index(self, device: str, size: int, boolean: bool) -> None:
        cc = torch.cuda.get_device_capability(device)
        if cc[0] not in (9, 10):
            self.skipTest("pointwise kernels require SM90+")
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
        if boolean:
            torch.gt(x, y, out=out)
        else:
            torch.add(x, y, out=out)
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(out[indices], expected)


instantiate_device_type_tests(TestPointwiseCuTeDSL, globals(), only_for="cuda")
instantiate_parametrized_tests(TestPointwiseIndexWidth)


if __name__ == "__main__":
    run_tests()
