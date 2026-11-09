from __future__ import annotations

from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from cutlass.cute.nvgpu import cpasync


class Transpose:
    def __init__(self, n: int, tile: int, threads: int, asynchronous: bool, add_one: bool, *, rows: int | None = None) -> None:
        self.n = n
        self.input_rows = n if rows is None else rows
        self.tile = tile
        self.threads = threads
        self.asynchronous = asynchronous
        self.add_one = add_one

    @cute.jit
    def __call__(self, x: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream) -> None:
        self.kernel(x, out).launch(
            grid=(self.input_rows // self.tile, self.n // self.tile, 1),
            block=(self.threads, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, x: cute.Tensor, out: cute.Tensor) -> None:
        tid, _, _ = cute.arch.thread_idx()
        bi, bj, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        shared = smem.allocate_tensor(cutlass.Float32, cute.make_layout(self.tile * self.tile), byte_alignment=128)
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), cutlass.Float32, num_bits_per_copy=128)
        groups = const_expr(self.tile * self.tile // (self.threads * 4))
        for group in cutlass.range_constexpr(groups):
            index = (tid + group * self.threads) * 4
            row, col = index // self.tile, index % self.tile
            source = cute.make_tensor((x.iterator + (bi * self.tile + row) * self.n + bj * self.tile + col).align(16), cute.make_layout(4))
            # XOR whole 16-byte groups, preserving LDGSTS alignment.
            swizzled = row * self.tile + (col ^ ((row % 8) * 4))
            target = cute.make_tensor((shared.iterator + swizzled).align(16), cute.make_layout(4))
            if const_expr(self.asynchronous):
                cute.copy(atom, source, target)
            else:
                fragment = cute.make_rmem_tensor((4,), cutlass.Float32)
                cute.autovec_copy(source, fragment)
                cute.autovec_copy(fragment, target)
        if const_expr(self.asynchronous):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        for group in cutlass.range_constexpr(groups):
            index = (tid + group * self.threads) * 4
            row, col = index // self.tile, index % self.tile
            result = cute.make_rmem_tensor((4,), cutlass.Float32)
            for v in cutlass.range_constexpr(4):
                source_index = (col + v) * self.tile + (row ^ (((col + v) % 8) * 4))
                value = shared[source_index]
                if const_expr(self.add_one):
                    value += cutlass.Float32(1)
                result[v] = value
            target = cute.make_tensor((out.iterator + (bj * self.tile + row) * self.input_rows + bi * self.tile + col).align(16), cute.make_layout(4))
            cute.autovec_copy(result, target)


class ClassicTranspose:
    def __init__(self, n: int, columns: int, threads: int, asynchronous: bool, add_one: bool, *, rows: int | None = None) -> None:
        self.n = n
        self.input_rows = n if rows is None else rows
        self.rows = 32
        self.columns = columns
        self.threads = threads
        self.asynchronous = asynchronous
        self.add_one = add_one

    @cute.jit
    def __call__(self, x: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream) -> None:
        self.kernel(x, out).launch(grid=(self.input_rows // self.rows, self.n // self.columns, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Tensor, out: cute.Tensor) -> None:
        tid, _, _ = cute.arch.thread_idx()
        bi, bj, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        shared = smem.allocate_tensor(cutlass.Float32, cute.make_layout(self.rows * (self.columns + 1)), byte_alignment=128)
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.Float32, num_bits_per_copy=32)
        for group in cutlass.range_constexpr(self.rows * self.columns // self.threads):
            index = tid + group * self.threads
            row, col = index // self.columns, index % self.columns
            offset = (bi * self.rows + row) * self.n + bj * self.columns + col
            target = row * (self.columns + 1) + col
            if const_expr(self.asynchronous):
                source = cute.make_tensor(x.iterator + offset, cute.make_layout(1))
                dest = cute.make_tensor(shared.iterator + target, cute.make_layout(1))
                cute.copy(atom, source, dest)
            else:
                shared[target] = x[offset]
        if const_expr(self.asynchronous):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        for group in cutlass.range_constexpr(self.rows * self.columns // self.threads):
            index = tid + group * self.threads
            row, col = index // self.rows, index % self.rows
            value = shared[col * (self.columns + 1) + row]
            if const_expr(self.add_one):
                value += cutlass.Float32(1)
            out[(bj * self.columns + row) * self.input_rows + bi * self.rows + col] = value
