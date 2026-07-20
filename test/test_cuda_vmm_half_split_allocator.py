# Owner(s): ["module: cuda"]

import gc
import unittest
import weakref

import torch
from torch.cuda.green_contexts import is_localization_supported
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    run_tests,
    serialTest,
    TEST_CUDA_GRAPH,
    TestCase,
)


MIB = 1024 * 1024


class TestVMMHalfSplitAllocator(TestCase):
    def setUp(self):
        super().setUp()
        if not is_localization_supported():
            self.skipTest("CUDA locality domains are not available")
        self.pool = torch.cuda.VMMHalfSplitAllocator()

    def tearDown(self):
        gc.collect()
        torch.cuda.synchronize()
        if self.pool.use_count() == 1:
            torch.cuda.empty_allocator_cache(self.pool)
        super().tearDown()

    @onlyCUDA
    @serialTest()
    def test_best_fit_reuse_is_centered(self, device):
        with torch.cuda.use_mem_pool(self.pool):
            large = torch.full((20 * MIB,), 1, dtype=torch.uint8, device=device)
            medium = torch.full((12 * MIB,), 2, dtype=torch.uint8, device=device)
        medium_ptr = medium.data_ptr()
        del large, medium
        gc.collect()

        with torch.cuda.use_mem_pool(self.pool):
            small = torch.empty(8 * MIB, dtype=torch.uint8, device=device)
        self.assertEqual(small.data_ptr(), medium_ptr + 2 * MIB)
        self.assertEqual(small[0], 2)
        active_segments = [
            segment
            for segment in self.pool.vmm_snapshot()["segments"]
            if any(block["state"].startswith("active_") for block in segment["blocks"])
        ]
        self.assertEqual(len(active_segments), 1)
        self.assertEqual(
            [(block["state"], block["size"]) for block in active_segments[0]["blocks"]],
            [
                ("inactive", 2 * MIB),
                ("active_allocated", 8 * MIB),
                ("inactive", 2 * MIB),
            ],
        )
        stats = self.pool.stats()
        self.assertEqual(stats["mapped_slot_creations"], 2)
        self.assertEqual(stats["mapped_slot_cache_hits"], 1)

    @onlyCUDA
    @serialTest()
    def test_handle_size_classes_and_empty_cache(self, device):
        with torch.cuda.use_mem_pool(self.pool):
            value = torch.empty(124 * MIB, dtype=torch.uint8, device=device)
        expected = {
            2 * MIB: 2,
            4 * MIB: 2,
            8 * MIB: 2,
            16 * MIB: 2,
            32 * MIB: 2,
        }
        stats = self.pool.stats()
        self.assertEqual(stats["api"]["cuMemCreate"]["size_histogram"], expected)
        self.assertEqual(
            stats["property_0_reserved_bytes"], stats["property_1_reserved_bytes"]
        )
        del value
        gc.collect()
        torch.cuda.empty_allocator_cache(self.pool)
        stats = self.pool.stats()
        self.assertEqual(stats["reserved_bytes"], 0)
        self.assertEqual(stats["api"]["cuMemUnmap"]["call_count"], 1)
        self.assertEqual(stats["api"]["cuMemRelease"]["call_count"], 10)

    @onlyCUDA
    @serialTest()
    def test_tensor_record_stream_delays_raw_free(self, device):
        with torch.cuda.use_mem_pool(self.pool):
            value = torch.empty(4 * MIB, dtype=torch.uint8, device=device)
        original_ptr = value.data_ptr()
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            value.fill_(7)
            torch.cuda._sleep(2_000_000_000)
            value.record_stream(side)
        del value
        self.assertEqual(self.pool.stats()["cached_mapped_bytes"], 0)

        with torch.cuda.use_mem_pool(self.pool):
            replacement = torch.empty(4 * MIB, dtype=torch.uint8, device=device)
        self.assertNotEqual(replacement.data_ptr(), original_ptr)
        side.synchronize()
        del replacement
        gc.collect()

        with torch.cuda.use_mem_pool(self.pool):
            reused = torch.empty(4 * MIB, dtype=torch.uint8, device=device)
        self.assertEqual(reused.data_ptr(), original_ptr)
        self.assertEqual(reused[0], 7)

    @onlyCUDA
    @serialTest()
    def test_empty_cache_drains_deferred_free(self, device):
        with torch.cuda.use_mem_pool(self.pool):
            value = torch.empty(4 * MIB, dtype=torch.uint8, device=device)
        side = torch.cuda.Stream()
        value.record_stream(side)
        del value
        gc.collect()
        self.assertEqual(self.pool.stats()["free_completions"], 0)

        torch.cuda.empty_allocator_cache(self.pool)
        stats = self.pool.stats()
        self.assertEqual(stats["free_completions"], 1)
        self.assertEqual(stats["cached_mapped_bytes"], 0)
        self.assertEqual(stats["reserved_bytes"], 0)

    @onlyCUDA
    @serialTest()
    def test_empty_cache_preserves_active_allocation(self, device):
        with torch.cuda.use_mem_pool(self.pool):
            value = torch.full((4 * MIB,), 3, dtype=torch.uint8, device=device)
        reserved = self.pool.stats()["reserved_bytes"]
        torch.cuda.empty_allocator_cache(self.pool)
        self.assertEqual(self.pool.stats()["reserved_bytes"], reserved)
        self.assertEqual(value[0], 3)

    @onlyCUDA
    @serialTest()
    def test_mapped_cache_is_per_stream(self, device):
        with torch.cuda.use_mem_pool(self.pool):
            first = torch.empty(4 * MIB, dtype=torch.uint8, device=device)
        first_ptr = first.data_ptr()
        del first
        gc.collect()

        side = torch.cuda.Stream()
        with torch.cuda.stream(side), torch.cuda.use_mem_pool(self.pool):
            second = torch.empty(4 * MIB, dtype=torch.uint8, device=device)
        self.assertNotEqual(second.data_ptr(), first_ptr)

    @onlyCUDA
    @serialTest()
    def test_checkpoint_restore_is_exact_and_reusable(self, device):
        with torch.cuda.use_mem_pool(self.pool):
            value = torch.full((4 * MIB,), 5, dtype=torch.uint8, device=device)
        value_ptr = value.data_ptr()
        state = torch._C._cuda_getCheckpointState(
            torch.cuda.current_device(), self.pool.id
        )

        for size in (12 * MIB, 8 * MIB):
            with torch.cuda.use_mem_pool(self.pool):
                stale = torch.empty(size, dtype=torch.uint8, device=device)
            api_before = self.pool.stats()["api"]
            torch._C._cuda_setCheckpointPoolState(
                torch.cuda.current_device(),
                state,
                [stale.untyped_storage()._cdata],
                [],
            )
            self.assertEqual(self.pool.stats()["api"], api_before)
            self.assertEqual(value.data_ptr(), value_ptr)
            self.assertEqual(value, torch.full_like(value, 5))
            active_blocks = [
                block
                for segment in self.pool.vmm_snapshot()["segments"]
                for block in segment["blocks"]
                if block["state"] == "active_allocated"
            ]
            self.assertEqual(len(active_blocks), 1)
            self.assertEqual(active_blocks[0]["address"], value_ptr)
            self.assertEqual(active_blocks[0]["requested_size"], 4 * MIB)
            del stale

    @onlyCUDA
    @serialTest()
    def test_checkpoint_restore_out_of_order(self, device):
        with torch.cuda.use_mem_pool(self.pool):
            first = torch.full((4 * MIB,), 1, dtype=torch.uint8, device=device)
        first_state = torch._C._cuda_getCheckpointState(
            torch.cuda.current_device(), self.pool.id
        )
        with torch.cuda.use_mem_pool(self.pool):
            second = torch.full((8 * MIB,), 2, dtype=torch.uint8, device=device)
        second_ptr = second.data_ptr()
        second_state = torch._C._cuda_getCheckpointState(
            torch.cuda.current_device(), self.pool.id
        )

        api_before = self.pool.stats()["api"]
        torch._C._cuda_setCheckpointPoolState(
            torch.cuda.current_device(),
            first_state,
            [second.untyped_storage()._cdata],
            [],
        )
        self.assertEqual(self.pool.stats()["api"], api_before)
        self.assertEqual(first, torch.full_like(first, 1))

        del second
        second_storage = torch._C._construct_storage_from_data_pointer(
            second_ptr, torch.device(device), 8 * MIB
        )
        second = torch.empty(0, dtype=torch.uint8, device=device)
        second.set_(second_storage, 0, (8 * MIB,), (1,))
        torch._C._cuda_setCheckpointPoolState(
            torch.cuda.current_device(),
            second_state,
            [],
            [second.untyped_storage()._cdata],
        )
        self.assertEqual(self.pool.stats()["api"], api_before)
        self.assertEqual(first, torch.full_like(first, 1))
        self.assertEqual(second, torch.full_like(second, 2))

    @onlyCUDA
    @serialTest()
    def test_checkpoint_restores_record_stream_dependency(self, device):
        with torch.cuda.use_mem_pool(self.pool):
            value = torch.zeros(4 * MIB, dtype=torch.uint8, device=device)
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            value.add_(1)
            value.record_stream(side)
        state = torch._C._cuda_getCheckpointState(
            torch.cuda.current_device(), self.pool.id
        )

        torch._C._cuda_setCheckpointPoolState(
            torch.cuda.current_device(), state, [], []
        )
        side.synchronize()
        frees_before = self.pool.stats()["free_completions"]
        del value
        gc.collect()
        self.assertEqual(self.pool.stats()["free_completions"], frees_before)

        torch.cuda.empty_allocator_cache(self.pool)
        self.assertEqual(self.pool.stats()["free_completions"], frees_before + 1)

    @onlyCUDA
    @serialTest()
    def test_checkpoint_restores_centered_pointer_and_metadata(self, device):
        side = torch.cuda.Stream()
        original_size = 12 * MIB + 257
        replacement_size = 8 * MIB + 129
        with torch.cuda.stream(side), torch.cuda.use_mem_pool(self.pool):
            original = torch.empty(original_size, dtype=torch.uint8, device=device)
        original_ptr = original.data_ptr()
        state = torch._C._cuda_getCheckpointState(
            torch.cuda.current_device(), self.pool.id
        )
        del original
        gc.collect()

        with torch.cuda.stream(side), torch.cuda.use_mem_pool(self.pool):
            replacement = torch.empty(
                replacement_size, dtype=torch.uint8, device=device
            )
        self.assertEqual(replacement.data_ptr(), original_ptr + 2 * MIB)
        api_before = self.pool.stats()["api"]
        torch._C._cuda_setCheckpointPoolState(
            torch.cuda.current_device(),
            state,
            [replacement.untyped_storage()._cdata],
            [],
        )
        self.assertEqual(self.pool.stats()["api"], api_before)
        active_segments = [
            segment
            for segment in self.pool.vmm_snapshot()["segments"]
            if any(block["state"] == "active_allocated" for block in segment["blocks"])
        ]
        self.assertEqual(len(active_segments), 1)
        active_block = next(
            block
            for block in active_segments[0]["blocks"]
            if block["state"] == "active_allocated"
        )
        self.assertEqual(active_block["address"], original_ptr)
        self.assertEqual(active_block["requested_size"], original_size)
        self.assertEqual(active_segments[0]["stream"], side.cuda_stream)
        del replacement
        torch._C._cuda_cudaCachingAllocator_raw_delete(original_ptr)
        side.synchronize()

    @onlyCUDA
    @serialTest()
    def test_checkpoint_pins_cached_mappings(self, device):
        with torch.cuda.use_mem_pool(self.pool):
            value = torch.empty(4 * MIB, dtype=torch.uint8, device=device)
        state = torch._C._cuda_getCheckpointState(
            torch.cuda.current_device(), self.pool.id
        )
        del value
        gc.collect()

        stats_before = self.pool.stats()
        reserved = stats_before["reserved_bytes"]
        unmaps = stats_before["api"].get("cuMemUnmap", {}).get("call_count", 0)
        torch.cuda.empty_allocator_cache(self.pool)
        stats = self.pool.stats()
        self.assertEqual(stats["reserved_bytes"], reserved)
        self.assertEqual(
            stats["api"].get("cuMemUnmap", {}).get("call_count", 0), unmaps
        )

        del state
        gc.collect()
        torch.cuda.empty_allocator_cache(self.pool)
        stats = self.pool.stats()
        self.assertEqual(stats["reserved_bytes"], 0)
        self.assertEqual(stats["api"]["cuMemUnmap"]["call_count"], unmaps + 1)

    @onlyCUDA
    @serialTest()
    def test_checkpoint_restore_drains_deferred_frees(self, device):
        with torch.cuda.use_mem_pool(self.pool):
            value = torch.full((4 * MIB,), 4, dtype=torch.uint8, device=device)
        state = torch._C._cuda_getCheckpointState(
            torch.cuda.current_device(), self.pool.id
        )
        with torch.cuda.use_mem_pool(self.pool):
            stale = torch.empty(4 * MIB, dtype=torch.uint8, device=device)
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            stale.fill_(7)
            torch.cuda._sleep(10_000_000)
            stale.record_stream(side)

        api_before = self.pool.stats()["api"]
        torch._C._cuda_setCheckpointPoolState(
            torch.cuda.current_device(),
            state,
            [stale.untyped_storage()._cdata],
            [],
        )
        self.assertEqual(self.pool.stats()["api"], api_before)
        self.assertEqual(value, torch.full_like(value, 4))
        del stale

    def _warm_graph_kernels(self, device):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            value = torch.ones(16, device=device)
            value.mul_(2)
        torch.cuda.current_stream().wait_stream(stream)

    @onlyCUDA
    @serialTest()
    @unittest.skipIf(not TEST_CUDA_GRAPH, "CUDA graph support is required")
    def test_explicit_graph_pool(self, device):
        self._warm_graph_kernels(device)
        source = torch.ones(16, device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            result = source * 3
        self.assertEqual(self.pool.use_count(), 2)
        graph.replay()
        self.assertEqual(result, torch.full_like(result, 3))
        with self.assertRaisesRegex(RuntimeError, "only when it is inactive"):
            torch.cuda.empty_allocator_cache(self.pool)
        graph.reset()
        self.assertEqual(self.pool.use_count(), 1)
        torch.cuda.empty_allocator_cache(self.pool)
        self.assertEqual(result, torch.full_like(result, 3))

    @onlyCUDA
    @serialTest()
    @unittest.skipIf(not TEST_CUDA_GRAPH, "CUDA graph support is required")
    def test_nested_pool_in_graph(self, device):
        self._warm_graph_kernels(device)
        source = torch.ones(16, device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prefix = source + 1
            with torch.cuda.use_mem_pool(self.pool):
                result = prefix * 2
            result.add_(1)
        self.assertEqual(self.pool.use_count(), 2)
        graph.replay()
        self.assertEqual(result, torch.full_like(result, 5))
        graph.reset()
        self.assertEqual(self.pool.use_count(), 1)
        torch.cuda.empty_allocator_cache(self.pool)
        self.assertEqual(result, torch.full_like(result, 5))

    @onlyCUDA
    @serialTest()
    @unittest.skipIf(not TEST_CUDA_GRAPH, "CUDA graph support is required")
    def test_first_mapping_during_capture(self, device):
        self._warm_graph_kernels(device)
        self.assertEqual(self.pool.stats()["mapped_slot_creations"], 0)
        source = torch.ones(16, device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            result = source + 1
        self.assertGreater(self.pool.stats()["mapped_slot_creations"], 0)
        graph.replay()
        self.assertEqual(result, torch.full_like(result, 2))
        graph.reset()

    @onlyCUDA
    @serialTest()
    @unittest.skipIf(not TEST_CUDA_GRAPH, "CUDA graph support is required")
    def test_graph_retains_allocator_after_wrapper_dies(self, device):
        self._warm_graph_kernels(device)
        pool = torch.cuda.VMMHalfSplitAllocator()
        pool_ref = weakref.ref(pool)
        source = torch.ones(16, device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            result = source * 3

        del pool
        gc.collect()
        self.assertIsNone(pool_ref())
        graph.replay()
        self.assertEqual(result, torch.full_like(result, 3))

        graph.reset()
        del result, source
        gc.collect()
        torch.cuda.empty_cache()

    @onlyCUDA
    @serialTest()
    def test_vmm_and_native_traces_have_separate_roles(self, device):
        torch.cuda.memory._record_memory_history(max_entries=1000)
        self.pool.start_recording(100, clear=True)
        try:
            self.pool.set_metadata('{"iteration":0,"phase":"forward"}')
            with torch.cuda.use_mem_pool(self.pool):
                value = torch.empty(1, dtype=torch.uint8, device=device)
            pointer = value.data_ptr()
            native = self.pool.snapshot()
            vmm = self.pool.vmm_snapshot()
            self.assertTrue(
                any(
                    segment["address"]
                    <= pointer
                    < segment["address"] + segment["total_size"]
                    for segment in native
                )
            )
            self.assertEqual(
                [entry["action"] for entry in vmm["trace"]],
                ["segment_alloc", "segment_map", "alloc"],
            )
            del value
            gc.collect()
            self.assertEqual(
                [entry["action"] for entry in self.pool.vmm_snapshot()["trace"]],
                ["segment_alloc", "segment_map", "alloc", "slot_cached"],
            )
        finally:
            self.pool.stop_recording()
            torch.cuda.memory._record_memory_history(enabled=None)


instantiate_device_type_tests(
    TestVMMHalfSplitAllocator,
    globals(),
    only_for="cuda",
)


if __name__ == "__main__":
    run_tests()
