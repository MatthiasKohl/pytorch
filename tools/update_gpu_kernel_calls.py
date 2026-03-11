#!/usr/bin/env python3
"""Update gpu_kernel and gpu_kernel_nocast call-sites with is_simple_func from SASS CSV.

Usage:
    python tools/update_gpu_kernel_calls.py kernels.csv [--aten-dir aten/src/ATen] [--dry-run]

Reads the CSV (from parse_sass.py), computes is_simple = (num_instructions <= 29
and adjusted_gpr_range <= 30) per (source_file, line, scalar_type). For each (source_file, line)
we set is_simple_func = true only if all scalar types at that line are simple (conservative:
if any type is not simple, we use false).

Finds all .cu files under aten that contain gpu_kernel(iter, or gpu_kernel_nocast(iter,
(without existing template args) and replaces them with gpu_kernel<true>(iter, or
gpu_kernel<false>(iter, (and gpu_kernel_nocast<...>) based on the CSV.
"""

import argparse
import math
import re
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
from plot_sass_summary import (  # noqa: E402
    DTYPE_SIZES,
    parse_csv,
)
from sass_common import KernelStats


def regs_for_elements(bits_per_element: int, n_elements: int) -> int:
    return math.ceil((bits_per_element * n_elements) / 32)


def adjusted_gpr_range(gpr_range: int, scalar_type: str) -> float:
    bits = DTYPE_SIZES.get(scalar_type, 32)
    return gpr_range - regs_for_elements(bits, 4) + regs_for_elements(bits, 16)


def is_simple(row: KernelStats) -> bool:
    adj = adjusted_gpr_range(row.gpr_range, row.scalar_type)
    return row.num_instructions <= 29 and adj <= 30


def build_line_lookup(rows: list[KernelStats]) -> dict[tuple[str, int], bool]:
    """(source_file, line) -> is_simple_func (true only if all types at that line are simple)."""
    by_key: dict[tuple[str, int], list[bool]] = {}
    for r in rows:
        key = (r.source_file, r.line)
        by_key.setdefault(key, []).append(is_simple(r))
    return {k: all(v) for k, v in by_key.items()}


def find_cu_files(aten_dir: Path) -> list[Path]:
    return list(aten_dir.rglob("*.cu"))


# Match gpu_kernel(iter, or gpu_kernel_nocast(iter, without a <...> before the (
RE_GPU_KERNEL = re.compile(r"\bgpu_kernel\s*\(\s*iter\s*,")
RE_GPU_KERNEL_NOCAST = re.compile(r"\bgpu_kernel_nocast\s*\(\s*iter\s*,")


def update_file(path: Path, lookup: dict[tuple[str, int], bool], dry_run: bool) -> int:
    """Replace gpu_kernel(iter, and gpu_kernel_nocast(iter, with templated form. Returns count of replacements."""
    name = path.name
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    replacements = 0
    for i, line in enumerate(lines):
        line_no = i + 1
        key = (name, line_no)
        is_simple_val = lookup.get(key, False)
        val_str = "true" if is_simple_val else "false"
        if RE_GPU_KERNEL.search(line) and "gpu_kernel<" not in line[: line.find("gpu_kernel") + 11]:
            new_line = RE_GPU_KERNEL.sub(f"gpu_kernel<{val_str}>(iter,", line)
            if new_line != line:
                lines[i] = new_line
                replacements += 1
        if RE_GPU_KERNEL_NOCAST.search(line) and "gpu_kernel_nocast<" not in line[: line.find("gpu_kernel_nocast") + 19]:
            new_line = RE_GPU_KERNEL_NOCAST.sub(f"gpu_kernel_nocast<{val_str}>(iter,", line)
            if new_line != line:
                lines[i] = new_line
                replacements += 1
    if replacements and not dry_run:
        path.write_text("".join(lines), encoding="utf-8")
    return replacements


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update gpu_kernel/gpu_kernel_nocast call-sites from SASS summary"
    )
    parser.add_argument("csv", help="Path to CSV file (output of parse_sass.py)")
    parser.add_argument(
        "--aten-dir",
        default="aten/src/ATen",
        help="Root directory containing .cu files (default: aten/src/ATen)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be changed, do not write")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = repo_root / csv_path
    aten_dir = Path(args.aten_dir)
    if not aten_dir.is_absolute():
        aten_dir = repo_root / aten_dir

    rows, _ = parse_csv(str(csv_path))
    if not rows:
        raise SystemExit("No kernel rows found in CSV.")

    lookup = build_line_lookup(rows)
    cu_files = find_cu_files(aten_dir)
    total = 0
    for path in sorted(cu_files):
        n = update_file(path, lookup, args.dry_run)
        if n:
            total += n
            rel = path.relative_to(repo_root)
            print(f"{'Would update' if args.dry_run else 'Updated'} {rel}: {n} replacement(s)")
    if total:
        print(f"\nTotal: {total} replacement(s)" + (" (dry run)" if args.dry_run else ""))
    else:
        print("No call-sites needed updating.")


if __name__ == "__main__":
    main()
