#!/usr/bin/env python3
"""Plot and inspect SASS data from the CSV produced by parse_sass.py.

Usage:
    python tools/plot_sass_summary.py kernels.csv [--list-types] [--scatter out.png]
    python tools/plot_sass_summary.py kernels.csv --filter-source AbsKernel.cu --scatter abs.png

Reads the CSV (from parse_sass.py --csv), lists data types or generates scatter plots.
With --scatter, writes two plots: one for all rows, one for scalar_type float only.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
from sass_common import DTYPE_SIZES, KernelStats

STATS_KEYS = ("num_instructions", "max_gpr", "min_gpr", "max_pred", "min_pred", "max_ugpr", "min_ugpr")


def regs_for_elements(bits_per_element: int, n_elements: int) -> int:
    """Registers needed for n_elements at given bit size (register = 32 bits)."""
    return math.ceil((bits_per_element * n_elements) / 32)


def adjusted_gpr_range_from_type(gpr_range: int, scalar_type: str) -> float:
    """GPR range minus 4 elements worth of regs, plus 16 elements worth."""
    bits = DTYPE_SIZES.get(scalar_type, 32)
    return gpr_range - regs_for_elements(bits, 4) + regs_for_elements(bits, 16)


def _cc_from_columns(columns: list[str]) -> str | None:
    """First compute capability suffix found in column names (e.g. num_instructions_sm100 -> sm100)."""
    for col in columns:
        if col.startswith("num_instructions_") and col != "num_instructions_":
            return col[len("num_instructions_") :]
    return None


def parse_csv(path: str, cc: str | None = None) -> tuple[list[KernelStats], str]:
    """Load CSV and return (list of KernelStats for the given CC, cc_used)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows_list = list(reader)
        columns = reader.fieldnames or []

    cc_used = cc or _cc_from_columns(columns)
    if not cc_used:
        raise SystemExit("CSV has no per-CC stat columns (num_instructions_smXX, etc.).")

    result: list[KernelStats] = []
    for r in rows_list:
        try:
            num_instructions = int(r.get(f"num_instructions_{cc_used}", 0))
            max_gpr = int(r.get(f"max_gpr_{cc_used}", 0))
            min_gpr = int(r.get(f"min_gpr_{cc_used}", 0))
            max_pred = int(r.get(f"max_pred_{cc_used}", 0))
            min_pred = int(r.get(f"min_pred_{cc_used}", 0))
            max_ugpr = int(r.get(f"max_ugpr_{cc_used}", 0))
            min_ugpr = int(r.get(f"min_ugpr_{cc_used}", 0))
        except ValueError:
            continue
        result.append(
            KernelStats(
                mangled=r.get("mangled", ""),
                demangled=r.get("demangled", ""),
                tag=int(r.get("tag", 0)),
                functor=r.get("functor", ""),
                scalar_type=r.get("scalar_type", ""),
                source_file=r.get("source_file", ""),
                num_instructions=num_instructions,
                max_gpr=max_gpr,
                min_gpr=min_gpr,
                max_pred=max_pred,
                min_pred=min_pred,
                max_ugpr=max_ugpr,
                min_ugpr=min_ugpr,
            )
        )
    return result, cc_used


def list_data_types(rows: list[KernelStats]) -> None:
    types = sorted(set(r.scalar_type for r in rows))
    print(f"Data types in summary ({len(types)} unique):")
    print()
    for t in types:
        count = sum(1 for r in rows if r.scalar_type == t)
        print(f"  {t!r}: {count} kernel(s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot and list types from parse_sass.py CSV output"
    )
    parser.add_argument(
        "csv",
        help="Path to CSV file (output of parse_sass.py)",
    )
    parser.add_argument(
        "--cc",
        help="Compute capability to use (default: first found in CSV, e.g. sm100)",
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="Print all scalar data types that appear in the CSV",
    )
    parser.add_argument(
        "--scatter",
        metavar="PATH",
        help="Write scatter plots to PATH and PATH with float-only variant",
    )
    parser.add_argument(
        "--scatter-max-insns",
        type=int,
        default=50,
        help="Instruction count cutoff for scatter plots (default: 50)",
    )
    parser.add_argument(
        "--scatter-dtype",
        type=str,
        default="float",
        help="Data type for filtering scatter plots (default: float)",
    )
    parser.add_argument(
        "--filter-source",
        metavar="GLOB",
        help="Filter by source file (substring match, e.g. AbsKernel.cu)",
    )
    parser.add_argument(
        "--filter-line",
        type=int,
        help="Filter by line number",
    )
    parser.add_argument(
        "--max-insns",
        type=int,
        default=150,
        help="Instruction count cutoff for zoomed plots (default: 150)",
    )
    args = parser.parse_args()

    rows, cc_used = parse_csv(args.csv, args.cc)
    if not rows:
        print("No kernel rows found in CSV.", file=sys.stderr)
        sys.exit(1)

    if args.filter_source:
        rows = [r for r in rows if args.filter_source in r.source_file]
    if args.filter_line is not None:
        rows = [r for r in rows if r.line == args.filter_line]

    if args.list_types or not args.scatter:
        if args.filter_source or args.filter_line is not None:
            print(f"Filtered to {len(rows)} kernel(s).")
        print(f"Using compute capability: {cc_used}")
        list_data_types(rows)
        if not args.scatter:
            return

    if args.scatter:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for --scatter", file=sys.stderr)
            sys.exit(1)

        def do_scatter(data: list[KernelStats], title: str, path: str) -> None:
            subset = [r for r in data if r.num_instructions <= args.scatter_max_insns]
            x = [r.num_instructions for r in subset]
            y = [adjusted_gpr_range_from_type(r.gpr_range, r.scalar_type) for r in subset]
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.scatter(x, y, alpha=0.4, s=12, edgecolors="none")
            ax.set_xlabel("Instruction count")
            ax.set_ylabel("Adjusted GPR range (range − 4 elem regs + 16 elem regs)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"Scatter plot saved to {path}", file=sys.stderr)

        out_path = args.scatter
        do_scatter(rows, f"GPU kernel instantiations ({len(rows)} kernels)", out_path)

        stem = Path(out_path).stem
        suffix = Path(out_path).suffix
        parent = Path(out_path).parent
        dtype_path = str(parent / f"{stem}_{args.scatter_dtype}{suffix}")
        dtype_rows = [r for r in rows if r.scalar_type == args.scatter_dtype]
        do_scatter(
            dtype_rows,
            f"GPU kernel instantiations — {args.scatter_dtype} only ({len(dtype_rows)} kernels)",
            dtype_path,
        )


if __name__ == "__main__":
    main()
