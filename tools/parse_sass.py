#!/usr/bin/env python3
"""Analyse simple_gpu_kernel instantiations in a CUDA static library.

Usage:
    python tools/parse_sass.py libsimple_gpu_kernels.a [-o summary.txt] [--csv out.csv] [--scatter scatter.png]

The script:
  1. Runs  cuobjdump -lelf <lib>           to list embedded cubins.
  2. Runs  cuobjdump -xelf <cubin> <lib>   to extract each cubin.
  3. Runs  nvdisasm -plr -lrm=count <cubin> and parses stdout to
     extract per-function instruction counts and peak register usage
     (GPR, PRED, UGPR) from the live-range columns.

Always writes a CSV (default: stem of -o or input + .csv) with identity columns
(mangled, demangled, tag, functor, scalar_type, source_file) and per-compute-capability
stats (num_instructions_smXX, max_gpr_smXX, etc.). Compute capability is inferred
from the library filename (e.g. libXXX_sm100.a -> sm100, libXXX_sm100a.a -> sm100a).
If the CSV already exists, new results are merged by demangled name so that the same
kernel from different libraries (e.g. sm75 vs sm80) maps to one row. The demangled name
is stable across CCs and distinguishes gpu_kernel_with_scalars variants.
"""

import argparse
import csv
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
from sass_common import DTYPE_SIZES, KernelStats

# Instruction line:  /*0000*/  OPCODE ... ;  // | GPR | PRED | UGPR |
RE_INSTRUCTION = re.compile(r"^\s+/\*([0-9a-f]+)\*/\s+(\S+)")
# Table header row with column names, e.g.  // | GPR   | PRED   | UGPR   |
RE_TABLE_HEADER = re.compile(r"//\s*\|([\w\s|]+)\|")
# Extract all pipe-separated values from the trailing comment
RE_PLR_VALUES = re.compile(r"//\s*\|([^/]+)$")
# Self-branch (nvdisasm uses label syntax)
RE_BRA = re.compile(r"^\s+/\*[0-9a-f]+\*/\s+BRA\b")
# Section header: //----- .text.MANGLED_NAME -----
RE_SECTION = re.compile(
    r"^//---+\s+\.text\.(_Z\S+)\s+---+"
)

# ---------------------------------------------------------------------------
# Demangling
# ---------------------------------------------------------------------------

def demangle_batch(mangled_names: list[str]) -> dict[str, str]:
    """Demangle a batch of C++ symbols using c++filt."""
    if not mangled_names:
        return {}
    proc = subprocess.run(
        ["c++filt"],
        input="\n".join(mangled_names),
        capture_output=True,
        text=True,
    )
    demangled = proc.stdout.strip().split("\n")
    return dict(zip(mangled_names, demangled))


# ---------------------------------------------------------------------------
# Demangled-name helpers
# ---------------------------------------------------------------------------

def parse_demangled(demangled: str) -> tuple[int, str, str] | None:
    """Extract (tag, functor, scalar_type) from a demangled simple_gpu_kernel name.

    Handles both signature forms:
      Unrolled: void at::native::simple_gpu_kernel<TAG, FUNCTOR, std::array<...>>(...)
      Scalar:   function_traits<FUNCTOR>::result_type at::native::simple_gpu_kernel<TAG, FUNCTOR, ARGS...>(...)
    """
    # Find the template args of simple_gpu_kernel<...>
    idx = demangled.find("at::native::simple_gpu_kernel<")
    if idx < 0:
        return None
    open_pos = idx + len("at::native::simple_gpu_kernel<") - 1
    close_pos = _find_matching_close(demangled, open_pos)
    if close_pos is None:
        return None
    inner = demangled[open_pos + 1 : close_pos]

    # Split on top-level commas: first arg is TAG, second is FUNCTOR,
    # remaining are either std::array (unrolled) or scalar arg types.
    args = _extract_template_args(demangled, open_pos)
    if len(args) < 2:
        return None
    try:
        tag = int(args[0].strip())
    except ValueError:
        return None
    functor_full = args[1].strip()
    scalar_type = _extract_scalar_type(functor_full)

    # For the scalar path, the template args after the functor are the
    # argument types themselves.  Use the first one if _extract_scalar_type
    # couldn't determine the type from the functor name alone.
    if scalar_type == "unknown" and len(args) > 2:
        candidate = args[2].strip()
        if candidate and not candidate.startswith("std::array"):
            scalar_type = candidate

    return tag, functor_full, scalar_type


def _find_matching_close(s: str, open_pos: int) -> int | None:
    depth = 0
    for i in range(open_pos, len(s)):
        if s[i] == "<":
            depth += 1
        elif s[i] == ">":
            depth -= 1
            if depth == 0:
                return i
    return None


def _extract_template_args(s: str, open_pos: int) -> list[str]:
    close_pos = _find_matching_close(s, open_pos)
    if close_pos is None:
        return []
    inner = s[open_pos + 1 : close_pos]
    args: list[str] = []
    angle_depth = 0
    paren_depth = 0
    start = 0
    for i, ch in enumerate(inner):
        if ch == "<":
            angle_depth += 1
        elif ch == ">":
            angle_depth -= 1
        elif ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
        elif ch == "," and angle_depth == 0 and paren_depth == 0:
            args.append(inner[start:i].strip())
            start = i + 1
    args.append(inner[start:].strip())
    return args


def _extract_scalar_type(functor_full: str) -> str:
    wrapper_match = re.match(
        r"at::native::(?:AUnaryFunctor|BUnaryFunctor|BinaryFunctor)\s*<",
        functor_full,
    )
    if wrapper_match:
        args = _extract_template_args(functor_full, wrapper_match.end() - 1)
        if args:
            return args[0]
    lambda_match = re.search(r"\{lambda\(([^,)]+)", functor_full)
    if lambda_match:
        return lambda_match.group(1).strip()
    last_open = None
    depth = 0
    for i in range(len(functor_full) - 1, -1, -1):
        if functor_full[i] == ">":
            if depth == 0:
                pass
            depth += 1
        elif functor_full[i] == "<":
            depth -= 1
            if depth == 0:
                last_open = i
                break
    if last_open is not None:
        args = _extract_template_args(functor_full, last_open)
        if args:
            return args[0]
    return "unknown"


# ---------------------------------------------------------------------------
# Instruction stripping helpers
# ---------------------------------------------------------------------------

def _opcode(line: str) -> str:
    m = RE_INSTRUCTION.match(line)
    return m.group(2).rstrip(";") if m else ""


def _strip_trailing_padding(insn_lines: list[str]) -> list[str]:
    """Strip trailing BRA (self-branch padding)."""
    # In nvdisasm output there are no NOPs, but BRA `(.L_x_N) appears last.
    while insn_lines and RE_BRA.match(insn_lines[-1]):
        insn_lines.pop()

    # note: we don't remove the last RET to get PLR values from it,
    # but we implicitly remove one instruction from the count

    return insn_lines


# ---------------------------------------------------------------------------
# Parsing nvdisasm -plr -lrm=count output
# ---------------------------------------------------------------------------

def _parse_table_header(lines: list[str]) -> list[str]:
    """Find the column names from the nvdisasm -plr table header.

    Looks for a line like:  // | GPR   | PRED   | UGPR   |
    Returns e.g. ["GPR", "PRED", "UGPR"] or ["GPR", "UGPR"].
    """
    for line in lines:
        m = RE_TABLE_HEADER.search(line)
        if m:
            cols = [c.strip() for c in m.group(1).split("|") if c.strip()]
            if any(c in ("GPR", "PRED", "UGPR") for c in cols):
                return cols
    return []


def _parse_plr_values(line: str, ncols: int) -> list[int]:
    """Extract register count values from the trailing // | ... | comment."""
    m = RE_PLR_VALUES.search(line)
    if not m:
        return [0] * ncols
    cells = m.group(1).split("|")
    result: list[int] = []
    for cell in cells[:ncols]:
        cell = cell.strip()
        if cell and cell.isdigit():
            result.append(int(cell))
        else:
            result.append(0)
    while len(result) < ncols:
        result.append(0)
    return result


def _parse_nvdisasm_output(text: str, member: str) -> list[KernelStats]:
    """Parse the output of  nvdisasm -plr -lrm=count  for one cubin."""
    sections: list[tuple[str, list[str]]] = []
    current_mangled: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        sm = RE_SECTION.match(line)
        if sm:
            if current_mangled and "simple_gpu_kernel" in current_mangled:
                sections.append((current_mangled, current_lines))
            current_mangled = sm.group(1)
            current_lines = []
            continue
        if current_mangled:
            current_lines.append(line)

    if current_mangled and "simple_gpu_kernel" in current_mangled:
        sections.append((current_mangled, current_lines))

    if not sections:
        return []

    mangled_names = list(set(s[0] for s in sections))
    demangled_map = demangle_batch(mangled_names)

    source_file = member.replace(".cu.o", ".cu") if member else "unknown"

    results: list[KernelStats] = []
    for mangled, lines in sections:
        demangled = demangled_map.get(mangled, mangled)
        parsed = parse_demangled(demangled)
        if parsed is None:
            continue
        tag, functor, scalar_type = parsed

        col_names = _parse_table_header(lines)
        ncols = len(col_names)
        col_idx = {name: i for i, name in enumerate(col_names)}

        insn_lines = [l for l in lines if RE_INSTRUCTION.match(l)]
        insn_lines = _strip_trailing_padding(insn_lines)

        gprs: list[int] = []
        preds: list[int] = []
        ugprs: list[int] = []

        for line in insn_lines:
            vals = _parse_plr_values(line, ncols)
            if "GPR" in col_idx:
                v = vals[col_idx["GPR"]]
                gprs.append(v)
            if "PRED" in col_idx:
                v = vals[col_idx["PRED"]]
                preds.append(v)
            if "UGPR" in col_idx:
                v = vals[col_idx["UGPR"]]
                ugprs.append(v)

        ks = KernelStats(
            mangled=mangled,
            demangled=demangled,
            tag=tag,
            functor=functor,
            scalar_type=scalar_type,
            source_file=source_file,
            # remove the last RET instruction from the count
            num_instructions=len(insn_lines) - 1,
            max_gpr=max(gprs) if gprs else 0,
            min_gpr=min(gprs) if gprs else 0,
            max_pred=max(preds) if preds else 0,
            min_pred=min(preds) if preds else 0,
            max_ugpr=max(ugprs) if ugprs else 0,
            min_ugpr=min(ugprs) if ugprs else 0,
        )
        results.append(ks)

    return results


def _nvdisasm_and_parse(args: tuple[str, str]) -> tuple[list[KernelStats], str | None]:
    """Run nvdisasm on a cubin and parse output. Returns (stats_list, error_msg or None)."""
    cubin_path, member = args
    cubin_name = os.path.basename(cubin_path)
    proc = subprocess.run(
        ["nvdisasm", "-plr", "-lrm=count", cubin_path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return ([], f"nvdisasm failed for {cubin_name}: {proc.stderr.strip()}")
    return (_parse_nvdisasm_output(proc.stdout, member), None)


# ---------------------------------------------------------------------------
# Top-level library parsing
# ---------------------------------------------------------------------------

def parse_library(lib_path: str, max_cubins: int | None = None) -> list[KernelStats]:
    """Extract all simple_gpu_kernel stats from a static library.

    Pipeline:  cuobjdump -lelf  ->  cuobjdump -xelf (all)  ->  pool: nvdisasm + parse
    """
    lib_path = os.path.abspath(lib_path)

    # Step 1: list embedded cubins and their source members
    proc = subprocess.run(
        ["cuobjdump", "-lelf", lib_path],
        capture_output=True, text=True, check=True,
    )
    re_member = re.compile(r"^member\s+.*?:(\S+\.cu\.o):")
    re_elf = re.compile(r"ELF file\s+\d+:\s+(\S+)")
    cubins: list[tuple[str, str]] = []  # (cubin_name, member)
    current_member = ""
    for line in proc.stdout.splitlines():
        mm = re_member.match(line)
        if mm:
            current_member = mm.group(1)
            continue
        em = re_elf.match(line.strip())
        if em:
            cubins.append((em.group(1), current_member))
    if max_cubins is not None:
        cubins = cubins[:max_cubins]
        print(f"Limiting to first {len(cubins)} cubin(s) (--max-cubins)", file=sys.stderr)
    print(f"Found {len(cubins)} cubin(s) in {lib_path}", file=sys.stderr)

    # Step 2: extract each cubin into its own subdir to avoid overwrites
    with tempfile.TemporaryDirectory(prefix="parse_sass_") as tmpdir:
        tasks: list[tuple[str, str]] = []  # (cubin_path, member)
        for i, (cubin_name, member) in enumerate(cubins):
            subdir = os.path.join(tmpdir, str(i))
            os.makedirs(subdir, exist_ok=True)
            subprocess.run(
                ["cuobjdump", "-xelf", cubin_name, lib_path],
                capture_output=True, text=True, check=True,
                cwd=subdir,
            )
            cubin_path = os.path.join(subdir, cubin_name)
            if not os.path.isfile(cubin_path):
                candidates = list(Path(subdir).glob(f"*{cubin_name}*"))
                if not candidates:
                    candidates = list(Path(subdir).glob(f"{member}*"))
                if candidates:
                    cubin_path = str(candidates[0])
                else:
                    print(
                        f"  Warning: could not find extracted cubin for {cubin_name}",
                        file=sys.stderr,
                    )
                    continue
            tasks.append((cubin_path, member))

        # Step 3: run nvdisasm + parse (sequential if max_cubins <= 4 for sandbox-friendly testing)
        print(f"Running nvdisasm on {len(tasks)} cubin(s)...", file=sys.stderr)
        all_stats: list[KernelStats] = []
        use_pool = max_cubins is None or max_cubins > 4
        if use_pool:
            with mp.Pool() as pool:
                for stats, err in pool.map(_nvdisasm_and_parse, tasks):
                    if err:
                        print(f"  Warning: {err}", file=sys.stderr)
                    else:
                        all_stats.extend(stats)
        else:
            for stats, err in [_nvdisasm_and_parse(t) for t in tasks]:
                if err:
                    print(f"  Warning: {err}", file=sys.stderr)
                else:
                    all_stats.extend(stats)
    return all_stats


# ---------------------------------------------------------------------------
# Legacy SASS file parsing (kept for backward compatibility)
# ---------------------------------------------------------------------------

def parse_sass_file(path: str) -> list[KernelStats]:
    """Parse a pre-generated SASS file (cuobjdump -sass output)."""
    sections: list[tuple[str, str, list[str]]] = []
    current_member = ""
    current_mangled: str | None = None
    current_lines: list[str] = []
    func_member = ""

    re_function = re.compile(r"^\s+Function\s*:\s*(\S+)")
    re_member = re.compile(r"^member\s+.*?:(\S+\.cu\.o):")

    # Patterns for old cuobjdump -sass format
    re_sass_insn = re.compile(r"^\s+/\*([0-9a-f]+)\*/\s+(\S+)")
    re_self_bra_sass = re.compile(r"^\s+/\*([0-9a-f]+)\*/\s+BRA\s+0x([0-9a-f]+)\s*;")
    re_reg = re.compile(r" R(\d{1,3})[,;\]\s\.]")
    re_ureg = re.compile(r" UR(\d{1,3})[,;\]\s\.]")
    re_pred = re.compile(r" P(\d)[,;\]\s\.]")
    re_upred = re.compile(r" UP(\d)[,;\]\s\.]")

    with open(path, "r") as f:
        for line in f:
            mm = re_member.match(line)
            if mm:
                current_member = mm.group(1)
                continue
            fm = re_function.match(line)
            if fm:
                if current_mangled and "simple_gpu_kernel" in current_mangled:
                    sections.append((func_member, current_mangled, current_lines))
                current_mangled = fm.group(1)
                func_member = current_member
                current_lines = []
                continue
            if current_mangled:
                current_lines.append(line)

        if current_mangled and "simple_gpu_kernel" in current_mangled:
            sections.append((func_member, current_mangled, current_lines))

    mangled_names = list(set(s[1] for s in sections))
    demangled_map = demangle_batch(mangled_names)

    results = []
    for member, mangled, lines in sections:
        demangled = demangled_map.get(mangled, mangled)
        parsed = parse_demangled(demangled)
        if parsed is None:
            continue
        tag, functor, scalar_type = parsed
        source_file = member.replace(".cu.o", ".cu") if member else "unknown"

        insn_lines = [l for l in lines if re_sass_insn.match(l)]

        # Strip trailing NOPs
        while insn_lines:
            m = re_sass_insn.match(insn_lines[-1])
            if m and m.group(2).rstrip(";") == "NOP":
                insn_lines.pop()
            else:
                break
        # Strip self-BRA
        if insn_lines:
            bm = re_self_bra_sass.match(insn_lines[-1])
            if bm and int(bm.group(1), 16) == int(bm.group(2), 16):
                insn_lines.pop()
        # Strip trailing EXIT / RET
        if insn_lines:
            m = re_sass_insn.match(insn_lines[-1])
            if m:
                op = m.group(2).rstrip(";")
                if op == "EXIT" or op.split(".")[0] == "RET":
                    insn_lines.pop()

        regs: set[int] = set()
        uregs: set[int] = set()
        preds_set: set[int] = set()
        upreds_set: set[int] = set()
        for line in insn_lines:
            for m in re_reg.finditer(line):
                regs.add(int(m.group(1)))
            for m in re_ureg.finditer(line):
                uregs.add(int(m.group(1)))
            for m in re_pred.finditer(line):
                preds_set.add(int(m.group(1)))
            for m in re_upred.finditer(line):
                upreds_set.add(int(m.group(1)))

        ks = KernelStats(
            mangled=mangled,
            demangled=demangled,
            tag=tag,
            functor=functor,
            scalar_type=scalar_type,
            source_file=source_file,
            num_instructions=len(insn_lines),
            max_gpr=len(regs),
            min_gpr=len(regs),
            max_pred=len(preds_set),
            min_pred=len(preds_set),
            max_ugpr=len(uregs),
            min_ugpr=len(uregs),
        )
        results.append(ks)

    return results


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------

def format_summary(kernels: list[KernelStats]) -> str:
    by_file: dict[str, dict[int, list[KernelStats]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for k in kernels:
        by_file[k.source_file][k.tag].append(k)

    lines = []
    lines.append("GPU Kernel SASS Summary")
    lines.append("=======================")
    lines.append(f"Total simple_gpu_kernel instantiations: {len(kernels)}")
    lines.append(f"Source files: {len(by_file)}")
    lines.append("")

    for source_file in sorted(by_file.keys()):
        tags = by_file[source_file]
        lines.append(f"=== {source_file} ===")

        for tag in sorted(tags.keys()):
            kstats = sorted(tags[tag], key=lambda k: k.scalar_type)
            short_functor = kstats[0].functor.replace("at::native::", "")
            lines.append(f"  Line {tag}: {short_functor}")

            for k in kstats:
                gpr_range = k.max_gpr - k.min_gpr
                ugpr_range = k.max_ugpr - k.min_ugpr
                pred_range = k.max_pred - k.min_pred
                lines.append(
                    f"    {k.scalar_type:>20s}: "
                    f"{k.num_instructions:5d} insns, "
                    f"{k.max_gpr:3d} GPR (range {gpr_range}), "
                    f"{k.max_ugpr:3d} UGPR (range {ugpr_range}), "
                    f"{k.max_pred:2d} PRED (range {pred_range})"
                )

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compute capability and CSV
# ---------------------------------------------------------------------------

RE_SM = re.compile(r"_sm(\d+)([a-z])?$", re.IGNORECASE)
RE_CC = re.compile(r"^sm(\d+)([a-z])?$", re.IGNORECASE)

def _cc_sort_key(cc: str) -> tuple[int, str]:
    """Sort key for compute capability: 75, 80, 86, ..., 100, 100a."""
    m = RE_CC.match(cc)
    if m:
        return (int(m.group(1)), (m.group(2) or "").lower())
    return (0, cc)

def infer_compute_capability(lib_path: str) -> str | None:
    """Infer compute capability from library filename, e.g. libXXX_sm100.a -> sm100."""
    stem = Path(lib_path).stem
    m = RE_SM.search(stem)
    if not m:
        return None
    base = f"sm{m.group(1)}"
    suffix = m.group(2)
    return base + (suffix.lower() if suffix else "")

IDENTITY_KEYS = ("mangled", "demangled", "tag", "functor", "scalar_type", "source_file")
STATS_KEYS = ("num_instructions", "max_gpr", "min_gpr", "max_pred", "min_pred", "max_ugpr", "min_ugpr")

# NVCC encodes internal linkage as _INTERNAL_<h1>_<n1>_<file>_cu_<h2>_<n2>
# where <h2>_<n2> varies per compute capability. Strip that suffix for matching.
RE_INTERNAL = re.compile(r"(_INTERNAL_[0-9a-f]+_\d+_\w+_cu)_[0-9a-f]+_\d+")

def _normalize_demangled(demangled: str) -> str:
    """Strip CC-varying hash suffixes from NVCC _INTERNAL_ prefixes."""
    return RE_INTERNAL.sub(r"\1", demangled)

def _merge_key(row: dict) -> str:
    """Unique key for a kernel across architectures.

    The demangled name encodes everything (tag, functor, array size, arg types)
    and is stable across CCs for most symbols (c++filt strips anonymous-namespace
    hashes). For NVCC _INTERNAL_ symbols, we additionally strip the CC-varying
    hash suffix."""
    return _normalize_demangled(row.get("demangled", ""))

def _regs_for_elements(bits_per_element: int, n_elements: int) -> int:
    return math.ceil((bits_per_element * n_elements) / 32)

def _adjusted_gpr_range(gpr_range: int, scalar_type: str) -> float:
    bits = DTYPE_SIZES.get(scalar_type, 32)
    return gpr_range - _regs_for_elements(bits, 4) + _regs_for_elements(bits, 16)

def is_simple(k: KernelStats) -> bool:
    adj = _adjusted_gpr_range(k.gpr_range, k.scalar_type)
    return k.num_instructions <= 29 and adj <= 30

def _stats_to_row(k: KernelStats, cc: str) -> dict[str, str | int]:
    row: dict[str, str | int] = {
        "mangled": k.mangled,
        "demangled": k.demangled,
        "tag": k.tag,
        "functor": k.functor,
        "scalar_type": k.scalar_type,
        "source_file": k.source_file,
    }
    for key in STATS_KEYS:
        row[f"{key}_{cc}"] = getattr(k, key)
    row[f"is_simple_{cc}"] = "true" if is_simple(k) else "false"
    return row

def _row_to_stats(r: dict, cc: str) -> KernelStats | None:
    try:
        return KernelStats(
            mangled=r["mangled"],
            demangled=r["demangled"],
            tag=int(r["tag"]),
            functor=r["functor"],
            scalar_type=r["scalar_type"],
            source_file=r["source_file"],
            num_instructions=int(r.get(f"num_instructions_{cc}", 0)),
            max_gpr=int(r.get(f"max_gpr_{cc}", 0)),
            min_gpr=int(r.get(f"min_gpr_{cc}", 0)),
            max_pred=int(r.get(f"max_pred_{cc}", 0)),
            min_pred=int(r.get(f"min_pred_{cc}", 0)),
            max_ugpr=int(r.get(f"max_ugpr_{cc}", 0)),
            min_ugpr=int(r.get(f"min_ugpr_{cc}", 0)),
        )
    except (KeyError, ValueError):
        return None

def load_csv(path: str) -> tuple[list[dict], set[str]]:
    """Load CSV; return (list of row dicts, set of CC suffixes found)."""
    if not os.path.isfile(path):
        return [], set()
    ccs: set[str] = set()
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
            for key in r:
                for sk in STATS_KEYS:
                    if key.startswith(f"{sk}_") and key != sk:
                        ccs.add(key[len(sk) + 1 :])
    return rows, ccs

def write_csv(path: str, rows: list[dict], all_columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=all_columns, extrasaction="ignore", restval=""
        )
        w.writeheader()
        w.writerows(rows)

def merge_and_write_csv(
    csv_path: str,
    new_kernels: list[KernelStats],
    cc: str,
) -> None:
    """Merge new_kernels (for this cc) into csv_path; match rows by demangled name.

    The demangled name is stable across CCs (anonymous-namespace hashes are stripped)
    and distinguishes gpu_kernel_with_scalars variants (different std::array sizes).
    When existing rows have duplicate demangled names (from a prior buggy merge),
    they are consolidated. A note is printed when new-kernel duplicates occur."""
    new_rows = [_stats_to_row(k, cc) for k in new_kernels]
    existing_rows, existing_ccs = load_csv(csv_path)

    by_key: dict[str, dict] = {}
    for r in existing_rows:
        k = _merge_key(r)
        if k not in by_key:
            by_key[k] = dict(r)
        else:
            ex = by_key[k]
            for col, val in r.items():
                if col not in IDENTITY_KEYS and val and (not ex.get(col)):
                    ex[col] = val

    n_new = len(new_rows)
    n_duplicates = 0
    for nr in new_rows:
        k = _merge_key(nr)
        if k in by_key:
            ex = by_key[k]
            for key, val in nr.items():
                if key not in IDENTITY_KEYS:
                    ex[key] = val
            n_duplicates += 1
        else:
            for c in existing_ccs:
                for sk in STATS_KEYS:
                    nr[f"{sk}_{c}"] = ""
                nr[f"is_simple_{c}"] = ""
            by_key[k] = nr

    if n_duplicates:
        print(
            f"Note: {n_new} kernel instantiations had {n_duplicates} duplicate demangled name(s); "
            f"{n_new - n_duplicates} unique rows for this CC (last occurrence kept).",
            file=sys.stderr,
        )

    all_ccs = sorted(existing_ccs | {cc}, key=_cc_sort_key)
    all_columns = (
        list(IDENTITY_KEYS)
        + [f"{sk}_{c}" for c in all_ccs for sk in STATS_KEYS]
        + [f"is_simple_{c}" for c in all_ccs]
    )
    write_csv(csv_path, list(by_key.values()), all_columns)
    print(f"CSV written to {csv_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def adjusted_reg_count(k: KernelStats) -> float:
    return k.max_gpr / 255 + k.max_ugpr / 80 + k.max_pred / 14


def adjusted_reg_count_no_preds(k: KernelStats) -> float:
    return k.max_gpr / 255 + k.max_ugpr / 80


def plot_scatter(kernels: list[KernelStats], output_path: str,
                 max_insns: int | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base, ext = os.path.splitext(output_path)

    # Full scatter
    x = [k.num_instructions for k in kernels]
    y = [adjusted_reg_count(k) for k in kernels]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(x, y, alpha=0.4, s=12, edgecolors="none")
    ax.set_xlabel("Instruction count")
    ax.set_ylabel("Adjusted register count  (GPR/255 + UGPR/80 + PRED/14)")
    ax.set_title(f"GPU kernel instantiations ({len(kernels)} kernels)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Scatter plot saved to {output_path}", file=sys.stderr)

    if max_insns is not None:
        filtered = [k for k in kernels if k.num_instructions <= max_insns]
        fx = [k.num_instructions for k in filtered]
        n = len(filtered)

        fy = [adjusted_reg_count(k) for k in filtered]
        path1 = f"{base}_zoom{ext}"
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(fx, fy, alpha=0.4, s=18, edgecolors="none")
        ax.set_xlabel("Instruction count")
        ax.set_ylabel("Adjusted register count  (GPR/255 + UGPR/80 + PRED/14)")
        ax.set_title(
            f"GPU kernel instantiations — insns ≤ {max_insns} ({n}/{len(kernels)} kernels)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path1, dpi=150)
        plt.close(fig)
        print(f"Zoomed scatter (with preds) saved to {path1}", file=sys.stderr)

        fy_np = [adjusted_reg_count_no_preds(k) for k in filtered]
        path2 = f"{base}_zoom_nopred{ext}"
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(fx, fy_np, alpha=0.4, s=18, edgecolors="none")
        ax.set_xlabel("Instruction count")
        ax.set_ylabel("Adjusted register count  (GPR/255 + UGPR/80)")
        ax.set_title(
            f"GPU kernel instantiations — insns ≤ {max_insns}, no predicates ({n}/{len(kernels)} kernels)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path2, dpi=150)
        plt.close(fig)
        print(f"Zoomed scatter (no preds) saved to {path2}", file=sys.stderr)

        fy_gpr = [k.max_gpr for k in filtered]
        path3 = f"{base}_zoom_nopred_nouniform{ext}"
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(fx, fy_gpr, alpha=0.4, s=18, edgecolors="none")
        ax.set_xlabel("Instruction count")
        ax.set_ylabel("GPR count")
        ax.set_title(
            f"GPU kernel instantiations — insns ≤ {max_insns}, GPR only ({n}/{len(kernels)} kernels)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path3, dpi=150)
        plt.close(fig)
        print(f"Zoomed scatter (GPR only) saved to {path3}", file=sys.stderr)

        fy_gpr_range = [k.max_gpr - k.min_gpr for k in filtered]
        path4 = f"{base}_zoom_nopred_nouniform_range{ext}"
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(fx, fy_gpr_range, alpha=0.4, s=18, edgecolors="none")
        ax.set_xlabel("Instruction count")
        ax.set_ylabel("GPR range")
        ax.set_title(
            f"GPU kernel instantiations — insns ≤ {max_insns}, GPR only ({n}/{len(kernels)} kernels)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path4, dpi=150)
        plt.close(fig)
        print(f"Zoomed scatter (GPR range) saved to {path4}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyse simple_gpu_kernel instantiations in a CUDA library"
    )
    parser.add_argument("input",
        help="Path to the static library (.a) or a pre-generated .sass file")
    parser.add_argument("-o", "--output", default=None,
        help="Output summary file (default: stdout)")
    parser.add_argument("--csv", default=None,
        help="Output CSV path (default: derived from -o or input stem + .csv)")
    parser.add_argument("--scatter", default=None,
        help="Output path for scatter plot (e.g. scatter.png)")
    parser.add_argument("--max-insns", type=int, default=150,
        help="Instruction count cutoff for zoomed plots (default: 150)")
    parser.add_argument("--max-cubins", type=int, default=None, metavar="N",
        help="Limit to first N cubins (for faster testing)")
    args = parser.parse_args()

    is_sass = args.input.endswith(".sass")

    if is_sass:
        print(f"Parsing SASS file {args.input}...", file=sys.stderr)
        kernels = parse_sass_file(args.input)
        cc = infer_compute_capability(args.input) or "unknown"
    else:
        print(f"Analysing library {args.input}...", file=sys.stderr)
        kernels = parse_library(args.input, max_cubins=args.max_cubins)
        cc = infer_compute_capability(args.input) or "unknown"

    print(f"Found {len(kernels)} simple_gpu_kernel instantiations", file=sys.stderr)
    print(f"Compute capability: {cc}", file=sys.stderr)

    summary = format_summary(kernels)

    if args.output:
        with open(args.output, "w") as f:
            f.write(summary)
        print(f"Summary written to {args.output}", file=sys.stderr)

    csv_path = args.csv
    if csv_path is None:
        if args.output:
            csv_path = str(Path(args.output).with_suffix(".csv"))
        else:
            csv_path = str(Path(args.input).with_suffix(".csv"))
    merge_and_write_csv(csv_path, kernels, cc)

    if not args.output:
        print(summary)

    if args.scatter:
        plot_scatter(kernels, args.scatter, max_insns=args.max_insns)


if __name__ == "__main__":
    main()
