#!/usr/bin/env python3
"""
Extract all instantiations of gpu_kernel_impl / gpu_kernel_impl_nocast from
aten/src/ATen/native/cuda/ source files.

For each call to gpu_kernel / gpu_kernel_nocast / gpu_kernel_with_scalars /
opmath_gpu_kernel_with_scalars / opmath_symmetric_gpu_kernel_with_scalars /
gpu_kernel_multiple_outputs, this script:

1. Extracts the functor/lambda passed to the call.
2. Finds the enclosing AT_DISPATCH_* macro (if any) to determine the
   scalar_t type instantiations.
3. Outputs a JSON record per instantiation with:
   - file, line of the gpu_kernel* call
   - kernel_entry: which gpu_kernel variant is called
   - dispatch_macro: the AT_DISPATCH_* macro name (or null)
   - scalar_type: the concrete C++ type for scalar_t
   - functor_code: the functor/lambda source with scalar_t replaced
   - raw_functor_code: the original functor/lambda source (before replacement)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# ScalarType -> C++ type mapping
# ---------------------------------------------------------------------------

SCALAR_TYPE_TO_CPP = {
    "Float": "float",
    "Double": "double",
    "Half": "at::Half",
    "BFloat16": "at::BFloat16",
    "Bool": "bool",
    "Byte": "uint8_t",
    "Char": "int8_t",
    "Int": "int32_t",
    "Long": "int64_t",
    "Short": "int16_t",
    "ComplexFloat": "c10::complex<float>",
    "ComplexDouble": "c10::complex<double>",
    "ComplexHalf": "c10::complex<at::Half>",
    "Float8_e4m3fn": "at::Float8_e4m3fn",
    "Float8_e5m2": "at::Float8_e5m2",
    "Float8_e4m3fnuz": "at::Float8_e4m3fnuz",
    "Float8_e5m2fnuz": "at::Float8_e5m2fnuz",
}

# Shortcuts used in source (kHalf, kBFloat16 etc.)
SCALAR_SHORT = {
    "kHalf": "Half",
    "kBFloat16": "BFloat16",
    "kBool": "Bool",
    "kByte": "Byte",
    "kChar": "Char",
    "kInt": "Int",
    "kLong": "Long",
    "kShort": "Short",
    "kFloat": "Float",
    "kDouble": "Double",
    "kComplexFloat": "ComplexFloat",
    "kComplexDouble": "ComplexDouble",
    "kComplexHalf": "ComplexHalf",
    "kFloat8_e4m3fn": "Float8_e4m3fn",
    "kFloat8_e5m2": "Float8_e5m2",
    "kQInt8": "QInt8",
    "kQUInt8": "QUInt8",
    "kQInt32": "QInt32",
}

FLOATING_TYPES = ["Float", "Double"]
INTEGRAL_TYPES = ["Byte", "Char", "Int", "Long", "Short"]
ALL_TYPES = INTEGRAL_TYPES + FLOATING_TYPES
COMPLEX_TYPES = ["ComplexFloat", "ComplexDouble"]

# AT_DISPATCH macro -> (base_types, n_extra_scalar_args)
# n_extra_scalar_args: how many leading macro arguments are extra ScalarType values
DISPATCH_MACROS: dict[str, tuple[list[str], int]] = {
    "AT_DISPATCH_FLOATING_TYPES": (FLOATING_TYPES, 0),
    "AT_DISPATCH_FLOATING_TYPES_AND_HALF": (FLOATING_TYPES + ["Half"], 0),
    "AT_DISPATCH_FLOATING_TYPES_AND": (FLOATING_TYPES, 1),
    "AT_DISPATCH_FLOATING_TYPES_AND2": (FLOATING_TYPES, 2),
    "AT_DISPATCH_FLOATING_TYPES_AND3": (FLOATING_TYPES, 3),
    "AT_DISPATCH_COMPLEX_TYPES": (COMPLEX_TYPES, 0),
    "AT_DISPATCH_COMPLEX_TYPES_AND": (COMPLEX_TYPES, 1),
    "AT_DISPATCH_COMPLEX_TYPES_AND2": (COMPLEX_TYPES, 2),
    "AT_DISPATCH_INTEGRAL_TYPES": (INTEGRAL_TYPES, 0),
    "AT_DISPATCH_INTEGRAL_TYPES_AND": (INTEGRAL_TYPES, 1),
    "AT_DISPATCH_INTEGRAL_TYPES_AND2": (INTEGRAL_TYPES, 2),
    "AT_DISPATCH_ALL_TYPES": (ALL_TYPES, 0),
    "AT_DISPATCH_ALL_TYPES_AND": (ALL_TYPES, 1),
    "AT_DISPATCH_ALL_TYPES_AND2": (ALL_TYPES, 2),
    "AT_DISPATCH_ALL_TYPES_AND3": (ALL_TYPES, 3),
    "AT_DISPATCH_ALL_TYPES_AND_COMPLEX": (ALL_TYPES + COMPLEX_TYPES, 0),
    "AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND": (ALL_TYPES + COMPLEX_TYPES, 1),
    "AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2": (ALL_TYPES + COMPLEX_TYPES, 2),
    "AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3": (ALL_TYPES + COMPLEX_TYPES, 3),
    "AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4": (ALL_TYPES + COMPLEX_TYPES, 4),
    "AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES": (FLOATING_TYPES + COMPLEX_TYPES, 0),
    "AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND": (FLOATING_TYPES + COMPLEX_TYPES, 1),
    "AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2": (FLOATING_TYPES + COMPLEX_TYPES, 2),
    "AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3": (FLOATING_TYPES + COMPLEX_TYPES, 3),
}

GPU_KERNEL_NAMES = [
    "gpu_kernel",
    "gpu_kernel_nocast",
    "gpu_kernel_with_scalars",
    "opmath_gpu_kernel_with_scalars",
    "opmath_symmetric_gpu_kernel_with_scalars",
    "gpu_kernel_multiple_outputs",
]

# Template type parameter names used instead of scalar_t in template functions
TEMPLATE_TYPE_PARAMS = {"T", "scalar_t", "Base_type", "Exp_type"}


# ---------------------------------------------------------------------------
# Balanced-delimiter helpers
# ---------------------------------------------------------------------------

def find_balanced(text: str, start: int, open_ch: str = "(", close_ch: str = ")") -> int:
    """Return the index of the matching close delimiter, or -1."""
    depth = 0
    i = start
    in_string = False
    string_char = None
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\" and i + 1 < len(text):
                i += 2
                continue
            if ch == string_char:
                in_string = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = True
            string_char = ch
            i += 1
            continue
        # Skip line comments
        if ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            nl = text.find("\n", i)
            i = nl + 1 if nl != -1 else len(text)
            continue
        # Skip block comments
        if ch == "/" and i + 1 < len(text) and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            i = end + 2 if end != -1 else len(text)
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def split_top_level_args(text: str) -> list[str]:
    """Split text by commas at the top level (respecting nested parens/braces/brackets)."""
    args: list[str] = []
    depth_paren = 0
    depth_brace = 0
    depth_angle = 0
    depth_bracket = 0
    current: list[str] = []
    in_string = False
    string_char = None
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            current.append(ch)
            if ch == "\\" and i + 1 < len(text):
                current.append(text[i + 1])
                i += 2
                continue
            if ch == string_char:
                in_string = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = True
            string_char = ch
            current.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            nl = text.find("\n", i)
            if nl == -1:
                current.append(text[i:])
                break
            current.append(text[i : nl + 1])
            i = nl + 1
            continue
        if ch == "/" and i + 1 < len(text) and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            if end == -1:
                current.append(text[i:])
                break
            current.append(text[i : end + 2])
            i = end + 2
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace -= 1
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket -= 1
        elif ch == "<" and depth_paren == 0 and depth_brace == 0:
            depth_angle += 1
        elif ch == ">" and depth_angle > 0:
            depth_angle -= 1

        if (
            ch == ","
            and depth_paren == 0
            and depth_brace == 0
            and depth_angle == 0
            and depth_bracket == 0
        ):
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
        i += 1

    rest = "".join(current).strip()
    if rest:
        args.append(rest)
    return args


# ---------------------------------------------------------------------------
# Normalize a scalar type token to a canonical ScalarType name
# ---------------------------------------------------------------------------

def normalize_scalar_type(tok: str) -> Optional[str]:
    tok = tok.strip()
    # at::ScalarType::Half, ScalarType::Half, at::kHalf, kHalf
    for prefix in ("at::ScalarType::", "ScalarType::", "at::"):
        if tok.startswith(prefix):
            tok = tok[len(prefix):]
            break
    if tok in SCALAR_SHORT:
        tok = SCALAR_SHORT[tok]
    if tok in SCALAR_TYPE_TO_CPP:
        return tok
    return None


# ---------------------------------------------------------------------------
# Data class for results
# ---------------------------------------------------------------------------

@dataclass
class KernelInstantiation:
    file: str
    line: int
    kernel_entry: str
    dispatch_macro: Optional[str]
    dispatch_line: Optional[int]
    scalar_type_name: Optional[str]  # e.g. "Float"
    cpp_type: Optional[str]  # e.g. "float"
    raw_functor_code: str
    functor_code: str
    # extra template args on the gpu_kernel call (e.g. for opmath_gpu_kernel_with_scalars)
    template_args: Optional[str] = None


# ---------------------------------------------------------------------------
# Find gpu_kernel* calls in a file
# ---------------------------------------------------------------------------

# Regex: optionally template args, then opening paren
GPU_KERNEL_RE = re.compile(
    r"\b("
    + "|".join(re.escape(n) for n in GPU_KERNEL_NAMES)
    + r")\s*(<[^>]*>)?\s*\("
)


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def find_enclosing_template_func(text: str, pos: int) -> Optional[tuple[str, str, int]]:
    """Find the enclosing template function for a position.
    Returns (function_name, template_type_param, func_start_pos) or None.
    Handles patterns like:
      template <typename T>
      void FuncName(...) {
        ... gpu_kernel(...) ...
      }
    """
    # Search backwards for 'template' keyword followed by function definition
    search_region = text[:pos]
    # Find all template function definitions before pos
    tmpl_re = re.compile(
        r'template\s*<\s*typename\s+(\w+)(?:\s*,\s*typename\s+\w+)*\s*>\s*'
        r'(?:static\s+)?(?:inline\s+)?(?:void|auto|\w+)\s+'
        r'(\w+)\s*(?:<[^>]*>)?\s*\(',
        re.DOTALL,
    )
    best = None
    for m in tmpl_re.finditer(search_region):
        type_param = m.group(1)
        func_name = m.group(2)
        # Find the opening brace of the function body
        after_sig = m.end() - 1  # at the '('
        close_sig = find_balanced(text, after_sig, "(", ")")
        if close_sig == -1:
            continue
        # Skip possible trailing qualifiers and find '{'
        brace_pos = text.find("{", close_sig + 1)
        if brace_pos == -1 or brace_pos > close_sig + 50:
            continue
        # Check that pos is inside this function body
        close_brace = find_balanced(text, brace_pos, "{", "}")
        if close_brace == -1:
            continue
        if brace_pos < pos <= close_brace:
            best = (func_name, type_param, m.start())
    return best


def find_template_func_call_sites(
    text: str, func_name: str, type_param: str, depth: int = 0
) -> list[tuple[int, list[str]]]:
    """Find call sites of a template function and resolve types from enclosing AT_DISPATCH.
    Recursively traces through intermediate template functions (up to 3 levels).
    Returns list of (call_pos, [scalar_type_names]).
    """
    if depth > 3:
        return []
    # Match func_name<...>( or func_name<scalar_t>(
    call_re = re.compile(
        r'\b' + re.escape(func_name) + r'\s*<\s*([^>]+)\s*>\s*\('
    )
    results = []
    for m in call_re.finditer(text):
        call_pos = m.start()
        tmpl_arg = m.group(1).strip()
        dispatch_info = find_enclosing_dispatch(text, call_pos)
        if dispatch_info is not None:
            macro_name, macro_pos, types = dispatch_info
            results.append((call_pos, types))
        else:
            # No dispatch, check if template arg is a concrete type
            st = normalize_scalar_type(tmpl_arg)
            if st:
                results.append((call_pos, [st]))
            else:
                # The call is inside another template function; trace up
                parent = find_enclosing_template_func(text, call_pos)
                if parent is not None:
                    parent_name, parent_type_param, _ = parent
                    if parent_name != func_name:  # avoid self-recursion
                        parent_sites = find_template_func_call_sites(
                            text, parent_name, parent_type_param, depth + 1
                        )
                        results.extend(parent_sites)
    return results


def find_using_typedef(text: str, pos: int) -> Optional[tuple[str, str]]:
    """Look for 'using scalar_t = SomeType;' before pos in the same scope.
    Returns (alias_name, scalar_type_name) or None."""
    # Build a known C++ type -> ScalarType mapping
    cpp_to_scalar: dict[str, str] = {}
    for st, cpp in SCALAR_TYPE_TO_CPP.items():
        cpp_to_scalar[cpp] = st
    # Also handle common aliases like c10::complex<at::Half>
    cpp_to_scalar["c10::complex<at::Half>"] = "ComplexHalf"

    using_re = re.compile(
        r'using\s+(\w+)\s*=\s*([^;]+);'
    )
    best = None
    for m in using_re.finditer(text[:pos]):
        alias = m.group(1)
        type_expr = m.group(2).strip()
        st = None
        for cpp, stype in cpp_to_scalar.items():
            if type_expr == cpp:
                st = stype
                break
        if st is not None:
            best = (alias, st)
    return best


def extract_calls(filepath: str, text: str) -> list[KernelInstantiation]:
    results: list[KernelInstantiation] = []
    for m in GPU_KERNEL_RE.finditer(text):
        kernel_name = m.group(1)
        template_args = m.group(2)
        # position of the opening '('
        open_paren = m.end() - 1
        close_paren = find_balanced(text, open_paren, "(", ")")
        if close_paren == -1:
            continue

        call_line = line_of(text, m.start())
        inner = text[open_paren + 1 : close_paren]
        top_args = split_top_level_args(inner)
        if len(top_args) < 2:
            continue

        # The functor is everything after the first argument (iter)
        # For most calls it's the second arg; for some there may be more
        functor_code = top_args[-1].strip()
        raw_functor_code = functor_code

        # Find enclosing AT_DISPATCH_* macro
        dispatch_info = find_enclosing_dispatch(text, m.start())

        # If no direct AT_DISPATCH, try indirect via template function
        type_param_name = "scalar_t"  # default replacement target
        if dispatch_info is None:
            tmpl_info = find_enclosing_template_func(text, m.start())
            if tmpl_info is not None:
                func_name, type_param, _ = tmpl_info
                type_param_name = type_param
                call_sites = find_template_func_call_sites(text, func_name, type_param)
                if call_sites:
                    all_types: list[str] = []
                    seen = set()
                    for _, types in call_sites:
                        for t in types:
                            if t not in seen:
                                seen.add(t)
                                all_types.append(t)
                    dispatch_info = ("(indirect via " + func_name + ")", 0, all_types)

        # If still no dispatch, look for explicit `using scalar_t = Type;`
        if dispatch_info is None:
            using_info = find_using_typedef(text, m.start())
            if using_info is not None:
                alias_name, st = using_info
                type_param_name = alias_name
                dispatch_info = ("(explicit using)", 0, [st])

        if dispatch_info is not None:
            macro_name, macro_pos, types = dispatch_info
            macro_line = line_of(text, macro_pos) if macro_pos > 0 else None
            for st in types:
                cpp = SCALAR_TYPE_TO_CPP.get(st)
                if cpp is None:
                    continue
                replaced = replace_type_param(functor_code, type_param_name, cpp, st)
                results.append(KernelInstantiation(
                    file=filepath,
                    line=call_line,
                    kernel_entry=kernel_name,
                    dispatch_macro=macro_name,
                    dispatch_line=macro_line,
                    scalar_type_name=st,
                    cpp_type=cpp,
                    raw_functor_code=raw_functor_code,
                    functor_code=replaced,
                    template_args=template_args,
                ))
        else:
            # No AT_DISPATCH found: the types are explicit in the code
            results.append(KernelInstantiation(
                file=filepath,
                line=call_line,
                kernel_entry=kernel_name,
                dispatch_macro=None,
                dispatch_line=None,
                scalar_type_name=None,
                cpp_type=None,
                raw_functor_code=raw_functor_code,
                functor_code=raw_functor_code,
                template_args=template_args,
            ))

    return results


# ---------------------------------------------------------------------------
# Find the enclosing AT_DISPATCH_* macro for a given position
# ---------------------------------------------------------------------------

DISPATCH_RE = re.compile(
    r"\b(AT_DISPATCH_\w+)\s*\("
)


def find_enclosing_dispatch(text: str, pos: int) -> Optional[tuple[str, int, list[str]]]:
    """Search backwards from pos for the nearest enclosing AT_DISPATCH_* macro
    whose balanced parentheses enclose pos. Return (macro_name, macro_pos, types)."""
    # Collect all dispatch macro occurrences before pos
    candidates = []
    for m in DISPATCH_RE.finditer(text):
        if m.start() > pos:
            break
        macro_name = m.group(1)
        if macro_name not in DISPATCH_MACROS and not macro_name.startswith("AT_DISPATCH_V2"):
            continue
        candidates.append((m.start(), m.end(), macro_name, m.end() - 1))

    # Check candidates from nearest to farthest
    for start, end, macro_name, open_paren in reversed(candidates):
        close = find_balanced(text, open_paren, "(", ")")
        if close == -1:
            continue
        if open_paren < pos <= close:
            # pos is inside this macro call
            types = resolve_dispatch_types(text, macro_name, open_paren, close)
            return (macro_name, start, types)
    return None


def resolve_dispatch_types(
    text: str, macro_name: str, open_paren: int, close_paren: int
) -> list[str]:
    """Given an AT_DISPATCH_* macro call, return the list of ScalarType names."""
    if macro_name.startswith("AT_DISPATCH_V2"):
        # AT_DISPATCH_V2 is complex; we handle it best-effort
        return resolve_dispatch_v2(text, open_paren, close_paren)

    if macro_name not in DISPATCH_MACROS:
        return []

    base_types, n_extra = DISPATCH_MACROS[macro_name]
    types = list(base_types)

    if n_extra > 0:
        inner = text[open_paren + 1 : close_paren]
        args = split_top_level_args(inner)
        for i in range(min(n_extra, len(args))):
            st = normalize_scalar_type(args[i])
            if st is not None:
                types.append(st)

    return types


def resolve_dispatch_v2(text: str, open_paren: int, close_paren: int) -> list[str]:
    """Best-effort: extract types from AT_DISPATCH_V2 by finding AT_DISPATCH_CASE* in the body."""
    inner = text[open_paren + 1 : close_paren]
    types: list[str] = []
    # Look for AT_DISPATCH_CASE_FLOATING_TYPES etc.
    for m in re.finditer(r"AT_DISPATCH_CASE_(\w+)", inner):
        case_name = m.group(1)
        mapping = {
            "FLOATING_TYPES": FLOATING_TYPES,
            "INTEGRAL_TYPES": INTEGRAL_TYPES,
            "ALL_TYPES": ALL_TYPES,
            "COMPLEX_TYPES": COMPLEX_TYPES,
            "ALL_TYPES_AND_COMPLEX": ALL_TYPES + COMPLEX_TYPES,
        }
        if case_name in mapping:
            types.extend(mapping[case_name])
    # Also look for AT_DISPATCH_CASE(ScalarType, ...) individual cases
    for m in re.finditer(r"AT_DISPATCH_CASE\s*\(\s*([^,)]+)", inner):
        st = normalize_scalar_type(m.group(1))
        if st is not None:
            types.append(st)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for t in types:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


# ---------------------------------------------------------------------------
# Replace scalar_t (and related type aliases) with concrete types
# ---------------------------------------------------------------------------

OPMATH_MAP = {
    "float": "float",
    "double": "double",
    "at::Half": "float",
    "at::BFloat16": "float",
    "c10::complex<float>": "c10::complex<float>",
    "c10::complex<double>": "c10::complex<double>",
    "c10::complex<at::Half>": "c10::complex<float>",
    "bool": "float",
    "uint8_t": "float",
    "int8_t": "float",
    "int16_t": "float",
    "int32_t": "float",
    "int64_t": "float",
}

# acc_type<T, true> (GPU) resolution
ACC_TYPE_MAP = {
    "float": "float",
    "double": "double",
    "at::Half": "float",
    "at::BFloat16": "float",
    "c10::complex<float>": "c10::complex<float>",
    "c10::complex<double>": "c10::complex<double>",
}


def replace_type_param(code: str, type_param: str, cpp_type: str, scalar_type_name: str) -> str:
    """Replace a type parameter (scalar_t, T, etc.) with the concrete C++ type."""
    result = re.sub(r'\b' + re.escape(type_param) + r'\b', cpp_type, code)

    # Also replace opmath_t
    opmath = OPMATH_MAP.get(cpp_type, "float")
    result = re.sub(r'\bopmath_t\b', opmath, result)

    # Also replace T_ACC if present (used in group_norm etc.)
    acc = ACC_TYPE_MAP.get(cpp_type, "float")
    result = re.sub(r'\bT_ACC\b', acc, result)

    return result


def replace_scalar_t(code: str, cpp_type: str, scalar_type_name: str) -> str:
    return replace_type_param(code, "scalar_t", cpp_type, scalar_type_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_file(filepath: str, root: str) -> list[KernelInstantiation]:
    with open(filepath) as f:
        text = f.read()

    relpath = os.path.relpath(filepath, root)
    return extract_calls(relpath, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=None,
        help="Root of the PyTorch repo (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file (default: stdout)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a human-readable summary instead of JSON",
    )
    args = parser.parse_args()

    if args.root is None:
        # Assume script is at tools/extract_gpu_kernels.py
        args.root = str(Path(__file__).resolve().parent.parent)

    cuda_dir = os.path.join(args.root, "aten", "src", "ATen", "native", "cuda")
    if not os.path.isdir(cuda_dir):
        print(f"Error: {cuda_dir} not found", file=sys.stderr)
        sys.exit(1)

    all_results: list[KernelInstantiation] = []

    for fname in sorted(os.listdir(cuda_dir)):
        if not (fname.endswith(".cu") or fname.endswith(".cuh")):
            continue
        # Skip the definition files themselves
        if fname in ("CUDALoops.cuh", "Loops.cuh", "JitLoops.cuh", "CUDAJitLoops.cuh"):
            continue
        fpath = os.path.join(cuda_dir, fname)
        results = process_file(fpath, args.root)
        all_results.extend(results)

    if args.summary:
        print(f"Found {len(all_results)} kernel instantiation(s) across files:\n")
        by_file: dict[str, list[KernelInstantiation]] = {}
        for r in all_results:
            by_file.setdefault(r.file, []).append(r)

        for filepath, insts in sorted(by_file.items()):
            print(f"=== {filepath} ===")
            # Group by (line, kernel_entry)
            by_call: dict[tuple[int, str], list[KernelInstantiation]] = {}
            for inst in insts:
                key = (inst.line, inst.kernel_entry)
                by_call.setdefault(key, []).append(inst)

            for (line, entry), group in sorted(by_call.items()):
                types = [g.cpp_type or "?" for g in group]
                macro = group[0].dispatch_macro or "(none)"
                print(f"  Line {line}: {entry}  [{macro}]")
                print(f"    Types: {', '.join(types)}")
                # Show raw functor code (truncated)
                raw = group[0].raw_functor_code
                preview = raw[:120].replace("\n", " ") + ("..." if len(raw) > 120 else "")
                print(f"    Functor: {preview}")
                print()
            print()
    else:
        output_data = [asdict(r) for r in all_results]
        json_str = json.dumps(output_data, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(json_str)
            print(f"Wrote {len(all_results)} instantiations to {args.output}", file=sys.stderr)
        else:
            print(json_str)


if __name__ == "__main__":
    main()
