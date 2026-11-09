from __future__ import annotations

import json
from pathlib import Path
import sys

import bench

root = Path(__file__).resolve().parent
kind = sys.argv[1]
if kind == "tma":
    settings = [(variant, stages) for variant in ("baseline", "linear") for stages in (2, 4, 6, 8, 12, 16)]
    for variant, stages in settings:
        print(f"VARIANT {variant} STAGES {stages}", flush=True)
        sys.argv = ["bench.py", "--cases", "transpose_copy", "dense_add_host", "dense_add_tensor", "mixed_add", "--buffers", "4", "--variant", variant, "--stages", str(stages), "--implementations", "tma", "--output", str(root / f"sweep-{variant}-s{stages}.json")]
        bench.main()
elif kind == "cute":
    for tile in (32, 64, 128):
        for threads in (128, 256):
            print(f"CUTE TILE {tile} THREADS {threads}", flush=True)
            sys.argv = ["bench.py", "--cases", "transpose_copy", "--buffers", "4", "--cute-tile", str(tile), "--cute-threads", str(threads), "--implementations", "cute_ldg", "cute_ldgsts", "--output", str(root / f"sweep-cute-t{tile}-b{threads}.json")]
            bench.main()
elif kind == "rect":
    for variant, stages in (("rect64", 1), ("rect64", 2), ("rect128", 1), ("rect128", 2), ("rect256", 1), ("rect256", 2)):
        print(f"VARIANT {variant} STAGES {stages}", flush=True)
        sys.argv = ["bench.py", "--cases", "transpose_copy", "dense_add_host", "dense_add_tensor", "mixed_add", "--buffers", "4", "--variant", variant, "--stages", str(stages), "--implementations", "tma", "--output", str(root / f"sweep-{variant}-s{stages}.json")]
        bench.main()
elif kind == "classic":
    for columns in (32, 64, 128, 256):
        for threads in (128, 256):
            print(f"CLASSIC COLUMNS {columns} THREADS {threads}", flush=True)
            sys.argv = ["bench.py", "--cases", "transpose_copy", "--buffers", "4", "--cute-tile", str(columns), "--cute-threads", str(threads), "--cute-classic", "--implementations", "cute_ldg", "cute_ldgsts", "--output", str(root / f"sweep-classic-c{columns}-b{threads}.json")]
            bench.main()
elif kind == "parallel":
    for stages in (2, 4, 6, 8):
        print(f"PARALLEL STAGES {stages}", flush=True)
        sys.argv = ["bench.py", "--cases", "transpose_copy", "dense_add_host", "dense_add_tensor", "mixed_add", "--buffers", "4", "--variant", "parallel", "--stages", str(stages), "--implementations", "tma", "--output", str(root / f"sweep-parallel-s{stages}.json")]
        bench.main()
elif kind == "rowmajor":
    for variant in ("rowmajor", "rowmajor_linear", "rowmajor_vecstore"):
        for stages in (2, 4, 6):
            print(f"VARIANT {variant} STAGES {stages}", flush=True)
            sys.argv = ["bench.py", "--cases", "transpose_copy", "dense_add_host", "dense_add_tensor", "mixed_add", "--buffers", "4", "--variant", variant, "--stages", str(stages), "--implementations", "tma", "--output", str(root / f"sweep-{variant}-s{stages}.json")]
            bench.main()
