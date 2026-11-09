from __future__ import annotations

from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
sections = ("SpeedOfLight", "MemoryWorkloadAnalysis", "MemoryWorkloadAnalysis_Tables", "Occupancy", "SchedulerStats", "WarpStateStats", "LaunchStats", "InstructionStats", "SourceCounters")
settings = [
    ("transpose_copy", "tma", []),
    ("transpose_copy", "triton", ["--config", "32", "64", "4"]),
    ("dense_add_host", "tma", []),
    ("dense_add_host", "triton", ["--config", "128", "32", "8"]),
    ("dense_add_tensor", "tma", []),
    ("dense_add_tensor", "triton", ["--config", "128", "32", "8"]),
    ("transpose_copy", "cute_ldgsts", ["--cute-classic", "--cute-tile", "128", "--cute-threads", "128"]),
    ("dense_add_host", "cute_vector", []),
    ("dense_add_host", "tma", ["--stages", "2"]),
    ("transpose_copy", "tma", ["--variant", "rect128", "--stages", "1"]),
    ("dense_add_host", "tma", ["--variant", "rowmajor_vecstore", "--stages", "2"]),
    ("dense_add_tensor", "tma", ["--variant", "rowmajor_vecstore", "--stages", "2"]),
    ("transpose_copy", "cute_ldg", ["--cute-classic", "--cute-tile", "64", "--cute-threads", "128"]),
]
for i, (case, impl, flags) in enumerate(settings):
    name = f"profile-{i:02d}-{case}-{impl}"
    if (root / f"{name}-raw.csv").exists():
        continue
    print(name, flush=True)
    command = ["ncu", "--profile-from-start", "off", "--target-processes", "all", "--cache-control", "none", "--clock-control", "none"]
    for section in sections:
        command.extend(("--section", section))
    command.extend(("--force-overwrite", "--export", str(root / name), sys.executable, str(root / "bench.py"), "--cases", case, "--profile", impl, *flags))
    with (root / f"{name}.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = root / f"{name}.ncu-repz"
    for page in ("details", "raw"):
        command = ["ncu", "--import", str(report), "--page", page]
        if page == "raw":
            command.append("--csv")
        with (root / f"{name}-{page}.{'csv' if page == 'raw' else 'txt'}").open("w") as output:
            subprocess.run(command, stdout=output, check=True)
