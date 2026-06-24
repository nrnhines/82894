#!/usr/bin/env python3
"""Run nrntraub benchmark cases, compare spikes, print a performance table."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


BENCHMARK_RE = re.compile(
    r"BENCHMARK runtime=(?P<runtime>\S+) setuptime=(?P<setup>\S+) "
    r"ncell=(?P<ncell>\d+) enable_gpu=(?P<enable_gpu>\d+) nhost=(?P<nhost>\d+)"
)
RUNTIME_RE = re.compile(r"^RunTime:\s+(\S+)", re.MULTILINE)


@dataclass
class CaseResult:
    name: str
    returncode: int
    runtime: float | None = None
    setuptime: float | None = None
    ncell: int | None = None
    enable_gpu: int | None = None
    nhost: int | None = None
    spike_count: int | None = None
    spikes_match_ref: bool | None = None
    speedup_vs_ref: float | None = None
    log_path: Path | None = None
    spikes_path: Path | None = None
    sorted_spikes_path: Path | None = None
    error: str | None = None


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise SystemExit("PyYAML required for YAML config (pip install pyyaml)")
        return yaml.safe_load(text)
    return json.loads(text)


def resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def hoc_to_argv(hoc: dict[str, Any]) -> list[str]:
    """Build special -c arguments. NEURON CLI cannot assign HOC strings; skip those."""
    argv: list[str] = []
    for key, value in hoc.items():
        if isinstance(value, str):
            continue
        if isinstance(value, bool):
            argv.extend(["-c", f"{key}={int(value)}"])
        else:
            argv.extend(["-c", f"{key}={value}"])
    return argv


def build_env(config: dict[str, Any], config_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in config.get("env", {}).items():
        env[key] = str(value)
    path_parts: list[str] = []
    nrn_bin = config.get("nrn_bin")
    if nrn_bin:
        path_parts.append(str(resolve_path(config_dir, nrn_bin)))
    nvhpc = env.get("NVHPC_BIN")
    if nvhpc:
        path_parts.append(nvhpc)
    if path_parts:
        env["PATH"] = ":".join(path_parts + [env.get("PATH", "")])
    return env


def parse_run_output(stdout: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    m = BENCHMARK_RE.search(stdout)
    if m:
        parsed["runtime"] = float(m.group("runtime"))
        parsed["setuptime"] = float(m.group("setup"))
        parsed["ncell"] = int(m.group("ncell"))
        parsed["enable_gpu"] = int(m.group("enable_gpu"))
        parsed["nhost"] = int(m.group("nhost"))
        return parsed
    m = RUNTIME_RE.search(stdout)
    if m:
        parsed["runtime"] = float(m.group(1))
    return parsed


def count_spikes(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def sort_spikes(sortspike: Path, src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(sortspike), str(src), str(dst)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def run_case(
    *,
    name: str,
    model_dir: Path,
    special: Path,
    sortspike: Path,
    hoc: dict[str, Any],
    output_dir: Path,
    env: dict[str, str],
) -> CaseResult:
    case_dir = output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / "stdout.log"

    cmd = [str(special), *hoc_to_argv(hoc), "init.hoc"]
    proc = subprocess.run(
        cmd,
        cwd=model_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(proc.stdout)

    result = CaseResult(name=name, returncode=proc.returncode, log_path=log_path)
    if proc.returncode != 0:
        result.error = f"exit {proc.returncode}"
        tail = "\n".join(proc.stdout.splitlines()[-20:])
        result.error = f"{result.error}\n{tail}"
        return result

    parsed = parse_run_output(proc.stdout)
    result.runtime = parsed.get("runtime")
    result.setuptime = parsed.get("setuptime")
    result.ncell = parsed.get("ncell")
    result.enable_gpu = parsed.get("enable_gpu")
    result.nhost = parsed.get("nhost")

    raw_spikes = model_dir / "out1.dat"
    if not raw_spikes.is_file():
        result.error = f"missing spike file {raw_spikes}"
        return result

    spikes_path = case_dir / "out1.dat"
    shutil.copy2(raw_spikes, spikes_path)
    result.spikes_path = spikes_path
    result.spike_count = count_spikes(spikes_path)

    sorted_path = case_dir / "spikes.srt"
    try:
        sort_spikes(sortspike, spikes_path, sorted_path)
        result.sorted_spikes_path = sorted_path
    except subprocess.CalledProcessError as exc:
        result.error = f"sortspike failed: {exc.stderr}"

    perf_src = model_dir / "perf.dat"
    if perf_src.is_file():
        shutil.copy2(perf_src, case_dir / "perf.dat")

    return result


def compare_spikes(reference: Path, candidate: Path) -> bool:
    return reference.read_text() == candidate.read_text()


def merge_hoc(config: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config.get("hoc_defaults", {}))
    merged.update(case.get("hoc", {}))
    return merged


def format_table(results: list[CaseResult], reference: str) -> str:
    headers = [
        "case",
        "ncell",
        "runtime_s",
        "setup_s",
        "speedup",
        "spikes",
        "match_ref",
        "status",
    ]
    rows: list[list[str]] = []
    ref_runtime = next(
        (r.runtime for r in results if r.name == reference and r.runtime is not None),
        None,
    )

    for r in results:
        speedup = ""
        if r.runtime is not None and ref_runtime and r.name != reference:
            speedup = f"{ref_runtime / r.runtime:.3f}x"
        elif r.runtime is not None and r.name == reference:
            speedup = "1.000x"

        if r.returncode != 0 or r.error:
            status = "FAIL"
        elif r.spikes_match_ref is False:
            status = "SPIKE_DIFF"
        else:
            status = "ok"

        rows.append(
            [
                r.name,
                str(r.ncell or ""),
                f"{r.runtime:.4f}" if r.runtime is not None else "",
                f"{r.setuptime:.4f}" if r.setuptime is not None else "",
                speedup,
                str(r.spike_count or ""),
                "" if r.spikes_match_ref is None else ("yes" if r.spikes_match_ref else "NO"),
                status,
            ]
        )

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
        help="benchmark config file (YAML or JSON)",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        help="run only these case names (default: all in config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without executing",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config_dir = config_path.parent
    config = load_config(config_path)

    model_dir = resolve_path(config_dir, config.get("model_dir", "."))
    special = resolve_path(config_dir, config["special"])
    sortspike = resolve_path(config_dir, config.get("sortspike", "sortspike"))
    if config.get("nrn_bin"):
        sortspike_candidate = resolve_path(config_dir, config["nrn_bin"]) / "sortspike"
        if sortspike_candidate.is_file():
            sortspike = sortspike_candidate

    output_dir = resolve_path(config_dir, config.get("output_dir", "results"))
    reference = config.get("reference_case", "neuron_cpu")
    env = build_env(config, config_dir)

    selected = {c["name"]: c for c in config["cases"]}
    if args.cases:
        unknown = set(args.cases) - set(selected)
        if unknown:
            raise SystemExit(f"unknown cases: {', '.join(sorted(unknown))}")
        case_list = [selected[name] for name in args.cases]
    else:
        case_list = config["cases"]

    if not special.is_file():
        raise SystemExit(f"special not found: {special} (run nrnivmodl mod first)")

    print(f"model_dir: {model_dir}")
    print(f"special:     {special}")
    print(f"output_dir:  {output_dir}")
    print()

    results: list[CaseResult] = []
    for case in case_list:
        name = case["name"]
        hoc = merge_hoc(config, case)
        cmd = [str(special), *hoc_to_argv(hoc), "init.hoc"]
        print(f"=== {name} ===")
        print(" ".join(cmd))
        if args.dry_run:
            results.append(CaseResult(name=name, returncode=0))
            continue

        result = run_case(
            name=name,
            model_dir=model_dir,
            special=special,
            sortspike=sortspike,
            hoc=hoc,
            output_dir=output_dir,
            env=env,
        )
        results.append(result)
        if result.error:
            print(f"ERROR: {result.error}")
        elif result.runtime is not None:
            print(
                f"runtime={result.runtime:.4f}s setup={result.setuptime:.4f}s "
                f"spikes={result.spike_count}"
            )
        print()

    if args.dry_run:
        return 0

    ref = next((r for r in results if r.name == reference), None)
    if ref and ref.sorted_spikes_path and ref.sorted_spikes_path.is_file():
        for r in results:
            if r.name == reference:
                r.spikes_match_ref = True
                if r.runtime is not None:
                    r.speedup_vs_ref = 1.0
                continue
            if r.sorted_spikes_path and r.sorted_spikes_path.is_file():
                r.spikes_match_ref = compare_spikes(
                    ref.sorted_spikes_path, r.sorted_spikes_path
                )
            if r.runtime is not None and ref.runtime:
                r.speedup_vs_ref = ref.runtime / r.runtime

    table = format_table(results, reference)
    print(table)

    summary_path = output_dir / "summary.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(table + "\n")
    print(f"\nWrote {summary_path}")

    failed = any(
        r.returncode != 0 or r.error or r.spikes_match_ref is False for r in results
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
