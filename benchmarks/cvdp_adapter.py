"""CVDP benchmark dataset adapter for the Veri-Sure framework.

Loads NVIDIA's CVDP (Comprehensive Verilog Design Problems) dataset from
HuggingFace or local JSONL files and converts each problem into the format
expected by Veri-Sure's TopAgent pipeline.

Supported configs (no_commercial only -- open-source Icarus Verilog / cocotb):
  - cvdp_nonagentic_code_generation_no_commercial  (302 problems)
  - cvdp_agentic_code_generation_no_commercial      (92 problems)

Task categories handled:
  cid002 - RTL Code Completion
  cid003 - Specification-to-RTL Translation
  cid004 - RTL Code Modification
  cid005 - Spec-to-RTL: Module Instantiation & Component Reuse
  cid007 - RTL Code Improvement / Lint
  cid016 - RTL Debugging and Bug Fixing
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

CID_LABELS: dict[str, str] = {
    "cid002": "RTL Code Completion",
    "cid003": "Spec-to-RTL Translation",
    "cid004": "RTL Code Modification",
    "cid005": "Spec-to-RTL: Module Reuse",
    "cid007": "RTL Code Improvement",
    "cid012": "TB Stimulus Generation",
    "cid013": "TB Checker Generation",
    "cid014": "Assertion Generation",
    "cid016": "RTL Debugging / Bug Fixing",
}

# Task IDs where Veri-Sure's spec-to-RTL pipeline is applicable.
VERISURE_COMPATIBLE_CIDS: set[str] = {"cid002", "cid003", "cid004", "cid005", "cid007", "cid016"}


@dataclass
class CVDPProblem:
    """A single CVDP benchmark problem, normalised for Veri-Sure."""

    id: str
    cid: str               # e.g. "cid003"
    difficulty: str         # "easy", "medium", or "hard"
    prompt: str             # Natural-language task description
    context_files: dict[str, str]   # filepath -> content (input files shown to model)
    output_targets: dict[str, str]  # filepath -> expected content (empty in public release)
    harness_files: dict[str, str]   # filepath -> content (evaluation infrastructure)
    is_agentic: bool = False
    system_message: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def cid_label(self) -> str:
        return CID_LABELS.get(self.cid, self.cid)

    @property
    def is_verisure_compatible(self) -> bool:
        return self.cid in VERISURE_COMPATIBLE_CIDS

    # ------------------------------------------------------------------
    # Build the "spec" string that Veri-Sure's TopAgent.run(spec=...) expects.
    # ------------------------------------------------------------------

    def build_verisure_spec(self) -> str:
        """Compose a specification string from the CVDP prompt + context.

        For cid003 (spec-to-RTL) this is essentially the prompt itself.
        For code-completion / modification / debugging tasks we prepend the
        relevant existing RTL so the Architect+Coder agents have full context.
        """
        parts: list[str] = []

        # ---- Existing RTL / spec context ----
        rtl_files: dict[str, str] = {}
        spec_files: dict[str, str] = {}
        other_files: dict[str, str] = {}

        for fpath, content in self.context_files.items():
            lower = fpath.lower()
            if lower.endswith((".sv", ".v", ".svh", ".vh")):
                rtl_files[fpath] = content
            elif lower.endswith((".md", ".txt", ".rst")):
                spec_files[fpath] = content
            else:
                other_files[fpath] = content

        # Include specification documents first (if any).
        if spec_files:
            parts.append("=== Reference Documents ===")
            for fpath, content in spec_files.items():
                parts.append(f"--- {fpath} ---")
                parts.append(content.strip())
            parts.append("")

        # Include existing RTL context for completion / modification / debug tasks.
        if rtl_files and self.cid in ("cid002", "cid004", "cid005", "cid007", "cid016"):
            parts.append("=== Existing RTL Context ===")
            for fpath, content in rtl_files.items():
                parts.append(f"--- {fpath} ---")
                parts.append(content.strip())
            parts.append("")

        # ---- The actual task prompt ----
        parts.append("=== Task ===")
        parts.append(self.prompt.strip())

        return "\n".join(parts)

    def get_output_target_paths(self) -> list[str]:
        """Return the list of file paths the model is expected to produce."""
        return [k for k, v in self.output_targets.items() if v is not None]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_cvdp_from_jsonl(jsonl_path: str | Path) -> list[CVDPProblem]:
    """Load problems from a local CVDP JSONL file."""
    problems: list[CVDPProblem] = []
    path = Path(jsonl_path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            problems.append(_parse_record(raw))
    return problems


def load_cvdp_from_huggingface(
    config: str = "cvdp_nonagentic_code_generation_no_commercial",
    split: str = "eval",
    cache_dir: str | None = None,
) -> list[CVDPProblem]:
    """Load CVDP problems from HuggingFace datasets hub.

    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The `datasets` package is required. Install with: pip install datasets"
        )

    ds = load_dataset(
        "nvidia/cvdp-benchmark-dataset",
        name=config,
        split=split,
        cache_dir=cache_dir,
    )

    problems: list[CVDPProblem] = []
    is_agentic = "agentic" in config

    for row in ds:
        if is_agentic:
            prob = _parse_agentic_record(dict(row))
        else:
            prob = _parse_record(dict(row))
        problems.append(prob)

    return problems


def _parse_record(raw: dict[str, Any]) -> CVDPProblem:
    """Parse a non-agentic (copilot) CVDP record."""
    cats = raw.get("categories", [])
    cid = cats[0] if len(cats) > 0 else "unknown"
    difficulty = cats[1] if len(cats) > 1 else "unknown"

    inp = raw.get("input", {})
    prompt = inp.get("prompt", "")
    ctx_raw = inp.get("context") or {}

    context_files = {k: v for k, v in ctx_raw.items() if v is not None and v != ""}

    out = raw.get("output", {})
    out_ctx = out.get("context") or {}
    output_targets = {k: v for k, v in out_ctx.items() if v is not None}

    harness_raw = raw.get("harness", {})
    # Non-agentic format wraps files under a "files" key.
    harness_files_raw = harness_raw.get("files", harness_raw) if isinstance(harness_raw, dict) else {}
    harness_files = {k: v for k, v in harness_files_raw.items() if v is not None and v != ""}

    return CVDPProblem(
        id=raw.get("id", ""),
        cid=cid,
        difficulty=difficulty,
        prompt=prompt,
        context_files=context_files,
        output_targets=output_targets,
        harness_files=harness_files,
        is_agentic=False,
        raw=raw,
    )


def _parse_agentic_record(raw: dict[str, Any]) -> CVDPProblem:
    """Parse an agentic CVDP record."""
    cats = raw.get("categories", [])
    cid = cats[0] if len(cats) > 0 else "unknown"
    difficulty = cats[1] if len(cats) > 1 else "unknown"

    prompt = raw.get("prompt", "")
    system_message = raw.get("system_message", "")
    ctx_raw = raw.get("context") or {}
    context_files = {k: v for k, v in ctx_raw.items() if v is not None and v != ""}

    patch_raw = raw.get("patch") or {}
    output_targets = {k: v for k, v in patch_raw.items() if v is not None}

    harness_raw = raw.get("harness") or {}
    harness_files = {k: v for k, v in harness_raw.items() if v is not None and v != ""}

    return CVDPProblem(
        id=raw.get("id", ""),
        cid=cid,
        difficulty=difficulty,
        prompt=prompt,
        context_files=context_files,
        output_targets=output_targets,
        harness_files=harness_files,
        is_agentic=True,
        system_message=system_message,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Workspace setup for evaluation
# ---------------------------------------------------------------------------

def setup_cvdp_workspace(problem: CVDPProblem, workspace_dir: Path) -> Path:
    """Create a workspace directory with all CVDP files needed for evaluation.

    Layout:
        workspace_dir/
            input/          <- context files shown to model
            output/         <- where generated code should be placed
            harness/        <- evaluation infrastructure (cocotb, docker, etc.)
    """
    workspace_dir = workspace_dir.expanduser().resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Write input context files.
    for fpath, content in problem.context_files.items():
        dest = workspace_dir / fpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    # Write harness files.
    for fpath, content in problem.harness_files.items():
        dest = workspace_dir / fpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    # Write problem metadata.
    meta = {
        "id": problem.id,
        "cid": problem.cid,
        "difficulty": problem.difficulty,
        "is_agentic": problem.is_agentic,
        "output_targets": list(problem.output_targets.keys()),
    }
    (workspace_dir / "_problem_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return workspace_dir


def place_generated_code(
    problem: CVDPProblem,
    workspace_dir: Path,
    generated_rtl: str,
) -> list[Path]:
    """Place Veri-Sure's generated RTL into the CVDP workspace output paths.

    For non-agentic problems, the output target paths indicate which file(s)
    the model should produce. We write the generated RTL to the first RTL
    target, or to all targets if there's only one piece of generated code.

    Returns the list of paths written.
    """
    workspace_dir = workspace_dir.expanduser().resolve()
    written: list[Path] = []

    # Identify RTL output targets.
    rtl_targets = [
        t for t in problem.output_targets
        if t.lower().endswith((".sv", ".v", ".svh", ".vh"))
    ]
    # If no RTL targets found, look at context files with null values
    # (files that exist but weren't shown -- these are often the ones to produce).
    if not rtl_targets:
        ctx_raw = problem.raw.get("input", {}).get("context") or {}
        rtl_targets = [
            k for k, v in ctx_raw.items()
            if v is None and k.lower().endswith((".sv", ".v"))
        ]

    if not rtl_targets:
        # Fallback: write to rtl/design.sv
        rtl_targets = ["rtl/design.sv"]

    for target in rtl_targets:
        dest = workspace_dir / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(generated_rtl.rstrip() + "\n", encoding="utf-8")
        written.append(dest)

    return written


# ---------------------------------------------------------------------------
# CVDP harness-based evaluation (cocotb + Icarus Verilog)
# ---------------------------------------------------------------------------

@dataclass
class CVDPEvalResult:
    problem_id: str
    cid: str
    difficulty: str
    passed: bool
    score: float           # 0.0 - 1.0, fraction of tests passed
    target_score: float    # pass threshold from harness .env
    log: str
    error: str | None = None


def _parse_env_file(env_content: str) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file."""
    env = {}
    for line in env_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def evaluate_with_cvdp_harness(
    problem: CVDPProblem,
    workspace_dir: Path,
    *,
    use_docker: bool = True,
    timeout_s: float = 300.0,
) -> CVDPEvalResult:
    """Run the CVDP evaluation harness on a prepared workspace.

    If use_docker=True (default), runs via docker-compose as CVDP intends.
    If use_docker=False, attempts direct cocotb + iverilog evaluation.
    """
    workspace_dir = workspace_dir.expanduser().resolve()

    # Extract target score from harness .env file.
    target_score = 0.8  # default
    for fpath, content in problem.harness_files.items():
        if fpath.endswith(".env"):
            env = _parse_env_file(content)
            if "TARGET" in env:
                try:
                    target_score = float(env["TARGET"]) / 100.0
                except ValueError:
                    pass

    if use_docker:
        return _eval_docker(problem, workspace_dir, target_score, timeout_s)
    else:
        return _eval_direct(problem, workspace_dir, target_score, timeout_s)


def _eval_docker(
    problem: CVDPProblem,
    workspace_dir: Path,
    target_score: float,
    timeout_s: float,
) -> CVDPEvalResult:
    """Evaluate using docker-compose (faithful to CVDP benchmark)."""
    compose_file = workspace_dir / "docker-compose.yml"
    if not compose_file.exists():
        return CVDPEvalResult(
            problem_id=problem.id,
            cid=problem.cid,
            difficulty=problem.difficulty,
            passed=False,
            score=0.0,
            target_score=target_score,
            log="",
            error="docker-compose.yml not found in workspace",
        )

    try:
        result = subprocess.run(
            ["docker", "compose", "up", "--build", "--abort-on-container-exit"],
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        log = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        score = _parse_cocotb_score(result.stdout + result.stderr)
        passed = score >= target_score

        return CVDPEvalResult(
            problem_id=problem.id,
            cid=problem.cid,
            difficulty=problem.difficulty,
            passed=passed,
            score=score,
            target_score=target_score,
            log=log,
        )
    except subprocess.TimeoutExpired:
        return CVDPEvalResult(
            problem_id=problem.id,
            cid=problem.cid,
            difficulty=problem.difficulty,
            passed=False,
            score=0.0,
            target_score=target_score,
            log="TIMEOUT",
            error=f"Docker evaluation timed out after {timeout_s}s",
        )
    except FileNotFoundError:
        return CVDPEvalResult(
            problem_id=problem.id,
            cid=problem.cid,
            difficulty=problem.difficulty,
            passed=False,
            score=0.0,
            target_score=target_score,
            log="",
            error="docker not found -- install Docker or use use_docker=False",
        )


def _eval_direct(
    problem: CVDPProblem,
    workspace_dir: Path,
    target_score: float,
    timeout_s: float,
) -> CVDPEvalResult:
    """Evaluate directly using cocotb + Icarus Verilog (no Docker)."""
    # Find the cocotb test files in the harness.
    makefile = workspace_dir / "Makefile"
    src_dir = workspace_dir / "src"

    # Look for a process.py or test_runner.py in the harness.
    process_script = None
    for candidate in [src_dir / "process.py", workspace_dir / "process.py"]:
        if candidate.exists():
            process_script = candidate
            break

    if process_script is not None:
        try:
            result = subprocess.run(
                ["python", str(process_script)],
                cwd=str(workspace_dir),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env={**os.environ, "SIM": "icarus"},
            )
            log = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            score = _parse_cocotb_score(result.stdout + result.stderr)
            passed = score >= target_score
            return CVDPEvalResult(
                problem_id=problem.id,
                cid=problem.cid,
                difficulty=problem.difficulty,
                passed=passed,
                score=score,
                target_score=target_score,
                log=log,
            )
        except subprocess.TimeoutExpired:
            return CVDPEvalResult(
                problem_id=problem.id,
                cid=problem.cid,
                difficulty=problem.difficulty,
                passed=False,
                score=0.0,
                target_score=target_score,
                log="TIMEOUT",
                error=f"Direct evaluation timed out after {timeout_s}s",
            )

    # Fallback: try running cocotb via Makefile.
    if makefile.exists():
        try:
            result = subprocess.run(
                ["make"],
                cwd=str(workspace_dir),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env={**os.environ, "SIM": "icarus"},
            )
            log = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            score = _parse_cocotb_score(result.stdout + result.stderr)
            passed = score >= target_score
            return CVDPEvalResult(
                problem_id=problem.id,
                cid=problem.cid,
                difficulty=problem.difficulty,
                passed=passed,
                score=score,
                target_score=target_score,
                log=log,
            )
        except subprocess.TimeoutExpired:
            return CVDPEvalResult(
                problem_id=problem.id,
                cid=problem.cid,
                difficulty=problem.difficulty,
                passed=False,
                score=0.0,
                target_score=target_score,
                log="TIMEOUT",
                error=f"Direct evaluation timed out after {timeout_s}s",
            )

    return CVDPEvalResult(
        problem_id=problem.id,
        cid=problem.cid,
        difficulty=problem.difficulty,
        passed=False,
        score=0.0,
        target_score=target_score,
        log="",
        error="No evaluation script found (process.py or Makefile)",
    )


def _parse_cocotb_score(log: str) -> float:
    """Extract pass ratio from cocotb/pytest output.

    cocotb test results typically look like:
        X passed, Y failed
    or pytest summary:
        === X passed, Y failed in ...s ===
    """
    # pytest-style: "X passed" / "Y failed"
    passed_m = re.search(r"(\d+)\s+passed", log)
    failed_m = re.search(r"(\d+)\s+failed", log)

    if passed_m:
        n_passed = int(passed_m.group(1))
        n_failed = int(failed_m.group(1)) if failed_m else 0
        total = n_passed + n_failed
        if total > 0:
            return n_passed / total

    # cocotb-style: "PASS (X/Y)" or percentage
    pct_m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:pass|coverage)", log, re.IGNORECASE)
    if pct_m:
        return float(pct_m.group(1)) / 100.0

    # Check for simple PASS / FAIL markers.
    if "PASS" in log and "FAIL" not in log:
        return 1.0

    return 0.0


# ---------------------------------------------------------------------------
# Filtering / selection helpers
# ---------------------------------------------------------------------------

def filter_problems(
    problems: list[CVDPProblem],
    *,
    cids: set[str] | None = None,
    difficulties: set[str] | None = None,
    verisure_compatible_only: bool = False,
    limit: int | None = None,
) -> list[CVDPProblem]:
    """Filter CVDP problems by category, difficulty, and compatibility."""
    result = problems

    if verisure_compatible_only:
        result = [p for p in result if p.is_verisure_compatible]

    if cids is not None:
        result = [p for p in result if p.cid in cids]

    if difficulties is not None:
        result = [p for p in result if p.difficulty in difficulties]

    if limit is not None:
        result = result[:limit]

    return result


def summarize_dataset(problems: list[CVDPProblem]) -> str:
    """Return a human-readable summary of the problem distribution."""
    from collections import Counter

    cid_counts = Counter(p.cid for p in problems)
    diff_counts = Counter(p.difficulty for p in problems)
    compat_count = sum(1 for p in problems if p.is_verisure_compatible)

    lines = [
        f"Total problems: {len(problems)}",
        f"Veri-Sure compatible: {compat_count}",
        "",
        "By category:",
    ]
    for cid in sorted(cid_counts):
        label = CID_LABELS.get(cid, cid)
        lines.append(f"  {cid} ({label}): {cid_counts[cid]}")

    lines.append("")
    lines.append("By difficulty:")
    for diff in ["easy", "medium", "hard"]:
        if diff in diff_counts:
            lines.append(f"  {diff}: {diff_counts[diff]}")

    return "\n".join(lines)
