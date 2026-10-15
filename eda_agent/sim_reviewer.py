from __future__ import annotations

import os
import re
import shlex
import shutil
from typing import Tuple

from .bash_tools import CommandResult, run_bash_command


def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"Required executable '{name}' not found in PATH.")


def _verilator_has_fatal(stdout: str, stderr: str) -> bool:
    # Verilator typically reports as "%Error:" / "%Error-<TAG>:".
    fatal_re = re.compile(r"^%Error", re.MULTILINE)
    if fatal_re.search(stderr) or fatal_re.search(stdout):
        return True
    # Some toolchains may prefix with "Error:" without the % marker.
    text = f"{stdout}\n{stderr}".lower()
    return ("syntax error" in text) or ("error:" in text)


def check_syntax(rtl_path: str) -> Tuple[bool, str]:
    _require_executable("verilator")
    cmd = f"verilator --lint-only --sv --timing -Wall -Wno-fatal --assert {shlex.quote(rtl_path)}"
    is_pass, sim_output = run_bash_command(cmd, timeout=60)
    sim_output_obj = CommandResult.model_validate_json(sim_output)
    stdout = sim_output_obj.stdout or ""
    stderr = sim_output_obj.stderr or ""

    # Be permissive about warnings: syntax check should fail only on actual errors.
    is_pass = bool(is_pass) and not _verilator_has_fatal(stdout, stderr)
    return is_pass, sim_output


def sim_review_mismatch_cnt(stdout: str) -> int:
    mismatch_cnt = 0
    if "SIMULATION FAILED" in stdout:
        re_str = r"SIMULATION FAILED - (\d*) MISMATCHES DETECTED"
        m = re.search(re_str, stdout)
        if m is not None:
            mismatch_cnt = int(m.group(1))
            return mismatch_cnt
    # Fallback for VerilogEval-style golden testbenches.
    m2 = re.search(r"^Mismatches:\s*(\d+)\s*in\s*(\d+)\s*samples$", stdout, re.MULTILINE)
    if m2:
        mismatch_cnt = int(m2.group(1))
    return mismatch_cnt


def sim_review(
    output_path_per_run: str,
    golden_rtl_path: str | None = None,
) -> Tuple[bool, int, str]:
    _require_executable("verilator")
    rtl_path = f"{output_path_per_run}/rtl.sv"
    sim_bin = f"{output_path_per_run}/sim_output.bin"
    tb_path = f"{output_path_per_run}/tb.sv"
    if golden_rtl_path is None:
        golden_rtl_path = ""
    if os.path.isfile(sim_bin):
        os.remove(sim_bin)

    # Build + run with Verilator.
    # Notes:
    # - Use --timing to support delays/event controls in testbenches.
    # - Use --trace so $dumpfile/$dumpvars can generate wave.vcd for debugging.
    # - Use --assert (or default-on in newer versions) for assertion support.
    srcs = " ".join(shlex.quote(p) for p in (tb_path, rtl_path, golden_rtl_path) if p)
    cmd = (
        "verilator --binary -j 0 --sv --timing --trace --assert -Wall -Wno-fatal "
        f"--Mdir obj_dir -o {shlex.quote(sim_bin)} {srcs}; "
        f"{shlex.quote(sim_bin)}"
    )
    cmd_ok, sim_output = run_bash_command(cmd, timeout=120, cwd=output_path_per_run)
    sim_output_obj = CommandResult.model_validate_json(sim_output)
    stdout = sim_output_obj.stdout or ""
    stderr = sim_output_obj.stderr or ""

    mismatch_cnt = sim_review_mismatch_cnt(stdout)
    if mismatch_cnt == 0 and re.search(r"assertion failed", f"{stdout}\n{stderr}", re.IGNORECASE):
        # Treat assertion failures as mismatches so the debugger loop can engage.
        mismatch_cnt = 1

    # Determine pass/fail primarily from the testbench's explicit result markers,
    # instead of the process return code. Some testbenches may exit non-zero
    # despite printing "SIMULATION PASSED" (or "Mismatches: 0 ...").
    has_pass_marker = "SIMULATION PASSED" in stdout
    has_mismatch_summary = bool(
        re.search(r"^Mismatches:\s*\d+\s*in\s*\d+\s*samples$", stdout, re.MULTILINE)
    )
    # Only treat *harness* timeouts as fatal. Many VerilogEval testbenches print
    # "TIMEOUT" as a watchdog message even for successful runs (they still emit
    # the "Mismatches: ..." summary). The harness timeout is surfaced in stderr
    # by `run_bash_command` as "Timeout ... reached.".
    has_timeout_marker = "Timeout" in stderr

    # Treat warnings as non-fatal; only block on clear compile/runtime errors.
    has_fatal = _verilator_has_fatal(stdout, stderr)

    is_pass_by_log = has_pass_marker or (has_mismatch_summary and mismatch_cnt == 0)
    is_pass = is_pass_by_log and not (has_timeout_marker or has_fatal)

    # If the command timed out or crashed without printing any summary, keep it failed.
    if not is_pass and not (has_pass_marker or has_mismatch_summary) and cmd_ok:
        # cmd_ok with no markers is ambiguous; treat as failed and keep mismatch_cnt as-is.
        pass

    assert isinstance(sim_output, str) and isinstance(is_pass, bool)
    return is_pass, mismatch_cnt, sim_output


class SimReviewer:
    def __init__(
        self,
        output_path_per_run: str,
        golden_rtl_path: str | None = None,
    ):
        self.output_path_per_run = output_path_per_run
        self.golden_rtl_path = golden_rtl_path

    def review(self) -> Tuple[bool, int, str]:
        return sim_review(
            self.output_path_per_run,
            self.golden_rtl_path,
        )
