from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.tool import ToolResponse, Toolkit

from .agents import SafeReActAgent, clear_memory_safely
from .asserter import Asserter
from .boolean_proofer import BooleanProofer
from .config import OpenAIConfig
from .model import make_formatter, make_openai_model
from .sim_reviewer import SimReviewer, check_syntax
from .trace_report import build_trace_report
from .trace_slicer import RtlBlock

logger = logging.getLogger(__name__)

_FAIL_TIME_HINT_RE = re.compile(
    r"Hint:\s+Output\s+'[^']+'\s+has\s+(?P<cnt>\d+)\s+mismatches\.\s+First mismatch occurred at time\s+(?P<t>\d+)\.",
    re.MULTILINE,
)
_FAIL_TIME_FAILED_RE = re.compile(
    r"SIMULATION FAILED - \d+\s+MISMATCHES DETECTED.*FIRST AT TIME\s+(?P<t>\d+)",
    re.IGNORECASE,
)


def _extract_fail_time_from_sim_log_json(sim_log_json: str) -> int | None:
    """Best-effort earliest mismatch time from a CommandResult JSON string."""
    try:
        obj = json.loads(sim_log_json)
    except Exception:  # noqa: BLE001
        return None
    stdout = str(obj.get("stdout") or "")
    times: list[int] = []
    for m in _FAIL_TIME_HINT_RE.finditer(stdout):
        try:
            if int(m.group("cnt")) > 0:
                times.append(int(m.group("t")))
        except Exception:  # noqa: BLE001
            continue
    if times:
        return min(times)
    m2 = _FAIL_TIME_FAILED_RE.search(stdout)
    if m2:
        try:
            return int(m2.group("t"))
        except Exception:  # noqa: BLE001
            return None
    return None


def _clip_text(s: str, *, max_chars: int) -> str:
    if not isinstance(s, str):
        s = str(s)
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    half = max_chars // 2
    return s[:half] + "\n...<snip>...\n" + s[-half:]


def _summarize_sim_log_json(sim_log_json: str, *, max_chars: int = 4000) -> str:
    """Return a compact excerpt from a CommandResult JSON string."""
    stdout = ""
    stderr = ""
    try:
        obj = json.loads(sim_log_json)
        stdout = str(obj.get("stdout") or "")
        stderr = str(obj.get("stderr") or "")
    except Exception:  # noqa: BLE001
        return _clip_text(sim_log_json, max_chars=max_chars)

    # Keep the most informative bits: mismatch banner + hints + summary.
    interesting: list[str] = []
    for line in stdout.splitlines():
        if (
            "=== MISMATCH DETECTED" in line
            or "Hint:" in line
            or line.startswith("Mismatches:")
            or "SIMULATION FAILED" in line
            or "SIMULATION PASSED" in line
            or line.strip() == "TIMEOUT"
        ):
            interesting.append(line)
    out = "\n".join(interesting).strip()
    if not out:
        out = _clip_text(stdout, max_chars=max_chars)

    if stderr.strip():
        out = out + "\n\n[stderr excerpt]\n" + _clip_text(stderr, max_chars=max(800, max_chars // 3))
    return _clip_text(out, max_chars=max_chars)


SYSTEM_PROMPT = r"""
You are Debugger, an expert in RTL debugging.

Goal: use tool calls to minimally edit and re-simulate SystemVerilog RTL so that:
1) The RTL matches the Architect contract (SOURCE OF TRUTH), and
2) The RTL passes the Verifier-generated testbench (or golden benchmark testbench).

Toolchain note:
- Simulation is run with Verilator; errors/warnings in logs may follow Verilator formatting.

Rules:
1. Do not modify the testbench. Only modify the RTL code.
2. Do not try to change or define RefModule. There is RefModule defined elsewhere if needed.
3. Always respect the simulation result. Keep debugging as long as mismatches exist.
4. Prefer small, targeted edits.
5. Preserve the module interface and the contract's timing assumptions.
6. Contract-only mode: do NOT change behavior based on input_spec if it conflicts with the contract.

When you are done (simulation passes), finish by calling:
  generate_response(response="...brief summary...")
The `response` argument MUST be a plain string (not JSON, not a dict, not a list).
"""


INIT_EDITION_PROMPT = r"""
The information below is given to help your work:
1. The Architect contract (JSON) that both TB and RTL MUST follow (SOURCE OF TRUTH);
2. The generated testbench (fixed; you must not modify it);
3. The simulation failure log excerpt (ground truth about mismatches; full log is saved on disk).
4. The input_spec is included only as background; it must NOT override the contract.
<input_spec>
{input_spec}
</input_spec>
<contract_json>
{contract_json}
</contract_json>
<generated_tb>
{generated_tb}
</generated_tb>
<sim_failed_log_excerpt>
{sim_failed_log_excerpt}
</sim_failed_log_excerpt>
{kmap_hint}
"""

KMAP_DEBUG_HINT_PROMPT = r"""
[K-map hint (only if the problem involves Karnaugh maps / K-maps)]:
- Use mismatch inputs from the sim log/trace to identify wrong minterms.
- Fix one mismatch class at a time while preserving all other truth-table entries.
"""

EXTRA_ORDER_PROMPT = r"""
Workflow (repeat until pass):
1) Use the contract + trace report to find the most likely root cause.
2) Check `trace_summary.alignment_diagnosis` first:
   - If it suggests a 1-cycle shift or wrong sampling edge, fix timing/reset/edge issues before changing core logic.
   - Otherwise focus on combinational correctness in the suspect block(s).
3) Call list_suspect_blocks(), then read_block(block_id) for the most relevant one.
4) Make ONE small change in ONE block via replace_block(block_id, new_code).
5) Immediately call run_simulation() and iterate based on the new trace summary.

Rules:
- Do not modify the testbench. Only modify the RTL code.
- Preserve the module interface and the contract's timing assumptions.
- Only modify code inside suspect blocks.
- Keep RTL synthesizable and Verilator-compatible (avoid fancy SVA, no delays).

You will also receive a structured trace-grounded bug report (JSON) that includes:
1) earliest mismatch time, 2) (expected vs actual) values when available, 3) a short input window,
4) a dynamically sliced list of suspect always/assign blocks.

You MUST only modify code inside suspect blocks.
Use tools:
- list_suspect_blocks()
- read_block(block_id)
- replace_block(block_id, new_code)
- run_simulation()

When simulation passes, end with:
  generate_response(response="...brief summary...")
"""


@dataclass
class _EditSession:
    tb_path: str
    rtl_path: str
    output_dir: str
    last_mismatch_cnt: int
    sim_reviewer: SimReviewer
    max_trials: int

    is_done: bool = False
    action_calls: int = 0
    trace_report: Dict[str, Any] | None = None
    blocks_by_id: Dict[str, RtlBlock] | None = None
    last_fail_time: int | None = None

    def read_rtl(self) -> str:
        with open(self.rtl_path, "r", encoding="utf-8") as f:
            return f.read()

    def read_rtl_with_lineno(self) -> str:
        lines = self.read_rtl().splitlines()
        return "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines)) + "\n"

    def write_rtl(self, content: str) -> None:
        with open(self.rtl_path, "w", encoding="utf-8") as f:
            f.write(content)

    def run_simulation(self) -> Dict[str, Any]:
        is_sim_pass, sim_mismatch_cnt, sim_output = self.sim_reviewer.review()
        # Persist full sim output for human inspection; provide excerpt to the agent.
        try:
            Path(self.output_dir, "debug_sim_output.json").write_text(sim_output, encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        return {
            "is_sim_pass": is_sim_pass,
            "sim_mismatch_cnt": sim_mismatch_cnt,
            # Full log for trace refresh; the tool wrapper pops it before replying to the agent.
            "sim_output": sim_output,
            "sim_output_excerpt": _summarize_sim_log_json(sim_output),
            "sim_output_path": str(Path(self.output_dir, "debug_sim_output.json")),
        }

    def list_suspect_blocks(self) -> list[dict[str, Any]]:
        if not self.trace_report:
            return []
        return list(self.trace_report.get("suspect_blocks") or [])

    def read_block(self, block_id: str) -> str:
        if not self.blocks_by_id or block_id not in self.blocks_by_id:
            return f"ERROR: Unknown block_id '{block_id}'."
        block = self.blocks_by_id[block_id]
        lines = block.code.splitlines()
        return "\n".join(f"{block.start_line + i}: {line}" for i, line in enumerate(lines)) + "\n"

    def _refresh_trace(self, *, sim_log_json: str) -> None:
        report, suspect_blocks = build_trace_report(
            rtl_path=Path(self.rtl_path),
            sim_log_json=sim_log_json,
            output_dir=Path(self.output_dir),
        )
        self.trace_report = report
        self.blocks_by_id = {b.id: b for b in suspect_blocks} if suspect_blocks else None
        ft = report.get("fail_time")
        self.last_fail_time = int(ft) if isinstance(ft, int) else None
        (Path(self.output_dir) / "trace_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def trace_report_path(self) -> str:
        return str(Path(self.output_dir) / "trace_report.json")

    def trace_summary(self) -> Dict[str, Any]:
        tr = self.trace_report or {}
        suspects = tr.get("suspect_blocks") or []
        fail_outputs = tr.get("fail_outputs") or []
        alignment = tr.get("alignment_diagnosis") or {}
        timing_hints = ((tr.get("notes") or {}).get("timing_hints")) or {}

        # Compact alignment summary: keep only the best_alignment and a couple of rates per output.
        alignment_compact: Dict[str, Any] = {}
        if isinstance(alignment, dict):
            for sig, info in alignment.items():
                if not isinstance(info, dict):
                    continue
                alignment_compact[str(sig)] = {
                    "best_alignment": info.get("best_alignment"),
                    "match_rate_current": info.get("match_rate_current"),
                    "match_rate_dut_lag1": info.get("match_rate_dut_lag1"),
                    "match_rate_posedge": info.get("match_rate_posedge"),
                    "match_rate_negedge": info.get("match_rate_negedge"),
                    "samples_considered": info.get("samples_considered"),
                }

        return {
            "trace_report_path": self.trace_report_path(),
            "fail_time": tr.get("fail_time"),
            "total_mismatches": tr.get("total_mismatches"),
            "fail_outputs": [
                {"sig": fo.get("sig"), "expected": fo.get("expected"), "actual": fo.get("actual")}
                for fo in fail_outputs
                if isinstance(fo, dict)
            ][:5],
            "suspect_blocks": [
                {
                    "id": b.get("id"),
                    "clocking": b.get("clocking"),
                    "start_line": b.get("start_line"),
                    "end_line": b.get("end_line"),
                    "writes": b.get("writes"),
                }
                for b in suspects
                if isinstance(b, dict)
            ][:8],
            "alignment_diagnosis": alignment_compact,
            "timing_hints": timing_hints,
            "dut_instance": ((tr.get("notes") or {}).get("dut_instance")),
            "vcd_available": ((tr.get("notes") or {}).get("vcd_available")),
        }

    def _base_result(self) -> Dict[str, Any]:
        return {
            "is_action_executed": False,
            "is_syntax_correct": False,
            "syntax_output": "",
            "is_sim_pass": False,
            "sim_mismatch_cnt": 0,
            "sim_output": "",
            "error_msg": "",
        }

    def _judge_replace_action_execution(self, *, old_file_content: str) -> Dict[str, Any]:
        result = self._base_result()
        prev_mismatch_cnt = int(self.last_mismatch_cnt)
        prev_fail_time = self.last_fail_time
        is_syntax_correct, syntax_output = check_syntax(self.rtl_path)
        result["is_syntax_correct"] = is_syntax_correct
        result["syntax_output"] = syntax_output
        if not is_syntax_correct:
            self.write_rtl(old_file_content)
            result["error_msg"] = "Syntax error. Action rolled back."
            return result

        is_sim_pass, sim_mismatch_cnt, sim_output = self.sim_reviewer.review()
        result["is_sim_pass"] = is_sim_pass
        result["sim_mismatch_cnt"] = sim_mismatch_cnt
        try:
            Path(self.output_dir, "debug_sim_output.json").write_text(sim_output, encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        result["sim_output_excerpt"] = _summarize_sim_log_json(sim_output)
        result["sim_output_path"] = str(Path(self.output_dir, "debug_sim_output.json"))
        result["prev_mismatch_cnt"] = prev_mismatch_cnt
        result["prev_fail_time"] = prev_fail_time
        new_fail_time = _extract_fail_time_from_sim_log_json(sim_output)
        result["new_fail_time"] = new_fail_time

        if sim_mismatch_cnt > prev_mismatch_cnt:
            # Sometimes fixing an early-cycle issue can expose additional later-cycle mismatches.
            # Allow a small mismatch increase only if the FIRST mismatch time moves later.
            increase = int(sim_mismatch_cnt) - int(prev_mismatch_cnt)
            max_increase = min(5, max(1, int(0.1 * max(1, int(prev_mismatch_cnt)))))
            allow = (
                (prev_fail_time is not None)
                and (new_fail_time is not None)
                and (new_fail_time > prev_fail_time)
                and (increase <= max_increase)
            )
            if not allow:
                self.write_rtl(old_file_content)
                result["error_msg"] = (
                    "Mismatch_cnt increased after replacement. Action rolled back. "
                    f"(prev={prev_mismatch_cnt}, new={sim_mismatch_cnt}, prev_fail_time={prev_fail_time}, new_fail_time={new_fail_time})"
                )
                return result
            result["accept_reason"] = (
                "Accepted despite slight mismatch increase because earliest mismatch moved later "
                f"(+{increase} mismatches, fail_time {prev_fail_time}->{new_fail_time})."
            )

        if sim_mismatch_cnt == 0 and not is_sim_pass:
            self.write_rtl(old_file_content)
            result["error_msg"] = "Mismatch_cnt is 0 but simulation failed. Action rolled back."
            return result

        # Accept.
        self.last_mismatch_cnt = sim_mismatch_cnt
        if new_fail_time is not None:
            self.last_fail_time = int(new_fail_time)
        result["is_action_executed"] = True
        if is_sim_pass and sim_mismatch_cnt == 0:
            self.is_done = True
        else:
            self._refresh_trace(sim_log_json=sim_output)
            result["trace_summary"] = self.trace_summary()
        return result

    def replace_block(self, block_id: str, new_code: str) -> Dict[str, Any]:
        if not self.blocks_by_id or block_id not in self.blocks_by_id:
            return {
                "is_action_executed": False,
                "error_msg": f"Unknown block_id '{block_id}'. Use list_suspect_blocks() first.",
            }

        old_file_content = self.read_rtl()
        old_lines = old_file_content.splitlines()
        block = self.blocks_by_id[block_id]
        start = block.start_line - 1
        end = block.end_line - 1
        if start < 0 or end >= len(old_lines) or start > end:
            return {
                "is_action_executed": False,
                "error_msg": f"Invalid block range for {block_id}: {block.start_line}-{block.end_line}.",
            }

        self.action_calls += 1
        if self.action_calls > self.max_trials:
            return {
                "is_action_executed": False,
                "error_msg": "Reached maximum debug trials; refusing further edits.",
            }

        new_block_lines = new_code.rstrip("\n").splitlines()
        new_lines = old_lines[:start] + new_block_lines + old_lines[end + 1 :]
        self.write_rtl("\n".join(new_lines) + ("\n" if old_file_content.endswith("\n") else ""))

        return self._judge_replace_action_execution(old_file_content=old_file_content)


class RTLEditor:
    def __init__(self, cfg: OpenAIConfig, *, sim_reviewer: SimReviewer, max_trials: int = 30) -> None:
        self._cfg = cfg
        self.sim_reviewer = sim_reviewer
        self.max_trials = int(max_trials)
        self._session: _EditSession | None = None
        # Kept across chat() calls so their (lazily created) models accumulate token usage.
        self._proofer = BooleanProofer(cfg)
        self._asserter = Asserter(cfg)

        # The method names are the tool names the model sees; keep them in sync with the prompts.
        toolkit = Toolkit()
        toolkit.register_tool_function(self.list_suspect_blocks)
        toolkit.register_tool_function(self.read_block)
        toolkit.register_tool_function(self.replace_block)
        toolkit.register_tool_function(self.run_simulation)

        self._agent = SafeReActAgent(
            name="Debugger",
            sys_prompt=SYSTEM_PROMPT,
            model=make_openai_model(cfg),
            formatter=make_formatter(cfg.model),
            toolkit=toolkit,
            memory=InMemoryMemory(),
            max_iters=10,
        )

    async def reset(self) -> None:
        await clear_memory_safely(self._agent)
        self._session = None

    def helper_models(self) -> list[Any]:
        """Models of the BooleanProofer/Asserter helper agents created so far (for token accounting)."""
        return [h._agent._agent.model for h in (self._proofer, self._asserter) if h._agent is not None]

    async def list_suspect_blocks(self) -> ToolResponse:
        """List dynamically sliced suspect blocks (always/assign)."""
        if self._session is None:
            return ToolResponse(content=[{"type": "text", "text": "ERROR: No active edit session."}])
        blocks = self._session.list_suspect_blocks()
        return ToolResponse(content=[{"type": "text", "text": json.dumps(blocks, indent=2)}])

    async def read_block(self, block_id: str) -> ToolResponse:
        """Read a suspect block by id with line numbers."""
        if self._session is None:
            return ToolResponse(content=[{"type": "text", "text": "ERROR: No active edit session."}])
        return ToolResponse(content=[{"type": "text", "text": self._session.read_block(block_id)}])

    async def run_simulation(self) -> ToolResponse:
        """Run simulation for current rtl.sv + tb.sv; returns pass/fail and mismatch count."""
        if self._session is None:
            return ToolResponse(content=[{"type": "text", "text": "ERROR: No active edit session."}])
        result = await asyncio.to_thread(self._session.run_simulation)
        sim_output = result.pop("sim_output", None)
        # Keep trace_report.json in sync even when the model triggers standalone re-simulations.
        try:
            if (not result.get("is_sim_pass")) and int(result.get("sim_mismatch_cnt") or 0) > 0:
                if isinstance(sim_output, str) and sim_output.strip():
                    await asyncio.to_thread(self._session._refresh_trace, sim_log_json=sim_output)
            if self._session.trace_report:
                result["trace_summary"] = self._session.trace_summary()
        except Exception:  # noqa: BLE001
            # Tooling should never crash due to trace refresh; keep sim result.
            pass
        return ToolResponse(content=[{"type": "text", "text": json.dumps(result, indent=4)}])

    async def replace_block(self, block_id: str, new_code: str) -> ToolResponse:
        """Replace a suspect block by id, then syntax-check + simulate (rollback if mismatch increases)."""
        if self._session is None:
            return ToolResponse(content=[{"type": "text", "text": "ERROR: No active edit session."}])
        result = await asyncio.to_thread(self._session.replace_block, block_id, new_code)
        # Even if the action was rolled back, return the latest available trace pointer/summary
        # so the agent can re-ground itself quickly.
        if self._session.trace_report:
            result.setdefault("trace_summary", self._session.trace_summary())
        return ToolResponse(content=[{"type": "text", "text": json.dumps(result, indent=4)}])

    async def chat(
        self,
        *,
        spec: str,
        output_dir_per_run: str,
        sim_failed_log: str,
        sim_mismatch_cnt: int,
        contract_json: str,
        max_trials: int | None = None,
    ) -> Tuple[bool, str, int]:
        await self.reset()
        tb_path = f"{output_dir_per_run}/tb.sv"
        rtl_path = f"{output_dir_per_run}/rtl.sv"
        session_max_trials = int(max_trials) if max_trials is not None else int(self.max_trials)
        if session_max_trials < 0:
            session_max_trials = 0

        self._session = _EditSession(
            tb_path=tb_path,
            rtl_path=rtl_path,
            output_dir=output_dir_per_run,
            last_mismatch_cnt=sim_mismatch_cnt,
            sim_reviewer=self.sim_reviewer,
            max_trials=session_max_trials,
        )

        with open(tb_path, "r", encoding="utf-8") as f:
            generated_tb = f.read()

        # Save full failed log for inspection, but only send a short excerpt to the agent.
        try:
            (Path(output_dir_per_run) / "sim_failed_log.json").write_text(sim_failed_log, encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        sim_failed_log_excerpt = _summarize_sim_log_json(sim_failed_log)

        needs_kmap_hint = any(
            k in f"{spec}\n{contract_json}".lower()
            for k in ["kmap", "k-map", "karnaugh"]
        )
        init = INIT_EDITION_PROMPT.format(
            input_spec=spec,
            contract_json=contract_json,
            generated_tb=_clip_text(generated_tb, max_chars=8000),
            sim_failed_log_excerpt=sim_failed_log_excerpt,
            kmap_hint=(KMAP_DEBUG_HINT_PROMPT if needs_kmap_hint else ""),
        )
        # Build a trace-grounded report to guide minimal patching.
        report, suspect_blocks = build_trace_report(
            rtl_path=Path(rtl_path),
            sim_log_json=sim_failed_log,
            output_dir=Path(output_dir_per_run),
        )
        (Path(output_dir_per_run) / "trace_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._session.trace_report = report
        self._session.blocks_by_id = {b.id: b for b in suspect_blocks} if suspect_blocks else None
        ft0 = report.get("fail_time")
        self._session.last_fail_time = int(ft0) if isinstance(ft0, int) else None

        # Best-effort: boolean (combinational) equivalence hint from the contract summary.
        # This is NOT a source of truth; treat it only as parallel guidance.
        boolean_hint = ""
        try:
            proof_res = await self._proofer.prove(
                contract_json=contract_json,
                rtl_path=rtl_path,
                output_dir=output_dir_per_run,
            )
            boolean_hint = (
                "<boolean_proof_result_json>\n"
                + json.dumps(asdict(proof_res), indent=2, ensure_ascii=False)
                + "\n</boolean_proof_result_json>\n"
            )
        except Exception:  # noqa: BLE001
            boolean_hint = ""

        # Best-effort: assertion-based (sequential/timing) hint in a separate sim sandbox.
        # This is NOT a source of truth; treat it only as parallel guidance.
        asserter_hint = ""
        try:
            fail_sigs: list[str] = []
            for fo in (report.get("fail_outputs") or []):
                if isinstance(fo, dict) and isinstance(fo.get("sig"), str) and fo.get("sig"):
                    fail_sigs.append(str(fo.get("sig")))
            fail_sigs = sorted(set(fail_sigs))

            asserter_res = await self._asserter.analyze(
                contract_json=contract_json,
                rtl_path=rtl_path,
                tb_path=tb_path,
                output_dir=output_dir_per_run,
                golden_rtl_path=getattr(self.sim_reviewer, "golden_rtl_path", None),
                target_outputs=fail_sigs,
            )
            asserter_hint = (
                "<asserter_result_json>\n"
                + json.dumps(asdict(asserter_res), indent=2, ensure_ascii=False)
                + "\n</asserter_result_json>\n"
            )
        except Exception:  # noqa: BLE001
            asserter_hint = ""

        first_prompt = (
            f"{init}\n\n<trace_report_json>\n{json.dumps(report, indent=2, ensure_ascii=False)}\n</trace_report_json>\n\n"
            f"{boolean_hint}\n{asserter_hint}\n{EXTRA_ORDER_PROMPT}\n\n"
            "Start by calling list_suspect_blocks(), then read_block(block_id) for the most relevant one, "
            "then apply one replace_block(block_id, new_code), then run_simulation()."
        )
        try:
            (Path(output_dir_per_run) / "debugger_prompt.txt").write_text(first_prompt + "\n", encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        await self._agent(Msg("user", first_prompt, role="user"))

        for _ in range(session_max_trials):
            if self._session.is_done:
                break
            await self._agent(
                Msg(
                    "user",
                    "Continue debugging. Preserve the contract and module interface. If mismatches remain, pick 1 suspect block and call read_block(block_id), then call replace_block(block_id, new_code) once, then run_simulation().",
                    role="user",
                )
            )

        with open(rtl_path, "r", encoding="utf-8") as f:
            rtl_code = f.read()
        used = int(getattr(self._session, "action_calls", 0) or 0)
        return self._session.is_done, rtl_code, used
