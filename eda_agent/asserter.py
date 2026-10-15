from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shlex
import shutil
from typing import Any, Iterable, Literal, Sequence

from .asserter_agent import AsserterAgent, AsserterSpec
from .bash_tools import CommandResult, run_bash_command
from .config import OpenAIConfig
from .utils import clip_text


AsserterStatus = Literal["pass", "fail", "skip", "error"]


@dataclass(frozen=True)
class ContractPort:
    name: str
    direction: Literal["input", "output", "inout"]
    width: int


@dataclass(frozen=True)
class AsserterResult:
    status: AsserterStatus
    summary: str
    asserter_dir: str | None = None
    report_path: str | None = None
    assertion_sv_path: str | None = None
    sim_output_path: str | None = None
    assertion_failures: list[dict[str, Any]] | None = None
    error: str | None = None


def _as_int(val: Any) -> int | None:
    try:
        return int(val)
    except Exception:  # noqa: BLE001
        return None


def _parse_contract(contract_json: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(contract_json)
    except Exception:  # noqa: BLE001
        return None
    return obj if isinstance(obj, dict) else None


def _extract_ports(contract: dict[str, Any]) -> list[ContractPort]:
    io = contract.get("io")
    if not isinstance(io, list):
        return []
    ports: list[ContractPort] = []
    for p in io:
        if not isinstance(p, dict):
            continue
        name = p.get("name")
        direction = p.get("dir")
        width = _as_int(p.get("width", 1))
        if not isinstance(name, str) or not name.strip():
            continue
        if direction not in {"input", "output", "inout"}:
            continue
        if width is None or width <= 0:
            continue
        ports.append(ContractPort(name=name.strip(), direction=direction, width=width))
    return ports


def _sv_range(width: int) -> str:
    return "" if int(width) == 1 else f" [{int(width) - 1}:0]"


def _render_assert_module_header(*, ports: Sequence[ContractPort]) -> str:
    lines: list[str] = ["module AssertModule("]
    for i, p in enumerate(ports):
        comma = "," if i < len(ports) - 1 else ""
        # Observe everything as input ports (bind connects by name).
        lines.append(f"  input  logic{_sv_range(p.width)} {p.name}{comma}")
    lines.append(");")
    return "\n".join(lines) + "\n"


def _infer_clock_port(ports: Sequence[ContractPort]) -> str | None:
    inputs = [p.name for p in ports if p.direction == "input"]
    if "clk" in inputs:
        return "clk"
    for cand in inputs:
        if cand.lower().endswith("clk") or cand.lower().startswith("clk"):
            return cand
    for cand in inputs:
        if "clock" in cand.lower():
            return cand
    return None


def _infer_reset_port(ports: Sequence[ContractPort]) -> str | None:
    inputs = [p.name for p in ports if p.direction == "input"]
    for cand in inputs:
        if cand.lower() in {"rst", "reset"}:
            return cand
    for cand in inputs:
        if "reset" in cand.lower() or cand.lower().endswith("rst"):
            return cand
    return None


def _clocking_info(contract: dict[str, Any], *, ports: Sequence[ContractPort]) -> dict[str, Any]:
    clocking = contract.get("clocking")
    clk_name = None
    clk_edge = "posedge"
    is_seq = False
    rst_name = None
    rst_active = None
    rst_type = None

    if isinstance(clocking, dict):
        is_seq = bool(clocking.get("is_sequential") is True)
        clk = clocking.get("clock")
        if isinstance(clk, dict) and isinstance(clk.get("name"), str) and clk.get("name"):
            clk_name = str(clk.get("name"))
        if isinstance(clk, dict) and isinstance(clk.get("edge"), str) and clk.get("edge") in {"posedge", "negedge"}:
            clk_edge = str(clk.get("edge"))

        rst = clocking.get("reset")
        if isinstance(rst, dict) and isinstance(rst.get("name"), str) and rst.get("name"):
            rst_name = str(rst.get("name"))
        if isinstance(rst, dict) and isinstance(rst.get("active"), str) and rst.get("active") in {"high", "low"}:
            rst_active = str(rst.get("active"))
        if isinstance(rst, dict) and isinstance(rst.get("type"), str) and rst.get("type") in {"synchronous", "asynchronous"}:
            rst_type = str(rst.get("type"))

    if clk_name is None:
        clk_name = _infer_clock_port(ports)
    if rst_name is None:
        rst_name = _infer_reset_port(ports)

    return {
        "is_sequential": is_seq,
        "clock": {"name": clk_name, "edge": clk_edge},
        "reset": {"name": rst_name, "active": rst_active, "type": rst_type},
    }


def _reset_active_expr(info: dict[str, Any]) -> str | None:
    rst = info.get("reset")
    if not isinstance(rst, dict):
        return None
    name = rst.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    active = rst.get("active")
    if active == "low":
        return f"({name} == 1'b0)"
    # default active-high
    return f"({name} == 1'b1)"


def _outputs_with_latency(contract: dict[str, Any]) -> dict[str, int]:
    timing = contract.get("timing")
    out: dict[str, int] = {}
    if not isinstance(timing, dict):
        return out
    for k, v in timing.items():
        if not isinstance(k, str) or not k.strip():
            continue
        if not isinstance(v, dict):
            continue
        lat = _as_int(v.get("latency_cycles"))
        if lat is None:
            continue
        out[k.strip()] = lat
    return out


def _select_monitored_outputs(
    *,
    contract: dict[str, Any],
    ports: Sequence[ContractPort],
    target_outputs: Sequence[str] | None,
) -> tuple[list[str], dict[str, Any]]:
    outputs = [p.name for p in ports if p.direction == "output"]
    lat = _outputs_with_latency(contract)
    explicit_registered = sorted([o for o in outputs if lat.get(o, 0) > 0])

    targets_norm = [s for s in (target_outputs or []) if isinstance(s, str) and s.strip()]
    targets_norm = [s.strip() for s in targets_norm if s.strip() in outputs]

    if explicit_registered:
        # Prefer contract-explicit sequential outputs.
        chosen = explicit_registered
        if targets_norm:
            chosen = [o for o in explicit_registered if o in set(targets_norm)] or explicit_registered
        return chosen, {
            "mode": "explicit_registered_outputs",
            "latency_by_output": lat,
            "targets_requested": targets_norm,
            "selected": chosen,
        }

    clocking = contract.get("clocking")
    is_seq = bool(isinstance(clocking, dict) and clocking.get("is_sequential") is True)
    if is_seq and targets_norm:
        return sorted(set(targets_norm)), {
            "mode": "fallback_to_failing_outputs",
            "reason": "contract is sequential but no outputs have explicit latency_cycles>0; using failing outputs.",
            "latency_by_output": lat,
            "selected": sorted(set(targets_norm)),
        }

    return [], {
        "mode": "none",
        "reason": "No outputs selected for sequential monitoring (no explicit latency_cycles>0 and no fallback targets).",
        "latency_by_output": lat,
    }


def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"Required executable '{name}' not found in PATH.")


def _verilator_has_fatal(stdout: str, stderr: str) -> bool:
    fatal_re = re.compile(r"^%Error", re.MULTILINE)
    if fatal_re.search(stderr) or fatal_re.search(stdout):
        return True
    text = f"{stdout}\n{stderr}".lower()
    return ("syntax error" in text) or ("error:" in text)


_ASSERTER_FAIL_RE = re.compile(r"ASSERTER_FAIL:\s*(?P<id>\S+)\s+t=(?P<t>\S+)\s*(?P<msg>.*)$")


def _extract_failures(stdout: str, stderr: str, *, max_items: int = 50) -> list[dict[str, Any]]:
    lines = (stdout or "") + "\n" + (stderr or "")
    out: list[dict[str, Any]] = []
    for line in lines.splitlines():
        if "ASSERTER_FAIL:" not in line:
            continue
        m = _ASSERTER_FAIL_RE.search(line.strip())
        if not m:
            out.append({"raw": line.strip()})
        else:
            out.append({"id": m.group("id"), "t": m.group("t"), "msg": m.group("msg").strip()})
        if len(out) >= max(1, int(max_items)):
            break

    # Also capture raw Verilator assertion failures if present (best-effort).
    for line in lines.splitlines():
        if re.search(r"assertion failed", line, re.IGNORECASE):
            out.append({"id": "VERILATOR_ASSERT", "raw": line.strip()})
            if len(out) >= max(1, int(max_items)):
                break
    return out


def _clocking_summary_text(info: dict[str, Any], *, monitored_outputs: Sequence[str], latency_by_output: dict[str, int]) -> str:
    clk = info.get("clock") if isinstance(info.get("clock"), dict) else {}
    rst = info.get("reset") if isinstance(info.get("reset"), dict) else {}
    clk_name = clk.get("name")
    clk_edge = clk.get("edge")
    rst_name = rst.get("name")
    rst_active = rst.get("active")
    rst_type = rst.get("type")

    lines: list[str] = [
        f"- clock: name={clk_name!r}, edge={clk_edge!r}",
        f"- reset: name={rst_name!r}, active={rst_active!r}, type={rst_type!r}",
    ]
    if monitored_outputs:
        lines.append("- monitored_outputs:")
        for o in monitored_outputs:
            lines.append(f"  - {o} (latency_cycles={latency_by_output.get(o)!r})")
    return "\n".join(lines) + "\n"


def _render_asserter_sv(
    *,
    dut_module_name: str,
    module_header: str,
    clk_name: str,
    clk_edge: str,
    reset_active_expr: str | None,
    monitored_outputs: Sequence[ContractPort],
    llm_body: str,
) -> str:
    if clk_edge not in {"posedge", "negedge"}:
        clk_edge = "posedge"
    reset_expr = reset_active_expr or "1'b0"

    decls: list[str] = []
    decls.append("  localparam int unsigned ASSERTER_MAX_FAILS = 40;")
    decls.append("  int unsigned asserter_fail_count = 0;")
    decls.append("  time asserter_last_edge_time = 0;")
    decls.append("  bit asserter_have_seen_edge = 0;")
    decls.append("  bit asserter_active = 0;")
    decls.append("")
    decls.append("  task automatic asserter_log(string id, string msg);")
    decls.append("    if (asserter_fail_count < ASSERTER_MAX_FAILS) begin")
    decls.append('      $display("ASSERTER_FAIL: %s t=%0t %s", id, $time, msg);')
    decls.append("    end")
    decls.append("    asserter_fail_count++;")
    decls.append("  endtask")
    decls.append("")
    decls.append("  logic asserter_reset_active;")
    decls.append(f"  assign asserter_reset_active = {reset_expr};")
    decls.append("")
    decls.append(f"  always @({clk_edge} {clk_name}) begin")
    decls.append("    // Use blocking assignments so output-change monitors see the updated edge time immediately.")
    decls.append("    asserter_last_edge_time = $time;")
    decls.append("    asserter_have_seen_edge = 1'b1;")
    decls.append("    if (asserter_reset_active) asserter_active = 1'b0;")
    decls.append("    else asserter_active = 1'b1;")
    decls.append("  end")

    llm_block = llm_body.strip()
    if llm_block:
        llm_block = llm_block.rstrip() + "\n"
    else:
        llm_block = ""

    monitors: list[str] = ["", "  // Baseline timing monitors (contract-guided, best-effort)."]
    for p in monitored_outputs:
        # Only check outputs (they are inputs in AssertModule), but keep widths for formatting.
        fmt = "%0b" if int(p.width) == 1 else "%0h"
        monitors.append(f"  always @({p.name}) begin")
        monitors.append("    if (asserter_active && asserter_have_seen_edge && ($time != asserter_last_edge_time)) begin")
        monitors.append(
            f'      asserter_log("OUT_CHANGE_OFF_EDGE", $sformatf("sig={p.name} val={fmt} last_edge=%0t", {p.name}, asserter_last_edge_time));'
        )
        monitors.append("    end")
        monitors.append("  end")
        monitors.append("")
        monitors.append(f"  always @({clk_edge} {clk_name}) begin")
        monitors.append("    if (asserter_active && asserter_have_seen_edge) begin")
        monitors.append(
            f'      if ((^{p.name}) === 1\'bX) asserter_log("X_ON_OUTPUT", $sformatf("sig={p.name} val={fmt}", {p.name}));'
        )
        monitors.append("    end")
        monitors.append("  end")

    monitors_text = "\n".join(monitors).rstrip() + "\n"
    header = module_header.rstrip() + "\n"
    return (
        header
        + "\n".join(decls).rstrip()
        + "\n\n  // === LLM-generated assertions (may be approximate; non-authoritative) ===\n"
        + llm_block
        + monitors_text
        + "endmodule\n\n"
        + f"bind {dut_module_name} AssertModule asserter_inst(.*);\n"
    )


class Asserter:
    def __init__(
        self,
        cfg: OpenAIConfig,
        *,
        sim_timeout_s: int = 120,
    ) -> None:
        self._cfg = cfg
        self._agent: AsserterAgent | None = None
        self.sim_timeout_s = int(sim_timeout_s)

    def _ensure_agent(self) -> AsserterAgent:
        if self._agent is None:
            self._agent = AsserterAgent(self._cfg)
        return self._agent

    async def analyze(
        self,
        *,
        contract_json: str,
        rtl_path: str,
        tb_path: str,
        output_dir: str,
        golden_rtl_path: str | None = None,
        target_outputs: Sequence[str] | None = None,
        assert_body_override: str | None = None,
    ) -> AsserterResult:
        contract = _parse_contract(contract_json)
        proof_root = Path(output_dir) / "asserter"
        proof_root.mkdir(parents=True, exist_ok=True)
        report_path = proof_root / "asserter_report.json"

        if contract is None:
            res = AsserterResult(status="error", summary="Invalid contract JSON.", report_path=str(report_path), error="Invalid JSON")
            report_path.write_text(json.dumps(asdict(res), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return res

        ports = _extract_ports(contract)
        if not ports:
            res = AsserterResult(status="error", summary="Invalid contract: missing ports.", report_path=str(report_path), error="Missing ports")
            report_path.write_text(json.dumps(asdict(res), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return res

        dut_module_name = str(contract.get("module_name") or "TopModule")
        info = _clocking_info(contract, ports=ports)
        clk = info.get("clock") if isinstance(info.get("clock"), dict) else {}
        clk_name = clk.get("name")
        clk_edge = clk.get("edge") or "posedge"

        if not isinstance(clk_name, str) or not clk_name.strip():
            res = AsserterResult(
                status="skip",
                summary="Skipped asserter: no clock name available in contract and could not infer one.",
                report_path=str(report_path),
            )
            report_path.write_text(json.dumps(asdict(res), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return res

        monitored_names, selection_meta = _select_monitored_outputs(contract=contract, ports=ports, target_outputs=target_outputs)
        if not monitored_names:
            res = AsserterResult(
                status="skip",
                summary="Skipped asserter: no outputs selected for sequential/timing monitoring.",
                report_path=str(report_path),
            )
            report = {"result": asdict(res), "selection": selection_meta, "clocking_info": info}
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return res

        monitored_ports = [p for p in ports if p.direction == "output" and p.name in set(monitored_names)]
        module_header = _render_assert_module_header(ports=ports)
        latency_by_output = _outputs_with_latency(contract)
        clocking_summary = _clocking_summary_text(info, monitored_outputs=monitored_names, latency_by_output=latency_by_output)

        # Snapshot artifacts (RTL/TB may be edited later by Debugger).
        try:
            rtl_text = Path(rtl_path).read_text(encoding="utf-8")
            tb_text = Path(tb_path).read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            res = AsserterResult(
                status="error",
                summary="Failed to read rtl.sv or tb.sv for asserter run.",
                report_path=str(report_path),
                error=f"{type(e).__name__}: {e}",
            )
            report_path.write_text(json.dumps(asdict(res), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return res

        (proof_root / "rtl.sv").write_text(rtl_text, encoding="utf-8")
        (proof_root / "tb.sv").write_text(tb_text, encoding="utf-8")

        golden_sv_arg = ""
        golden_sv_copy_path: str | None = None
        if golden_rtl_path and isinstance(golden_rtl_path, str):
            try:
                gp = Path(golden_rtl_path)
                if gp.is_file():
                    golden_text = gp.read_text(encoding="utf-8")
                    dst = proof_root / "golden_rtl_blackbox.sv"
                    dst.write_text(golden_text, encoding="utf-8")
                    golden_sv_copy_path = str(dst)
                    golden_sv_arg = " golden_rtl_blackbox.sv"
            except Exception:  # noqa: BLE001
                golden_sv_arg = ""
                golden_sv_copy_path = None

        reset_expr = _reset_active_expr(info)

        spec: AsserterSpec
        if assert_body_override is not None:
            spec = AsserterSpec(reasoning="(assert_body_override)", assert_body=assert_body_override)
        elif self._cfg.api_key is None:
            spec = AsserterSpec(reasoning="(no api_key; baseline-only asserter)", assert_body="")
        else:
            agent = self._ensure_agent()
            spec = await agent.chat(
                contract_json=clip_text(contract_json, max_chars=20000),
                module_header=module_header,
                clocking_summary=clocking_summary,
                target_outputs=list(monitored_names),
            )

        # Render asserter bind module.
        asserter_sv_path = proof_root / "asserter.sv"
        asserter_sv = _render_asserter_sv(
            dut_module_name=dut_module_name,
            module_header=module_header,
            clk_name=str(clk_name),
            clk_edge=str(clk_edge),
            reset_active_expr=reset_expr,
            monitored_outputs=monitored_ports,
            llm_body=spec.assert_body,
        )
        asserter_sv_path.write_text(asserter_sv, encoding="utf-8")
        (proof_root / "assert_body_reasoning.txt").write_text(spec.reasoning.strip() + "\n", encoding="utf-8")

        # Run Verilator simulation in the isolated asserter folder.
        try:
            _require_executable("verilator")
        except Exception as e:  # noqa: BLE001
            res = AsserterResult(
                status="error",
                summary="Missing Verilator toolchain.",
                report_path=str(report_path),
                asserter_dir=str(proof_root),
                assertion_sv_path=str(asserter_sv_path),
                error=str(e),
            )
            report_path.write_text(json.dumps(asdict(res), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return res

        # Use an absolute output path for the binary so we can reliably run it
        # even when Verilator places build artifacts under --Mdir.
        sim_bin = str((proof_root / "sim_output.bin").resolve())
        cmd = (
            "verilator --binary -j 0 --sv --timing --trace --assert -Wall -Wno-fatal "
            f"--Mdir obj_dir -o {shlex.quote(sim_bin)} tb.sv rtl.sv asserter.sv{golden_sv_arg} && "
            f"{shlex.quote(sim_bin)}"
        )
        cmd_ok, out_json = await asyncio.to_thread(
            run_bash_command,
            cmd,
            self.sim_timeout_s,
            cwd=str(proof_root),
        )
        sim_output_path = proof_root / "asserter_sim_output.json"
        sim_output_path.write_text(out_json, encoding="utf-8")
        out_obj = CommandResult.model_validate_json(out_json)
        stdout = out_obj.stdout or ""
        stderr = out_obj.stderr or ""

        failures = _extract_failures(stdout, stderr, max_items=60)
        has_fatal = _verilator_has_fatal(stdout, stderr)

        if has_fatal and not failures:
            # If the LLM-generated body caused a compile error, rerun baseline-only once.
            if spec.assert_body.strip():
                baseline_sv = _render_asserter_sv(
                    dut_module_name=dut_module_name,
                    module_header=module_header,
                    clk_name=str(clk_name),
                    clk_edge=str(clk_edge),
                    reset_active_expr=reset_expr,
                    monitored_outputs=monitored_ports,
                    llm_body="",
                )
                (proof_root / "asserter_baseline_only.sv").write_text(baseline_sv, encoding="utf-8")
                asserter_sv_path.write_text(baseline_sv, encoding="utf-8")
                cmd_ok, out_json = await asyncio.to_thread(
                    run_bash_command,
                    cmd,
                    self.sim_timeout_s,
                    cwd=str(proof_root),
                )
                sim_output_path.write_text(out_json, encoding="utf-8")
                out_obj = CommandResult.model_validate_json(out_json)
                stdout = out_obj.stdout or ""
                stderr = out_obj.stderr or ""
                failures = _extract_failures(stdout, stderr, max_items=60)
                has_fatal = _verilator_has_fatal(stdout, stderr)

        status: AsserterStatus
        if failures:
            status = "fail"
            summary = f"Asserter observed {len(failures)} assertion-trigger lines (see report)."
        elif has_fatal:
            status = "error"
            summary = "Asserter failed to compile/run (fatal Verilator error)."
        else:
            status = "pass"
            summary = "Asserter run completed with no parsed assertion-trigger lines."

        res = AsserterResult(
            status=status,
            summary=summary,
            asserter_dir=str(proof_root),
            report_path=str(report_path),
            assertion_sv_path=str(asserter_sv_path),
            sim_output_path=str(sim_output_path),
            assertion_failures=failures,
            error=None if cmd_ok else "verilator command failed",
        )

        report: dict[str, Any] = {
            "result": asdict(res),
            "contract_module_name": dut_module_name,
            "clocking_info": info,
            "selection": selection_meta,
            "paths": {
                "workdir": str(proof_root),
                "tb_sv": str(proof_root / "tb.sv"),
                "rtl_sv": str(proof_root / "rtl.sv"),
                "golden_rtl_sv": golden_sv_copy_path,
                "asserter_sv": str(asserter_sv_path),
                "sim_output_json": str(sim_output_path),
                "vcd_path": str(proof_root / "wave.vcd") if (proof_root / "wave.vcd").exists() else None,
            },
            "agent_generation": {
                "reasoning": spec.reasoning.strip(),
                "assert_body_excerpt": clip_text(spec.assert_body, max_chars=2000),
                "used_override": assert_body_override is not None,
            },
            "verilator_invocation": {
                "command": cmd,
                "timeout_s": self.sim_timeout_s,
                "cmd_ok": bool(cmd_ok),
                "stdout_excerpt": clip_text(stdout, max_chars=5000),
                "stderr_excerpt": clip_text(stderr, max_chars=5000),
            },
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return res
