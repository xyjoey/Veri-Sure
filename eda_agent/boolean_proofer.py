from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Literal, Sequence

from .bash_tools import CommandResult, run_bash_command
from .boolean_proofer_agent import BooleanProoferAgent, BooleanProoferSpec
from .config import LLMConfig
from .trace_slicer import build_driver_map, dynamic_slice, parse_rtl_blocks
from .utils import clip_text


BooleanProoferStatus = Literal["pass", "fail", "skip", "error"]


@dataclass(frozen=True)
class ContractPort:
    name: str
    direction: Literal["input", "output", "inout"]
    width: int


@dataclass(frozen=True)
class BooleanProoferResult:
    status: BooleanProoferStatus
    summary: str
    proof_dir: str | None = None
    report_path: str | None = None
    trace_vcd_path: str | None = None
    asserted_outputs: list[str] | None = None
    counterexample: dict[str, Any] | None = None
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


def _is_purely_combinational(contract: dict[str, Any]) -> bool:
    clocking = contract.get("clocking")
    if isinstance(clocking, dict) and clocking.get("is_sequential") is True:
        return False
    timing = contract.get("timing")
    if isinstance(timing, dict):
        for _k, v in timing.items():
            if not isinstance(v, dict):
                continue
            lat = _as_int(v.get("latency_cycles"))
            if lat is not None and lat > 0:
                return False
    return True


def _contract_is_sequential(contract: dict[str, Any]) -> bool:
    clocking = contract.get("clocking")
    return bool(isinstance(clocking, dict) and clocking.get("is_sequential") is True)


def _targets_from_contract_timing(
    contract: dict[str, Any], *, outputs: Sequence[str]
) -> tuple[set[str], set[str], set[str]]:
    """Return (explicit_zero_latency, explicit_nonzero_latency, timing_unknown)."""
    timing = contract.get("timing")
    if not isinstance(timing, dict):
        return set(), set(), set(outputs)

    explicit_zero: set[str] = set()
    explicit_nonzero: set[str] = set()
    unknown: set[str] = set()
    for out in outputs:
        tinfo = timing.get(out)
        if not isinstance(tinfo, dict):
            unknown.add(out)
            continue
        lat = _as_int(tinfo.get("latency_cycles"))
        if lat is None:
            unknown.add(out)
        elif lat == 0:
            explicit_zero.add(out)
        elif lat > 0:
            explicit_nonzero.add(out)
    return explicit_zero, explicit_nonzero, unknown


def _infer_pure_comb_outputs_from_rtl(rtl_text: str, *, outputs: Sequence[str], max_depth: int = 6) -> set[str]:
    """Heuristic: outputs whose backward slice contains no seq/latch blocks."""
    blocks = parse_rtl_blocks(rtl_text)
    if not blocks:
        return set()
    drivers = build_driver_map(blocks)

    pure_comb: set[str] = set()
    for out in outputs:
        sl = dynamic_slice(fail_signals=[out], drivers=drivers, max_depth=max(2, int(max_depth)))
        if not sl:
            continue
        if any(b.clocking in {"seq", "latch"} for b in sl):
            continue
        if not any(b.clocking == "comb" for b in sl):
            continue
        pure_comb.add(out)
    return pure_comb


def _select_assert_outputs(
    *,
    contract: dict[str, Any],
    ports: Sequence[ContractPort],
    rtl_text: str,
) -> tuple[list[str], dict[str, Any]]:
    output_names = [p.name for p in ports if p.direction == "output"]

    # If the contract is fully combinational, prove all outputs.
    if _is_purely_combinational(contract):
        targets = sorted(output_names)
        return targets, {
            "mode": "all_outputs",
            "reason": "contract is purely combinational (non-sequential, no nonzero latency).",
            "targets": targets,
        }

    explicit_zero, explicit_nonzero, timing_unknown = _targets_from_contract_timing(contract, outputs=output_names)
    inferred_pure_comb = _infer_pure_comb_outputs_from_rtl(rtl_text, outputs=output_names)

    # Only prove outputs that are plausibly pure combinational w.r.t. primary inputs.
    # - If contract marks an output latency > 0, never prove it here.
    # - For outputs missing timing, fall back to RTL inference.
    targets_set = (set(explicit_zero) | (set(timing_unknown) & set(inferred_pure_comb))) & set(inferred_pure_comb)
    targets_set -= set(explicit_nonzero)
    targets = sorted(targets_set)
    return targets, {
        "mode": "combinational_subset",
        "contract_is_sequential": _contract_is_sequential(contract),
        "timing_explicit_zero": sorted(explicit_zero),
        "timing_explicit_nonzero": sorted(explicit_nonzero),
        "timing_unknown": sorted(timing_unknown),
        "rtl_inferred_pure_comb_outputs": sorted(inferred_pure_comb),
        "targets": targets,
    }


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


def _render_module_header(*, module_name: str, ports: Sequence[ContractPort]) -> str:
    lines: list[str] = [f"module {module_name}("]
    for i, p in enumerate(ports):
        comma = "," if i < len(ports) - 1 else ""
        dir_pad = "input " if p.direction == "input" else ("output" if p.direction == "output" else "inout ")
        # Align like: input  logic [3:0] a,
        lines.append(f"  {dir_pad:6} logic{_sv_range(p.width)} {p.name}{comma}")
    lines.append(");")
    return "\n".join(lines) + "\n"


def _render_spec_module(*, ports: Sequence[ContractPort], spec_body: str) -> str:
    header = _render_module_header(module_name="SpecModule", ports=ports)
    body = spec_body.rstrip() + ("\n" if spec_body and not spec_body.endswith("\n") else "")
    return f"{header}{body}endmodule\n"


def _render_miter_module(
    *,
    dut_module_name: str,
    ports: Sequence[ContractPort],
    assert_outputs: Sequence[str],
) -> str:
    inout_ports = [p for p in ports if p.direction == "inout"]
    if inout_ports:
        names = ", ".join(p.name for p in inout_ports)
        raise ValueError(f"inout ports are not supported for boolean proof: {names}")

    inputs = [p for p in ports if p.direction == "input"]
    outputs = [p for p in ports if p.direction == "output"]
    assert_set = {s.strip() for s in assert_outputs if isinstance(s, str) and s.strip()}

    lines: list[str] = []
    lines.append(_render_module_header(module_name="Miter", ports=inputs).rstrip("\n"))
    for p in outputs:
        rng = _sv_range(p.width)
        lines.append(f"  logic{rng} {p.name}_dut;")
        lines.append(f"  logic{rng} {p.name}_spec;")

    def inst_lines(*, inst_name: str, module_name: str, out_suffix: str) -> list[str]:
        conn: list[str] = []
        for p in ports:
            if p.direction == "output":
                conn.append(f"    .{p.name}({p.name}_{out_suffix})")
            else:
                conn.append(f"    .{p.name}({p.name})")
        return [f"  {module_name} {inst_name}(", ",\n".join(conn), "  );"]

    lines.append("")
    lines.extend(inst_lines(inst_name="dut", module_name=dut_module_name, out_suffix="dut"))
    lines.append("")
    lines.extend(inst_lines(inst_name="spec", module_name="SpecModule", out_suffix="spec"))
    lines.append("")
    lines.append("  always @* begin")
    for p in outputs:
        if p.name in assert_set:
            lines.append(f"    assert({p.name}_dut == {p.name}_spec);")
    lines.append("  end")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


_VCD_VAR_RE = re.compile(
    r"^\$var\s+\S+\s+\d+\s+(?P<code>\S+)\s+(?P<name>\S+)(?:\s+\[[^\]]+\])?\s+\$end\s*$"
)
_VCD_SCOPE_RE = re.compile(r"^\$scope\s+module\s+(?P<name>\S+)\s+\$end\s*$")


def _vcd_parse_defs(vcd_text: str) -> dict[str, str]:
    scope: list[str] = []
    code_to_name: dict[str, str] = {}
    for line in vcd_text.splitlines():
        line = line.strip()
        if not line:
            continue
        mscope = _VCD_SCOPE_RE.match(line)
        if mscope:
            scope.append(mscope.group("name"))
            continue
        if line.startswith("$upscope"):
            if scope:
                scope.pop()
            continue
        mvar = _VCD_VAR_RE.match(line)
        if mvar:
            code = mvar.group("code")
            name = mvar.group("name")
            full = ".".join([*scope, name]) if scope else name
            code_to_name[code] = full
            continue
        if line.startswith("$enddefinitions"):
            break
    return code_to_name


def _vcd_value_changes(vcd_text: str) -> Iterable[tuple[int, str, str]]:
    """Yield (time, code, value_bits_or_scalar) from a VCD text."""
    t = 0
    for raw in vcd_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            try:
                t = int(line[1:])
            except Exception:  # noqa: BLE001
                continue
            continue
        if line.startswith("b"):
            parts = line[1:].split()
            if len(parts) != 2:
                continue
            val, code = parts
            yield t, code, val
            continue
        if line[0] in {"0", "1", "x", "z"} and len(line) >= 2:
            yield t, line[1:], line[0]


def _vcd_pick_signal(code_to_name: dict[str, str], leaf: str, *, prefer_prefix: str | None = None) -> str | None:
    leaf = leaf.strip()
    candidates: list[tuple[str, str]] = []
    for code, full in code_to_name.items():
        last = full.split(".")[-1]
        if last == leaf or last.startswith(f"{leaf}["):
            candidates.append((code, full))
    if not candidates:
        return None

    def rank(item: tuple[str, str]) -> tuple[int, int, str]:
        _code, full = item
        return (full.count("."), len(full), full)

    if prefer_prefix:
        preferred = [it for it in candidates if it[1].startswith(prefer_prefix)]
        if preferred:
            return sorted(preferred, key=rank)[0][0]
    return sorted(candidates, key=rank)[0][0]


def _extract_counterexample_from_vcd(
    vcd_path: Path,
    *,
    input_leaves: Sequence[str],
    output_leaves: Sequence[str],
    t: int = 0,
) -> dict[str, Any] | None:
    try:
        text = vcd_path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None

    defs = _vcd_parse_defs(text)
    if not defs:
        return None

    # Prefer top-level Miter scope (Yosys uses module name as top scope).
    prefer = "Miter."

    codes: dict[str, str] = {}
    for leaf in input_leaves:
        code = _vcd_pick_signal(defs, leaf, prefer_prefix=prefer)
        if code:
            codes[leaf] = code

    out_codes: dict[str, str] = {}
    for leaf in output_leaves:
        for suffix in ["dut", "spec"]:
            k = f"{leaf}_{suffix}"
            code = _vcd_pick_signal(defs, k, prefer_prefix=prefer)
            if code:
                out_codes[k] = code

    if not codes and not out_codes:
        return None

    # Track last-known values up to time t.
    vals: dict[str, str] = {}
    for tt, code, val in _vcd_value_changes(text):
        if tt > t:
            break
        for leaf, want_code in codes.items():
            if code == want_code:
                vals[leaf] = val
        for leaf, want_code in out_codes.items():
            if code == want_code:
                vals[leaf] = val

    return {
        "t": t,
        "inputs": {k: vals.get(k) for k in codes},
        "outputs": {k: vals.get(k) for k in out_codes},
    }


def _find_trace_vcd(proof_dir: Path) -> Path | None:
    engine = proof_dir / "engine_0"
    cand = engine / "trace.vcd"
    if cand.exists():
        return cand
    # Fallback: any vcd under engine_0.
    if engine.exists():
        for p in sorted(engine.glob("*.vcd")):
            return p
    return None


def _read_sby_status(proof_dir: Path) -> str | None:
    status_path = proof_dir / "status"
    if not status_path.exists():
        return None
    try:
        text = status_path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001
        return None
    if not text:
        return None
    return text.split()[0].strip().upper()


def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"Required executable '{name}' not found in PATH.")


class BooleanProofer:
    def __init__(
        self,
        cfg: LLMConfig,
        *,
        sby_timeout_s: int = 120,
    ) -> None:
        self._cfg = cfg
        self._agent: BooleanProoferAgent | None = None
        self.sby_timeout_s = int(sby_timeout_s)

    def _ensure_agent(self) -> BooleanProoferAgent:
        if self._agent is None:
            self._agent = BooleanProoferAgent(self._cfg)
        return self._agent

    async def prove(
        self,
        *,
        contract_json: str,
        rtl_path: str,
        output_dir: str,
        spec_body_override: str | None = None,
    ) -> BooleanProoferResult:
        contract = _parse_contract(contract_json)
        if contract is None:
            return BooleanProoferResult(status="error", summary="Invalid contract JSON.", error="Invalid JSON")

        ports = _extract_ports(contract)
        if not ports:
            return BooleanProoferResult(status="error", summary="Invalid contract: missing ports.", error="Missing ports")

        dut_module_name = str(contract.get("module_name") or "TopModule")
        module_header = _render_module_header(module_name="SpecModule", ports=ports)

        proof_root = Path(output_dir) / "boolean_proof"
        proof_root.mkdir(parents=True, exist_ok=True)
        report_path = proof_root / "boolean_proof_report.json"

        # Snapshot artifacts for reproducibility (RTL may be edited later by Debugger).
        try:
            rtl_text = Path(rtl_path).read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            res = BooleanProoferResult(
                status="error",
                summary="Failed to read rtl.sv.",
                report_path=str(report_path),
                error=f"{type(e).__name__}: {e}",
            )
            report_path.write_text(json.dumps(asdict(res), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return res

        (proof_root / "dut.sv").write_text(rtl_text, encoding="utf-8")

        assert_outputs, target_meta = _select_assert_outputs(contract=contract, ports=ports, rtl_text=rtl_text)
        if not assert_outputs:
            res = BooleanProoferResult(
                status="skip",
                summary="Skipped boolean proof: no combinational outputs selected to prove.",
                report_path=str(report_path),
                asserted_outputs=[],
            )
            report = {
                "result": asdict(res),
                "target_selection": target_meta,
                "contract_module_name": dut_module_name,
                "contract_is_purely_combinational": _is_purely_combinational(contract),
            }
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return res

        spec: BooleanProoferSpec
        if spec_body_override is not None:
            spec = BooleanProoferSpec(reasoning="(spec_body_override)", spec_body=spec_body_override)
        else:
            agent = self._ensure_agent()
            spec = await agent.chat(
                contract_json=clip_text(contract_json, max_chars=20000),
                module_header=module_header,
                target_outputs=assert_outputs,
            )
            if not spec.spec_body:
                res = BooleanProoferResult(
                    status="error",
                    summary="BooleanProofer LLM failed to produce a spec body.",
                    report_path=str(report_path),
                    asserted_outputs=list(assert_outputs),
                    error=spec.reasoning,
                )
                report = {
                    "result": asdict(res),
                    "target_selection": target_meta,
                    "contract_module_name": dut_module_name,
                    "contract_is_purely_combinational": _is_purely_combinational(contract),
                }
                report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return res
        (proof_root / "spec.sv").write_text(_render_spec_module(ports=ports, spec_body=spec.spec_body), encoding="utf-8")
        (proof_root / "miter.sv").write_text(
            _render_miter_module(dut_module_name=dut_module_name, ports=ports, assert_outputs=assert_outputs),
            encoding="utf-8",
        )

        sby_text = "\n".join(
            [
                "[options]",
                "mode prove",
                "depth 1",
                "",
                "[engines]",
                "smtbmc z3",
                "",
                "[script]",
                "read -formal -sv dut.sv spec.sv miter.sv",
                "prep -top Miter",
                "",
                "[files]",
                "dut.sv",
                "spec.sv",
                "miter.sv",
                "",
            ]
        )
        (proof_root / "proof.sby").write_text(sby_text, encoding="utf-8")
        (proof_root / "spec_reasoning.txt").write_text(spec.reasoning.strip() + "\n", encoding="utf-8")

        # Run sby.
        try:
            _require_executable("sby")
            _require_executable("yosys")
            _require_executable("z3")
        except Exception as e:  # noqa: BLE001
            res = BooleanProoferResult(status="error", summary="Missing formal toolchain.", error=str(e), report_path=str(report_path))
            report_path.write_text(json.dumps(asdict(res), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return res

        cmd_ok, out_json = await asyncio.to_thread(
            run_bash_command,
            "sby -f proof.sby",
            self.sby_timeout_s,
            cwd=str(proof_root),
        )
        cmd = CommandResult.model_validate_json(out_json)

        proof_dir = proof_root / "proof"
        status = _read_sby_status(proof_dir)

        inputs = [p.name for p in ports if p.direction == "input"]
        all_outputs = [p.name for p in ports if p.direction == "output"]
        trace_vcd = _find_trace_vcd(proof_dir) if status == "FAIL" else None
        cex = (
            _extract_counterexample_from_vcd(trace_vcd, input_leaves=inputs, output_leaves=assert_outputs)
            if trace_vcd
            else None
        )

        # Build report.
        if status == "PASS":
            result = BooleanProoferResult(
                status="pass",
                summary="Boolean proof PASS: SpecModule matches DUT (for asserted outputs).",
                proof_dir=str(proof_dir),
                report_path=str(report_path),
                asserted_outputs=list(assert_outputs),
            )
        elif status == "FAIL":
            result = BooleanProoferResult(
                status="fail",
                summary="Boolean proof FAIL: SpecModule differs from DUT (see counterexample).",
                proof_dir=str(proof_dir),
                report_path=str(report_path),
                trace_vcd_path=str(trace_vcd) if trace_vcd else None,
                asserted_outputs=list(assert_outputs),
                counterexample=cex,
            )
        else:
            detail = "Unknown status" if status is None else f"status={status}"
            err_excerpt = clip_text((cmd.stderr or "") + "\n" + (cmd.stdout or ""), max_chars=4000).strip()
            result = BooleanProoferResult(
                status="error",
                summary=f"Boolean proof error: {detail}.",
                proof_dir=str(proof_dir) if proof_dir.exists() else None,
                report_path=str(report_path),
                asserted_outputs=list(assert_outputs),
                error=err_excerpt or ("sby failed" if not cmd_ok else "unknown"),
            )

        report: dict[str, Any] = {
            "result": asdict(result),
            "inputs": inputs,
            "asserted_outputs": list(assert_outputs),
            "all_outputs": all_outputs,
            "target_selection": target_meta,
            "contract_module_name": dut_module_name,
            "contract_is_purely_combinational": _is_purely_combinational(contract),
            "spec_generation": {
                "reasoning": spec.reasoning.strip(),
                "spec_body_excerpt": clip_text(spec.spec_body, max_chars=2000),
            },
            "paths": {
                "workdir": str(proof_root),
                "dut_sv": str(proof_root / "dut.sv"),
                "spec_sv": str(proof_root / "spec.sv"),
                "miter_sv": str(proof_root / "miter.sv"),
                "sby": str(proof_root / "proof.sby"),
                "sby_proof_dir": str(proof_dir) if proof_dir.exists() else None,
                "sby_logfile": str(proof_dir / "logfile.txt") if proof_dir.exists() else None,
            },
            "sby_invocation": {
                "command": "sby -f proof.sby",
                "timeout_s": self.sby_timeout_s,
                "stdout_excerpt": clip_text(cmd.stdout or "", max_chars=4000),
                "stderr_excerpt": clip_text(cmd.stderr or "", max_chars=4000),
            },
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return result
