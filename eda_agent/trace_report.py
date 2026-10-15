from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .bash_tools import CommandResult
from .trace_slicer import RtlBlock, build_driver_map, dynamic_slice, parse_rtl_blocks

try:  # optional dependency
    from vcdvcd import VCDVCD  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    VCDVCD = None  # type: ignore[assignment]


_HINT_RE = re.compile(
    r"Hint:\s+Output\s+'(?P<sig>[^']+)'\s+has\s+(?P<cnt>\d+)\s+mismatches\.\s+First mismatch occurred at time\s+(?P<t>\d+)\.",
)
_FAILED_RE = re.compile(r"SIMULATION FAILED - (?P<cnt>\d+)\s+MISMATCHES DETECTED.*FIRST AT TIME\s+(?P<t>\d+)", re.IGNORECASE)
_MISMATCHES_RE = re.compile(r"^Mismatches:\s*(?P<cnt>\d+)\s*in\s*(?P<samples>\d+)\s*samples$", re.MULTILINE)


def _extract_fail_signals_and_time(stdout: str) -> tuple[int | None, list[dict[str, Any]]]:
    fail_outputs: list[dict[str, Any]] = []
    times: list[int] = []
    for m in _HINT_RE.finditer(stdout):
        cnt = int(m.group("cnt"))
        sig = m.group("sig")
        t = int(m.group("t"))
        if cnt > 0:
            fail_outputs.append({"sig": sig, "mismatches": cnt, "time": t})
            times.append(t)

    if times:
        return min(times), fail_outputs

    m2 = _FAILED_RE.search(stdout)
    if m2:
        t = int(m2.group("t"))
        return t, [{"sig": None, "mismatches": int(m2.group("cnt")), "time": t}]

    return None, []


def _extract_total_mismatches(stdout: str) -> int | None:
    m = _MISMATCHES_RE.search(stdout)
    if m:
        return int(m.group("cnt"))
    return None


def _extract_topmodule_ports(rtl_text: str) -> dict[str, list[str]]:
    # Very small parser: extract ports from the first "module ... ( ... );"
    m = re.search(r"\bmodule\b\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(", rtl_text)
    if not m:
        return {"inputs": [], "outputs": []}

    start = m.end()
    end = rtl_text.find(");", start)
    if end == -1:
        return {"inputs": [], "outputs": []}

    portlist = rtl_text[start:end]
    ports: list[tuple[str, str]] = []
    direction: str | None = None
    for seg in portlist.split(","):
        seg = seg.strip().strip("()").strip()
        if not seg:
            continue
        mdir = re.search(r"\b(input|output|inout)\b", seg)
        if mdir:
            direction = mdir.group(1)
        elif direction is None:
            # Non-ANSI port list (names only); directions are declared in the body.
            continue
        # Otherwise a continuation like `input logic [3:0] a, b`: `b` inherits the direction.
        names = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", seg)
        names = [
            n
            for n in names
            if n
            not in {
                "input",
                "output",
                "inout",
                "wire",
                "logic",
                "reg",
                "bit",
                "signed",
                "unsigned",
                "integer",
                "int",
            }
        ]
        if names:
            ports.append((direction, names[-1]))

    inputs = sorted({name for d, name in ports if d == "input"})
    outputs = sorted({name for d, name in ports if d == "output"})
    return {"inputs": inputs, "outputs": outputs}
_ATTR_PREFIX = r"(?:\(\*.*?\*\)\s*)*"


def _extract_module_name_from_rtl(rtl_text: str) -> str | None:
    m = re.search(rf"^\s*{_ATTR_PREFIX}module\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", rtl_text, re.MULTILINE)
    if not m:
        return None
    return m.group(1)


def _extract_module_name_from_contract(output_dir: Path) -> str | None:
    p = output_dir / "contract.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("module_name")
    if not isinstance(name, str) or not name.strip():
        return None
    return name.strip()


def _extract_dut_instance_name(tb_text: str, *, module_name: str) -> str | None:
    # Best-effort: match `<module_name> <inst>(` in the testbench.
    m = re.search(rf"\b{re.escape(module_name)}\b\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", tb_text)
    if m:
        return m.group(1)

    # Fallback: in non-golden mode we usually instantiate as `<something> dut ( ... )`.
    if re.search(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b\s+dut\s*\(", tb_text):
        return "dut"
    return None


def _vcd_find_signal(vcd: Any, leaf: str, *, prefer_substrings: list[str] | None = None) -> str | None:
    leaf = leaf.strip()
    candidates: list[str] = []
    for s in getattr(vcd, "signals", []):
        last = s.split(".")[-1]
        if last == leaf or last.startswith(f"{leaf}["):
            candidates.append(s)
    if not candidates:
        return None
    # Prefer the closest-to-top match to reduce ambiguity (e.g., prefer tb.clk over tb.stim1.clk).
    def rank(sig: str) -> tuple[int, int, str]:
        return (sig.count("."), len(sig), sig)

    if prefer_substrings:
        preferred = [s for s in candidates if any(sub in s for sub in prefer_substrings)]
        if preferred:
            return sorted(preferred, key=rank)[0]
    return sorted(candidates, key=rank)[0]


def _vcd_value_at(tv: list[tuple[int, str]], t: int, *, inclusive: bool = True) -> str | None:
    last: str | None = None
    for tt, vv in tv:
        if tt < t or (inclusive and tt == t):
            last = vv
        else:
            break
    return last


def _normalize_vcd_value(val: str | None) -> str | None:
    if val is None:
        return None
    val = val.strip()
    if not val:
        return None
    # vcdvcd may prefix vectors with 'b'
    if (val.startswith("b") or val.startswith("B")) and len(val) > 1:
        val = val[1:]
    return val.lower()


def _wildcard_match(expected: str | None, actual: str | None) -> bool:
    """Approximate VerilogEval tb_match semantics: X/Z in expected are wildcards."""
    exp = _normalize_vcd_value(expected)
    act = _normalize_vcd_value(actual)
    if exp is None or act is None:
        return False

    n = max(len(exp), len(act))
    exp = exp.rjust(n, "0")
    act = act.rjust(n, "0")

    for e, a in zip(exp, act):
        if e in {"x", "z"}:
            continue
        if a != e:
            return False
    return True


def _build_window_times(vcd: Any, *, t_star: int, k_edges: int = 10) -> list[int]:
    clk_sig = _vcd_find_signal(vcd, "clk")
    if clk_sig:
        tv = vcd[clk_sig].tv  # type: ignore[index]
        edges: list[int] = []
        prev = None
        for tt, vv in tv:
            if prev is not None and vv != prev:
                edges.append(tt)
            prev = vv
        edges = [t for t in edges if t <= t_star]
        if edges:
            keep = edges[-min(len(edges), k_edges) :]
            if t_star not in keep:
                keep.append(t_star)
            return keep

    # Fallback: pick latest change times in the window from all signals.
    times = {t_star}
    for s in getattr(vcd, "signals", [])[:200]:
        try:
            for tt, _vv in vcd[s].tv:  # type: ignore[index]
                if tt <= t_star:
                    times.add(tt)
        except Exception:  # noqa: BLE001
            continue
    return sorted(times)[-min(len(times), k_edges + 2) :]


def build_trace_report(
    *,
    rtl_path: Path,
    sim_log_json: str,
    output_dir: Path,
    max_depth: int = 3,
    window_cycles: int = 10,
    diag_edges: int = 200,
) -> tuple[dict[str, Any], list[RtlBlock]]:
    """Build a structured bug report (waveform + coarse RTL slice)."""
    sim = CommandResult.model_validate_json(sim_log_json)
    stdout = sim.stdout or ""

    rtl_text = rtl_path.read_text(encoding="utf-8")
    ports = _extract_topmodule_ports(rtl_text)
    module_name = _extract_module_name_from_contract(output_dir) or _extract_module_name_from_rtl(rtl_text) or "TopModule"

    fail_time, fail_outputs = _extract_fail_signals_and_time(stdout)
    total_mismatches = _extract_total_mismatches(stdout)

    blocks = parse_rtl_blocks(rtl_text)
    if not blocks:
        # Last-resort fallback: allow the debugger to edit something even if
        # the RTL parser fails (e.g., malformed output). This keeps the tool
        # contract stable: debug can always get at least one block_id.
        lines = rtl_text.splitlines()
        if lines:
            code = rtl_text if rtl_text.endswith("\n") else rtl_text + "\n"
            blocks = [
                RtlBlock(
                    id="A1",
                    kind="always",
                    start_line=1,
                    end_line=len(lines),
                    clocking="unknown",
                    code=code,
                    writes=tuple(ports["outputs"]),
                    reads=tuple(ports["inputs"]),
                )
            ]

    drivers = build_driver_map(blocks)

    fail_signal_names: list[str] = [
        fo["sig"] for fo in fail_outputs if fo.get("sig")
    ]
    if not fail_signal_names:
        fail_signal_names = ports["outputs"]

    suspect_blocks = dynamic_slice(
        fail_signals=fail_signal_names,
        drivers=drivers,
        max_depth=max_depth,
    )
    if not suspect_blocks and blocks:
        suspect_blocks = blocks

    vcd_path = output_dir / "wave.vcd"
    vcd_available = vcd_path.exists() and VCDVCD is not None
    dut_instance: str | None = None
    tb_path = output_dir / "tb.sv"
    if tb_path.exists():
        try:
            dut_instance = _extract_dut_instance_name(tb_path.read_text(encoding="utf-8"), module_name=module_name)
        except Exception:  # noqa: BLE001
            dut_instance = None

    fail_time_star = fail_time
    fail_details: list[dict[str, Any]] = []
    input_window: list[dict[str, Any]] = []
    timing_hints: dict[str, Any] = {}
    alignment_diagnosis: dict[str, Any] = {}

    if vcd_available and fail_time_star is not None:
        vcd = VCDVCD(str(vcd_path), store_tvs=True)  # type: ignore[operator]
        times = _build_window_times(vcd, t_star=fail_time_star, k_edges=window_cycles)

        # Classify clock edges for the window (if clk is available).
        edge_at_time: dict[int, str] = {}
        edge_idx_at_time: dict[int, int] = {}
        edges_in_order: list[tuple[int, str]] = []
        clk_full = _vcd_find_signal(vcd, "clk")
        if clk_full:
            prev = None
            for tt, vv in vcd[clk_full].tv:  # type: ignore[index]
                if prev is not None and vv != prev:
                    if prev != "1" and vv == "1":
                        edge_at_time[tt] = "posedge"
                    elif prev == "1" and vv != "1":
                        edge_at_time[tt] = "negedge"
                    else:
                        edge_at_time[tt] = "edge"
                    edge_idx_at_time[tt] = len(edges_in_order)
                    edges_in_order.append((tt, edge_at_time[tt]))
                prev = vv

        key_signals: list[str] = []
        # include top-level inputs
        key_signals.extend(ports["inputs"])
        # include ref/dut for failing outputs if present
        for sig in fail_signal_names:
            key_signals.append(f"{sig}_ref")
            key_signals.append(f"{sig}_dut")

        # Include likely internal signals (best-effort, only if present in VCD).
        internal_candidates: list[str] = []
        for b in suspect_blocks:
            internal_candidates.extend(list(b.writes))
            internal_candidates.extend(list(b.reads))
        internal_candidates = [
            s
            for s in sorted(set(internal_candidates))
            if s and s not in ports["inputs"] and s not in ports["outputs"]
        ]
        key_signals.extend(internal_candidates[:20])
        internal_candidate_set = set(internal_candidates[:20])

        key_signals = sorted({s for s in key_signals if s})

        def sample_signal(*, leaf: str, t: int) -> str | None:
            prefer_substrings = None
            if dut_instance and leaf in internal_candidate_set:
                prefer_substrings = [f".{dut_instance}."]
            full = _vcd_find_signal(vcd, leaf, prefer_substrings=prefer_substrings)
            if not full:
                return None
            inclusive = (leaf == "clk") or (t == 0)
            try:
                return _vcd_value_at(vcd[full].tv, t, inclusive=inclusive)  # type: ignore[index]
            except Exception:  # noqa: BLE001
                return None

        # fail output expected/actual at t*
        for sig in fail_signal_names:
            exp = sample_signal(leaf=f"{sig}_ref", t=fail_time_star)
            act = sample_signal(leaf=f"{sig}_dut", t=fail_time_star)
            fail_details.append({"sig": sig, "expected": exp, "actual": act})

        # input window values
        for window_idx, t in enumerate(times):
            edge = edge_at_time.get(t)
            row: Dict[str, Any] = {
                "t": t,
                "edge": edge,
                "event": f"@({edge} clk)" if edge in {"posedge", "negedge"} else None,
                "sample_idx": edge_idx_at_time.get(t),
                "window_idx": window_idx,
            }
            for s in key_signals:
                val = sample_signal(leaf=s, t=t)
                if val is not None:
                    row[s] = val
            input_window.append(row)

        # Simple timing hints: detect 1-edge latency between expected and actual.
        rows_by_t = {row["t"]: row for row in input_window if isinstance(row.get("t"), int)}
        if times and fail_time_star in rows_by_t:
            idx = times.index(fail_time_star) if fail_time_star in times else -1
            if idx > 0:
                prev_t = times[idx - 1]
                cur = rows_by_t.get(fail_time_star, {})
                prev = rows_by_t.get(prev_t, {})
                for sig in fail_signal_names:
                    exp_k = f"{sig}_ref"
                    act_k = f"{sig}_dut"
                    exp = cur.get(exp_k)
                    act = cur.get(act_k)
                    exp_prev = prev.get(exp_k)
                    act_prev = prev.get(act_k)
                    hint: dict[str, Any] = {}
                    if exp is not None and act is not None and exp_prev is not None and act_prev is not None:
                        if _wildcard_match(exp_prev, act) and not _wildcard_match(exp, act):
                            hint["actual_matches_prev_expected"] = True
                        if _wildcard_match(exp, act_prev) and not _wildcard_match(exp_prev, act_prev):
                            hint["prev_actual_matches_current_expected"] = True
                    if hint:
                        hint["prev_t"] = prev_t
                        hint["t"] = fail_time_star
                        timing_hints[sig] = hint

        # Alignment diagnosis over a wider (but bounded) window of edge samples.
        if edges_in_order:
            diag_candidates = [e for e in edges_in_order if e[0] <= fail_time_star]
            diag_candidates = diag_candidates[-min(len(diag_candidates), max(20, diag_edges)) :]

            for sig in fail_signal_names:
                exp_leaf = f"{sig}_ref"
                act_leaf = f"{sig}_dut"
                seq: list[dict[str, Any]] = []
                for tt, et in diag_candidates:
                    seq.append(
                        {
                            "t": tt,
                            "edge": et,
                            "exp": sample_signal(leaf=exp_leaf, t=tt),
                            "act": sample_signal(leaf=act_leaf, t=tt),
                        }
                    )

                def rate(pairs: list[tuple[str | None, str | None]]) -> float | None:
                    total = 0
                    ok = 0
                    for e, a in pairs:
                        if e is None or a is None:
                            continue
                        total += 1
                        if _wildcard_match(e, a):
                            ok += 1
                    return (ok / total) if total else None

                exp_seq = [r["exp"] for r in seq]
                act_seq = [r["act"] for r in seq]

                cur_rate = rate(list(zip(exp_seq, act_seq)))
                lag_rate = rate(list(zip(exp_seq[1:], act_seq[:-1])))
                lead_rate = rate(list(zip(exp_seq[:-1], act_seq[1:])))

                pos_rate = rate(
                    [(r["exp"], r["act"]) for r in seq if r.get("edge") == "posedge"]
                )
                neg_rate = rate(
                    [(r["exp"], r["act"]) for r in seq if r.get("edge") == "negedge"]
                )

                choices = {
                    "current": cur_rate,
                    "dut_lag1": lag_rate,
                    "dut_lead1": lead_rate,
                }
                best = max(choices.items(), key=lambda kv: (-1.0 if kv[1] is None else kv[1]))
                alignment_diagnosis[sig] = {
                    "samples_considered": len(seq),
                    "match_rate_current": cur_rate,
                    "match_rate_dut_lag1": lag_rate,
                    "match_rate_dut_lead1": lead_rate,
                    "match_rate_posedge": pos_rate,
                    "match_rate_negedge": neg_rate,
                    "best_alignment": best[0] if best[1] is not None else None,
                }

    report: dict[str, Any] = {
        "fail_time": fail_time_star,
        "total_mismatches": total_mismatches,
        "fail_outputs": fail_details or fail_outputs,
        "input_window": input_window,
        "alignment_diagnosis": alignment_diagnosis,
        "suspect_blocks": [
            {
                "id": b.id,
                "kind": b.kind,
                "clocking": b.clocking,
                "writes": list(b.writes),
                "reads": list(b.reads),
                "start_line": b.start_line,
                "end_line": b.end_line,
                "code_snippet": b.code[:1200],
            }
            for b in sorted(suspect_blocks, key=lambda x: (x.start_line, x.id))
        ],
        "notes": {
            "vcd_path": str(vcd_path) if vcd_path.exists() else None,
            "vcd_available": bool(vcd_available),
            "dut_instance": dut_instance,
            "sampling": {
                "non_clk": "value immediately before time t (to approximate values seen in always@(posedge/negedge) before NBAs)",
                "clk": "value at time t (edge value)",
            },
            "timing_hints": timing_hints,
            "ports": ports,
            "rtl_blocks": len(blocks),
        },
    }

    return report, suspect_blocks
