from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from pydantic import BaseModel

from .agents import SafeReActAgent, clear_memory_safely
from .config import LLMConfig
from .model import make_formatter, make_chat_model
from .prompts import ORDER_PROMPT
from .utils import extract_json_object


SYSTEM_PROMPT = r"""
You are Architect, a senior hardware design lead for SystemVerilog RTL.

Goal: turn an ambiguous natural-language spec into a precise, testable "contract"
that other agents (Verifier, Coder, Debugger, Formal) will follow.

Principles:
- Be concrete: define timing/latency, reset behavior, and edge semantics when possible.
- If the spec is ambiguous, pick the smallest reasonable assumption and record it explicitly.
- Never change the interface unless the spec (or golden testbench) demands it.
- Keep the contract compact and machine-usable.
- IMPORTANT: In contract-only mode, downstream agents will treat the contract as the ONLY source of truth.
  Therefore, the contract must be complete (interface, timing, corner cases) and internally consistent.

Toolchain note:
- The simulation harness uses Verilator. When adding verifier guidance, prefer procedural checks or simple assertions
  (avoid requiring advanced SVA features that may not be supported).
"""


CONTRACT_PROMPT = r"""
Create a design contract for the DUT described in <input_spec>.

The contract will be consumed by:
- Verifier: generates a testbench consistent with the contract
- Coder: writes RTL consistent with the contract
- Debugger: fixes RTL while preserving the contract
- Formal: generates miter/spec properties from the contract

Requirements:
1) Output MUST be a single valid JSON object (no extra text).
2) Write everything in English.
3) Prefer facts from the spec. If ambiguous, add an explicit assumption.
4) If <golden_testbench> is provided, treat its interface/timing expectations as ground truth and align the contract to it.

What to include (keep it compact):
- source_of_truth: "spec" or "golden_tb"
- module_name: usually "TopModule" unless spec/golden TB indicates otherwise
- io: concise list of ports with direction and bit width when known
- clocking: whether sequential; clock/reset names and edge semantics if applicable
- timing: per-output latency expectations (0 = combinational / same-cycle, 1 = next-cycle, etc.) when inferable
- functional_summary: 3-8 bullets describing behavior precisely
- corner_cases: 3-8 bullets (overflows, resets, illegal inputs, boundary indices, etc.)
- test_plan: 5-10 bullets of directed tests the Verifier should include
- guidance:
  - verifier: TB guidance (sampling edge, latency handling, reset sequencing)
  - coder: RTL guidance (structure, state, arithmetic width, initialization)
  - debugger: common failure patterns to check first

<input_spec>
{input_spec}
</input_spec>

{golden_tb_block}
"""

REVISE_CONTRACT_PROMPT = r"""
Revise the contract below to fix the reported lint errors.

Hard rules:
- Output MUST be a single valid JSON object (no extra text).
- Do NOT invent new ports unless required by the spec/golden TB.
- Keep module_name and io consistent with the golden testbench when provided.
- In contract-only mode, all downstream agents will follow this contract strictly.

Lint errors:
{lint_errors}

Original spec (for context only; contract is the deliverable):
<input_spec>
{input_spec}
</input_spec>

Current contract:
<contract_json>
{contract_json}
</contract_json>

{golden_tb_block}
"""

EXAMPLE_OUTPUT: Dict[str, Any] = {
    "source_of_truth": "spec",
    "module_name": "TopModule",
    "io": [
        {"name": "clk", "dir": "input", "width": 1, "notes": "posedge clock"},
        {"name": "reset", "dir": "input", "width": 1, "notes": "active-high synchronous reset"},
        {"name": "in_", "dir": "input", "width": 8, "notes": ""},
        {"name": "out", "dir": "output", "width": 8, "notes": "registered output"},
    ],
    "clocking": {
        "is_sequential": True,
        "clock": {"name": "clk", "edge": "posedge"},
        "reset": {"name": "reset", "active": "high", "type": "synchronous"},
    },
    "timing": {"out": {"latency_cycles": 1, "notes": "output updates on next cycle"}},
    "functional_summary": [
        "On each rising edge, register the input.",
        "Output equals (registered input + 1) modulo 2^8.",
    ],
    "corner_cases": ["Reset behavior on first active clock edge.", "Overflow wraps around."],
    "test_plan": [
        "Reset then verify known outputs.",
        "Random inputs; check one-cycle latency.",
        "Overflow case at max value.",
    ],
    "guidance": {
        "verifier": [
            "Sample outputs on the opposite edge to avoid race (if posedge sequential, check at negedge).",
            "If latency_cycles=1, compute expected with a one-cycle queue.",
        ],
        "coder": [
            "Match the exact port list and names.",
            "Use always_ff/always_comb or equivalent; keep widths explicit.",
        ],
        "debugger": [
            "First check off-by-one-cycle issues and reset polarity/type.",
            "Check signed/unsigned and bit-width truncation.",
        ],
    },
}


class ContractFormat(BaseModel):
    source_of_truth: str
    module_name: str
    io: List[Dict[str, Any]]
    clocking: Dict[str, Any]
    timing: Dict[str, Any]
    functional_summary: List[str]
    corner_cases: List[str]
    test_plan: List[str]
    guidance: Dict[str, Any]


class ArchitectAgent:
    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        self._agent = SafeReActAgent(
            name="Architect",
            sys_prompt=SYSTEM_PROMPT,
            model=make_chat_model(cfg),
            formatter=make_formatter(cfg),
            memory=InMemoryMemory(),
            max_iters=10,
        )
        self.last_prompt: str = ""
        self.last_raw_output: str = ""

    async def reset(self) -> None:
        await clear_memory_safely(self._agent)

    def _read_golden_tb_excerpt(self, golden_tb_path: str, *, max_chars: int = 6000) -> str:
        try:
            text = Path(golden_tb_path).read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return ""
        text = text.strip()
        if not text:
            return ""
        if len(text) > max_chars:
            text = text[:max_chars] + "\n/* ... truncated ... */\n"
        return text + ("\n" if not text.endswith("\n") else "")

    def parse_output(self, response_text: str) -> ContractFormat:
        try:
            obj: Dict[str, Any] = extract_json_object(response_text)
            return ContractFormat.model_validate(obj)
        except Exception as e:  # noqa: BLE001
            # Keep the failure visible while still returning a valid object shape.
            return ContractFormat(
                source_of_truth="spec",
                module_name="TopModule",
                io=[],
                clocking={},
                timing={},
                functional_summary=[f"Contract parse error: {type(e).__name__}: {e}"],
                corner_cases=[],
                test_plan=[],
                guidance={"verifier": [], "coder": [], "debugger": []},
            )

    async def chat(self, input_spec: str, *, golden_tb_path: Optional[str] = None) -> ContractFormat:
        await self.reset()

        golden_tb_block = ""
        if golden_tb_path:
            excerpt = self._read_golden_tb_excerpt(golden_tb_path)
            if excerpt:
                golden_tb_block = f"<golden_testbench>\n{excerpt}</golden_testbench>\n"

        prompt = CONTRACT_PROMPT.format(input_spec=input_spec, golden_tb_block=golden_tb_block)
        order = ORDER_PROMPT.format(output_format=json.dumps(EXAMPLE_OUTPUT, indent=4))
        full_prompt = f"{prompt}\n\n{order}"
        self.last_prompt = full_prompt
        msg = await self._agent(Msg("user", full_prompt, role="user"))
        text = msg.get_text_content() or ""
        self.last_raw_output = text
        return self.parse_output(text)

    async def revise_contract(
        self,
        *,
        input_spec: str,
        contract_json: str,
        lint_errors: str,
        golden_tb_path: Optional[str] = None,
    ) -> ContractFormat:
        await self.reset()

        golden_tb_block = ""
        if golden_tb_path:
            excerpt = self._read_golden_tb_excerpt(golden_tb_path)
            if excerpt:
                golden_tb_block = f"<golden_testbench>\n{excerpt}</golden_testbench>\n"

        prompt = REVISE_CONTRACT_PROMPT.format(
            input_spec=input_spec,
            contract_json=contract_json,
            lint_errors=lint_errors.strip(),
            golden_tb_block=golden_tb_block,
        )
        order = ORDER_PROMPT.format(output_format=json.dumps(EXAMPLE_OUTPUT, indent=4))
        full_prompt = f"{prompt}\n\n{order}"
        self.last_prompt = full_prompt
        msg = await self._agent(Msg("user", full_prompt, role="user"))
        text = msg.get_text_content() or ""
        self.last_raw_output = text
        return self.parse_output(text)
