from __future__ import annotations

from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from pydantic import BaseModel

from .agents import SafeReActAgent, clear_memory_safely
from .config import LLMConfig
from .model import make_formatter, make_chat_model
from .utils import clip_text, extract_xml_tag, strip_markdown_code_fences


SYSTEM_PROMPT = r"""
You are BooleanProofer, an expert in combinational logic reasoning and formal equivalence checking.

Goal:
- Convert the Architect contract's `functional_summary` into a precise, purely-combinational SystemVerilog
  specification for the DUT outputs.

Scope and constraints:
- Only target purely combinational behavior (same-cycle outputs). No sequential elements.
- Do NOT write a module header. You will be given the exact module header; output ONLY the module body.
- Assign EVERY output for ALL input combinations (avoid latches).
- Use synthesizable constructs compatible with Yosys/Verilator: `assign` or `always_comb`/`always @(*)`.
- Avoid unsupported/fragile features: no `sequence`, no fancy SVA, no `unique`/`priority`, no delays, no DPI.
- Do not invent ports or change names/widths.

Important:
- `functional_summary` may be incomplete or ambiguous. If so, make the smallest reasonable assumptions and
  document them in <reasoning>. This spec is used only as a *hint*; it does not override simulation.

When you are done, output using the required tags and NOTHING else.
"""


GENERATION_PROMPT = r"""
You are given a DUT contract (JSON). Extract the intended *combinational* logic from:
- contract.functional_summary (primary)
- contract.corner_cases (secondary, if relevant)

You are also given the exact SystemVerilog module header that MUST be preserved.

Task:
- Write a complete module body for `SpecModule` that implements the combinational behavior.
- Focus on the outputs listed in <target_outputs>. These are intended to be same-cycle / combinational outputs.
- For outputs NOT listed in <target_outputs>, assign a deterministic default (e.g., `'0`) to avoid latches.

<contract_json>
{contract_json}
</contract_json>

<module_header>
{module_header}
</module_header>

<target_outputs>
{target_outputs}
</target_outputs>
"""


EXAMPLE_OUTPUT_FORMAT = """<reasoning>
Concise assumptions/limitations (no step-by-step chain-of-thought)
</reasoning>
<spec_body>
SystemVerilog module body only (no module/endmodule)
</spec_body>
"""


TAG_ORDER_PROMPT = r"""
Output format (follow exactly):
<output_format>
{output_format}
</output_format>

Rules:
- Output ONLY the tags in <output_format> (no extra text).
- Do NOT wrap code in Markdown fences (```).
"""


PARSE_REPAIR_PROMPT = r"""
Your previous response could not be parsed by the program.

Parser error:
{parse_error}

Previous response (truncated):
<bad_output>
{bad_output}
</bad_output>

Please output again, strictly following the required tags in <output_format>, and output NOTHING else.
Do NOT output JSON. Do NOT wrap code in Markdown code fences (```).
"""


class BooleanProoferSpec(BaseModel):
    reasoning: str
    spec_body: str


class BooleanProoferAgent:
    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        self._agent = SafeReActAgent(
            name="BooleanProofer",
            sys_prompt=SYSTEM_PROMPT,
            model=make_chat_model(cfg),
            formatter=make_formatter(cfg),
            memory=InMemoryMemory(),
            max_iters=10,
        )
        self.max_trials = 3

    async def reset(self) -> None:
        await clear_memory_safely(self._agent)

    async def _call_agent(self, prompt: str) -> str:
        msg = await self._agent(Msg("user", prompt, role="user"))
        return msg.get_text_content() or ""

    def parse_output(self, response_text: str) -> BooleanProoferSpec:
        try:
            spec_body = strip_markdown_code_fences(extract_xml_tag(response_text, "spec_body")).strip()
            if not spec_body:
                raise ValueError("Empty <spec_body> block")
            reasoning = extract_xml_tag(response_text, "reasoning", required=False).strip()
            return BooleanProoferSpec(reasoning=reasoning, spec_body=spec_body)
        except Exception as e:  # noqa: BLE001
            return BooleanProoferSpec(reasoning=f"Parse Error: {type(e).__name__}: {e}", spec_body="")

    async def chat(self, *, contract_json: str, module_header: str, target_outputs: list[str]) -> BooleanProoferSpec:
        await self.reset()
        order = TAG_ORDER_PROMPT.format(output_format=EXAMPLE_OUTPUT_FORMAT)
        targets = "\n".join(f"- {s}" for s in target_outputs if isinstance(s, str) and s.strip())
        prompt = f"{GENERATION_PROMPT.format(contract_json=contract_json, module_header=module_header, target_outputs=targets)}\n\n{order}"

        last: BooleanProoferSpec = BooleanProoferSpec(reasoning="Parse Error: unknown", spec_body="")
        for _ in range(max(1, int(self.max_trials))):
            text = await self._call_agent(prompt)
            last = self.parse_output(text)
            if not last.reasoning.startswith("Parse Error"):
                return last
            repair = PARSE_REPAIR_PROMPT.format(
                parse_error=last.reasoning,
                bad_output=clip_text(text, max_chars=6000),
            )
            prompt = f"{repair}\n\n{order}"

        return last
