from __future__ import annotations

from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from pydantic import BaseModel

from .agents import SafeReActAgent, clear_memory_safely
from .config import LLMConfig
from .model import make_formatter, make_chat_model
from .utils import clip_text, extract_xml_tag, strip_markdown_code_fences


SYSTEM_PROMPT = r"""
You are Asserter, an expert in SystemVerilog assertions (SVA) and sequential logic debugging.

Goal:
- Generate a small set of non-intrusive assertions to diagnose timing/latency/reset/edge issues in the DUT,
  guided by the Architect contract. These assertions are *hints* and do not override simulation mismatch results.

Scope and constraints:
- You will be given the exact SystemVerilog module header for `AssertModule` that mirrors the DUT ports
  (all ports declared as inputs for observation). Output ONLY the module body.
- Do NOT write a module header or `endmodule`. Do NOT invent ports or change names/widths.
- You MUST only reference signals that exist in the provided <module_header>.
- Keep assertions compatible with Verilator (partial SVA support):
  - Prefer immediate `assert (...) else ...` inside `always @(posedge clk)` / `always @(negedge clk)`.
  - Verilator translates assertions to simple `if (...) error` checks; keep expressions simple.
  - Avoid SEREs and `sequence ... endsequence`. Avoid repetition operators like `[*]`.
  - Avoid multi-cycle temporal sequences (e.g., `##N`, `|->`, `|=>`) unless you are certain it stays single-cycle.
    If you need 1-cycle history, prefer `$past(sig)` or explicit `*_prev` registers.
- Assertions MUST be non-fatal: do not call `$fatal`. Prefer `$display`/`$error`.
- When an assertion fails, emit a line that starts with `ASSERTER_FAIL:` (so the harness can parse it).
- Keep the number of assertions small and focus on <target_outputs>.

When you are done, output using the required tags and NOTHING else.
"""


GENERATION_PROMPT = r"""
You are given:
1) A DUT contract (JSON; source of truth for timing/interface).
2) The exact `AssertModule` header (MUST be preserved).
3) A list of <target_outputs> that are currently failing in simulation.

Task:
- Write a module body for `AssertModule` that adds diagnostic assertions for sequential/timing issues.
- Prefer checks that help distinguish:
  - wrong clock edge (posedge vs negedge),
  - reset polarity/type mismatches,
  - off-by-one-cycle latency (when detectable),
  - outputs changing outside the expected clock edge.

Notes:
- A helper task is available in the surrounding file:
    task automatic asserter_log(string id, string msg);
      // prints "ASSERTER_FAIL: <id> t=<time> <msg>" (and rate-limits)
    endtask
  You MAY call `asserter_log(...)` in your `else` blocks to standardize output.
- If you need a previous-value register, derive its width from the port to avoid width mistakes:
    logic [$bits(sig)-1:0] sig_prev;
  (This is synthesizable and supported by Verilator.)

<contract_json>
{contract_json}
</contract_json>

<module_header>
{module_header}
</module_header>

<clocking_summary>
{clocking_summary}
</clocking_summary>

<target_outputs>
{target_outputs}
</target_outputs>
"""


EXAMPLE_OUTPUT_FORMAT = """<reasoning>
Concise assumptions/limitations (no step-by-step chain-of-thought)
</reasoning>
<assert_body>
SystemVerilog module body only (no module/endmodule)
</assert_body>
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


class AsserterSpec(BaseModel):
    reasoning: str
    assert_body: str


class AsserterAgent:
    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        self._agent = SafeReActAgent(
            name="Asserter",
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

    def parse_output(self, response_text: str) -> AsserterSpec:
        try:
            assert_body = strip_markdown_code_fences(extract_xml_tag(response_text, "assert_body")).strip()
            reasoning = extract_xml_tag(response_text, "reasoning", required=False).strip()
            return AsserterSpec(reasoning=reasoning, assert_body=assert_body)
        except Exception as e:  # noqa: BLE001
            return AsserterSpec(reasoning=f"Parse Error: {type(e).__name__}: {e}", assert_body="")

    async def chat(
        self,
        *,
        contract_json: str,
        module_header: str,
        clocking_summary: str,
        target_outputs: list[str],
    ) -> AsserterSpec:
        await self.reset()
        order = TAG_ORDER_PROMPT.format(output_format=EXAMPLE_OUTPUT_FORMAT)
        targets = "\n".join(f"- {s}" for s in target_outputs if isinstance(s, str) and s.strip())
        prompt = (
            f"{GENERATION_PROMPT.format(contract_json=contract_json, module_header=module_header, clocking_summary=clocking_summary, target_outputs=targets)}\n\n{order}"
        )

        last: AsserterSpec = AsserterSpec(reasoning="Parse Error: unknown", assert_body="")
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
