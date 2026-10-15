from __future__ import annotations

import asyncio
from typing import List, Tuple

from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from pydantic import BaseModel

from .agents import SafeReActAgent, clear_memory_safely
from .config import LLMConfig
from .model import make_formatter, make_chat_model
from .prompts import FAILED_TRIAL_PROMPT, RTL_4_SHOT_EXAMPLES, TAG_ORDER_PROMPT
from .sim_reviewer import check_syntax
from .utils import add_lineno, clip_text, extract_xml_tag, strip_markdown_code_fences

SYSTEM_PROMPT = r"""
You are Coder, an expert in RTL design.

You write clean, synthesizable, syntactically-correct SystemVerilog RTL that matches the Architect's contract.

Toolchain note:
- The harness uses Verilator. Keep RTL compatible with Verilator (standard synthesizable SystemVerilog).

Contract-only mode:
- Treat <contract_json> as the ONLY source of truth for interface/timing/behavior.
- <input_spec> and any testbench text are non-authoritative background and must NOT override the contract.
"""

# Ablation mode (standalone generation + syntax check): there is no Architect contract,
# so <input_spec> is the only source of truth.
ABLATION_SYSTEM_PROMPT = r"""
You are Coder, an expert in RTL design.

You write clean, synthesizable, syntactically-correct SystemVerilog RTL that implements the given specification.

Toolchain note:
- The harness uses Verilator. Keep RTL compatible with Verilator (standard synthesizable SystemVerilog).
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

GENERATION_PROMPT = r"""
Write a complete SystemVerilog RTL module that implements the DUT.

Hard rules:
- Follow the Architect contract strictly. Do NOT invent behavior/timing not stated in the contract.
- Match the exact module name and port list.
- Produce synthesizable RTL (no delays, no testbench constructs).
- Keep the code small and readable.

In `reasoning`, write a short summary of key decisions and assumptions (no step-by-step chain-of-thought).

The module interface must EXACTLY match the contract (module name, port names, widths).

<contract_json>
{contract_json}
</contract_json>

{examples_prompt}
<input_spec>
{input_spec}
</input_spec>
"""

ABLATION_GENERATION_PROMPT = r"""
Write a complete SystemVerilog RTL module that implements the specification in <input_spec>.

Hard rules:
- Follow the specification. Do NOT invent behavior/timing that it does not state.
- Match the exact module name and port list given in the specification.
- Produce synthesizable RTL (no delays, no testbench constructs).
- Keep the code small and readable.

In `reasoning`, write a short summary of key decisions and assumptions (no step-by-step chain-of-thought).

{examples_prompt}
<input_spec>
{input_spec}
</input_spec>
"""

KMAP_HINT_PROMPT = r"""
[K-map hint (only if the problem involves Karnaugh maps / K-maps)]:
- Carefully interpret the K-map variable ordering from the spec/contract.
- Note: x[i] in x[N:1] corresponds to x[i-1] in x[N-1:0].
- Enumerate minterms/implicants for output=1/0/don't-care as required; implement minimal logic.
- If you have a descending range like `logic x[M:N]` (M>N), do NOT use reverse selects like `x[1:2]`;
  instead use concatenation: `{x[1], x[2]}`.
"""

EXTRA_ORDER_PROMPT = r"""
Other requirements:
1. Don't use state_t to define the parameter. Use `localparam` or Use 'reg' or 'logic' for signals as registers or Flip-Flops.
2. Declare all ports as logic. For internal nets/regs, prefer logic; `wire` is fine for continuous assigns.
3. If the contract specifies a reset, use it to ensure deterministic behavior as specified.
   If the contract does NOT include a reset but the testbench expects deterministic startup, you MAY use a minimal `initial`
   initialization to avoid X-propagation in simulation (Verilator-focused). Do not rely on `initial` for core functionality.
4. For combinational logic with an always block do not explicitly specify the sensitivity list; instead use always @(*).
5. NEVER USE 'inside' operator in RTL code. Code like 'state inside {STATE_B, STATE_C, STATE_D}' should NOT be used.
6. Never USE 'unique' or 'unique0' keywords in RTL code. Code like 'unique case' should NOT be used.
7. Respect any explicit latency/timing described in the contract (e.g., next-cycle outputs).
"""

ABLATION_EXTRA_ORDER_PROMPT = r"""
Other requirements:
1. Don't use state_t to define the parameter. Use `localparam` or Use 'reg' or 'logic' for signals as registers or Flip-Flops.
2. Declare all ports as logic. For internal nets/regs, prefer logic; `wire` is fine for continuous assigns.
3. If the specification includes a reset, use it to ensure deterministic behavior as specified.
   If it does NOT include a reset, you MAY use a minimal `initial` initialization to avoid X-propagation
   in simulation (Verilator-focused). Do not rely on `initial` for core functionality.
4. For combinational logic with an always block do not explicitly specify the sensitivity list; instead use always @(*).
5. NEVER USE 'inside' operator in RTL code. Code like 'state inside {STATE_B, STATE_C, STATE_D}' should NOT be used.
6. Never USE 'unique' or 'unique0' keywords in RTL code. Code like 'unique case' should NOT be used.
7. Respect any explicit latency/timing described in the specification (e.g., next-cycle outputs).
"""
# Some prompts above comes from:
# @misc{ho2024verilogcoderautonomousverilogcoding,
#       title={VerilogCoder: Autonomous Verilog Coding Agents with Graph-based Planning and Abstract Syntax Tree (AST)-based Waveform Tracing Tool},
#       author={Chia-Tung Ho and Haoxing Ren and Brucek Khailany},
#       year={2024},
#       eprint={2408.08927},
#       archivePrefix={arXiv},
#       primaryClass={cs.AI},
#       url={https://arxiv.org/abs/2408.08927},
# }

IF_PROMPT = r"""
The module interface is given below:
<module_interface>
{module_interface}
</module_interface>
"""

TB_PROMPT = r"""
Verifier has generated a testbench (intended to follow the contract/spec):
<testbench>
{testbench}
</testbench>
"""

FORMAT_ERROR_PROMPT = r"""
The error below has been reported by the format tool:
<format_error>
{format_error}
</format_error>
To understand the error message better, we offered a version of generated module with line number:
<module_with_lineno>
{module_with_lineno}
</module_with_lineno>
"""

EXAMPLE_OUTPUT_FORMAT = """<reasoning>
Concise rationale + key assumptions (no step-by-step chain-of-thought)
</reasoning>
<module>
Complete synthesizable SystemVerilog RTL module
</module>
"""


class RTLOutputFormat(BaseModel):
    reasoning: str
    module: str


def _needs_kmap_hints(*texts: str) -> bool:
    text = "\n".join(texts).lower()
    return any(k in text for k in ["kmap", "k-map", "karnaugh"])


class RTLGenerator:
    def __init__(
        self,
        cfg: LLMConfig,
        *,
        ablation: bool = False,
    ):
        self._cfg = cfg
        # Ablation mode has no contract, so it needs a spec-only system prompt (see ablation_chat).
        self.ablation = ablation
        self._agent = self._new_agent(
            name="Coder",
            sys_prompt=ABLATION_SYSTEM_PROMPT if ablation else SYSTEM_PROMPT,
        )
        self.generated_tb: str | None = None
        self.generated_if: str | None = None
        self.failed_trial: List[str] = []
        self.max_trials = 5
        self.last_prompt: str = ""
        self.last_raw_output: str = ""

    async def reset(self) -> None:
        await clear_memory_safely(self._agent)

    def _new_agent(self, *, name: str, sys_prompt: str) -> SafeReActAgent:
        return SafeReActAgent(
            name=name,
            sys_prompt=sys_prompt,
            model=make_chat_model(self._cfg),
            formatter=make_formatter(self._cfg),
            memory=InMemoryMemory(),
            max_iters=10,
        )

    def set_failed_trial(
        self, failed_sim_log: str, previous_code: str, previous_tb: str
    ) -> None:
        cur_failed_trial = FAILED_TRIAL_PROMPT.format(
            failed_sim_log=failed_sim_log,
            previous_code=add_lineno(previous_code),
            previous_tb=add_lineno(previous_tb),
        )
        self.failed_trial.append(cur_failed_trial)

    async def _call_agent(self, agent: SafeReActAgent, prompt: str) -> str:
        self.last_prompt = prompt
        msg = await agent(Msg("user", prompt, role="user"))
        text = msg.get_text_content() or ""
        self.last_raw_output = text
        return text

    def get_init_prompt_messages(self, input_spec: str, *, contract_json: str) -> List[Msg]:
        parts: list[str] = [
            GENERATION_PROMPT.format(
                input_spec=input_spec,
                examples_prompt=RTL_4_SHOT_EXAMPLES,
                contract_json=contract_json,
            )
        ]
        if _needs_kmap_hints(input_spec, contract_json):
            parts.append(KMAP_HINT_PROMPT)
        if self.generated_tb:
            parts.append(TB_PROMPT.format(testbench=self.generated_tb))
        parts.extend(self.failed_trial)
        if self.generated_if:
            parts.append(IF_PROMPT.format(module_interface=self.generated_if))
        return [Msg("user", "\n\n".join(parts), role="user")]

    def get_order_prompt_messages(self) -> List[Msg]:
        return [
            Msg(
                "user",
                TAG_ORDER_PROMPT.format(output_format=EXAMPLE_OUTPUT_FORMAT)
                + EXTRA_ORDER_PROMPT,
                "user",
            )
        ]

    def get_format_error_prompt_messages(
        self, format_error: str, rtl_code: str
    ) -> List[Msg]:
        return [
            Msg(
                "user",
                FORMAT_ERROR_PROMPT.format(
                    format_error=format_error, module_with_lineno=add_lineno(rtl_code)
                ),
                "user",
            )
        ]

    def parse_output(self, response_text: str) -> RTLOutputFormat:
        try:
            module = strip_markdown_code_fences(extract_xml_tag(response_text, "module")).strip()
            reasoning = extract_xml_tag(response_text, "reasoning", required=False).strip()
            if not module:
                raise ValueError("Empty <module> block")
            ret = RTLOutputFormat(reasoning=reasoning, module=module)
        except Exception as e:  # noqa: BLE001
            ret = RTLOutputFormat(reasoning=f"Parse Error: {type(e).__name__}: {e}", module="")
        return ret

    async def chat(
        self,
        input_spec: str,
        testbench: str,
        interface: str,
        rtl_path: str,
        contract_json: str,
    ) -> Tuple[bool, str]:
        await self.reset()
        self.generated_tb = testbench
        self.generated_if = interface

        init = self.get_init_prompt_messages(input_spec, contract_json=contract_json)[0].content
        order = self.get_order_prompt_messages()[0].content
        return await self._generate_until_syntax_ok(init=init, order=order, rtl_path=rtl_path)

    async def ablation_chat(self, input_spec: str, rtl_path: str) -> Tuple[bool, str]:
        """Standalone generation from the spec alone (no contract/testbench), with syntax retries."""
        if not self.ablation:
            raise ValueError("ablation_chat requires RTLGenerator(cfg, ablation=True).")
        await self.reset()
        self.generated_tb = None
        self.generated_if = None

        parts = [ABLATION_GENERATION_PROMPT.format(input_spec=input_spec, examples_prompt=RTL_4_SHOT_EXAMPLES)]
        if _needs_kmap_hints(input_spec):
            parts.append(KMAP_HINT_PROMPT)
        init = "\n\n".join(parts)
        order = TAG_ORDER_PROMPT.format(output_format=EXAMPLE_OUTPUT_FORMAT) + ABLATION_EXTRA_ORDER_PROMPT
        return await self._generate_until_syntax_ok(init=init, order=order, rtl_path=rtl_path)

    async def _generate_until_syntax_ok(self, *, init: str, order: str, rtl_path: str) -> Tuple[bool, str]:
        prompt = f"{init}\n\n{order}"
        syntax_correct = False
        rtl_code = ""
        for _ in range(self.max_trials):
            response_text = await self._call_agent(self._agent, prompt)
            resp_obj = self.parse_output(response_text)
            if resp_obj.reasoning.startswith("Parse Error"):
                repair = PARSE_REPAIR_PROMPT.format(
                    parse_error=resp_obj.reasoning,
                    bad_output=clip_text(response_text, max_chars=6000),
                )
                prompt = f"{repair}\n\n{order}"
                continue
            rtl_code = resp_obj.module
            with open(rtl_path, "w") as f:
                f.write(rtl_code)
            syntax_correct, syntax_output = await asyncio.to_thread(
                check_syntax, rtl_path
            )
            if syntax_correct:
                break
            fmt = self.get_format_error_prompt_messages(syntax_output, rtl_code)[0].content
            prompt = f"{fmt}\n\n{order}"
        return (syntax_correct, rtl_code)
