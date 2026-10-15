from __future__ import annotations

from typing import List, Tuple

from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from pydantic import BaseModel

from .agents import SafeReActAgent, clear_memory_safely
from .config import OpenAIConfig
from .model import make_formatter, make_openai_model
from .prompts import FAILED_TRIAL_PROMPT, TAG_ORDER_PROMPT, TB_4_SHOT_EXAMPLES
from .utils import add_lineno, clip_text, extract_xml_tag, strip_markdown_code_fences

SYSTEM_PROMPT = r"""
You are Verifier, an expert in SystemVerilog verification.

You write high-signal, self-checking testbenches that accurately reflect the spec/contract and
produce clear, minimal logs when failures happen.

You always generate syntactically-correct SystemVerilog.

Simulator target (important):
- The harness uses Verilator. Keep the testbench compatible with Verilator's SystemVerilog support.

Contract-only mode:
- Treat <contract_json> as the ONLY source of truth for interface/timing/behavior.
- <input_spec> is non-authoritative background and must NOT override the contract.
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

TB_LINT_FAILED_PROMPT = r"""
The previously generated testbench failed Verilator linting.

Verilator lint output:
<verilator_lint_log>
{lint_log}
</verilator_lint_log>

Previous testbench (with line numbers):
<previous_tb_with_lineno>
{previous_tb_with_lineno}
</previous_tb_with_lineno>

Regenerate the <interface> and <testbench> to fix ONLY the lint/syntax/unsupported-feature issues while preserving the contract behavior.
Output must still follow <output_format> exactly.
"""

NON_GOLDEN_TB_PROMPT = r"""
You are given:
1) A JSON contract written by the Architect agent (SOURCE OF TRUTH);
2) An optional input spec (non-authoritative background).

Task:
1) Write the DUT IO interface (module header only; no implementation) exactly as specified by the contract;
2) Write a testbench to verify the DUT strictly against the contract.

Hard rules:
- Follow the contract. Do NOT invent behavior/timing not stated in the contract.
- The module interface MUST match the contract/spec exactly (module name, port names, widths).
- Name the DUT instance `dut` (non-golden mode).
- Do not use the SystemVerilog `continue` keyword.
- Verilator target: keep the TB compatible (avoid `sequence ... endsequence` and SVA `[*]` repetition; prefer simple assertions).

<contract_json>
{contract_json}
</contract_json>

<input_spec>
{input_spec}
</input_spec>

Testbench requirements:
1) Instantiate the DUT according to the interface (instance name `dut`).
2) Drive stimuli and compute expected outputs consistent with the contract (including any stated latency).
3) Count mismatches between DUT outputs and expected outputs.
4) Logging (keep logs small):
   - Do NOT print on every match.
   - On mismatch, display inputs, DUT outputs, and expected outputs.
   - On the FIRST mismatch, print extra debug context per the display prompt below (moment or queue window).
5) Generate a VCD named `wave.vcd`:
   initial begin
     $dumpfile("wave.vcd");
     $dumpvars(0, dut);
   end
6) End-of-sim summary (these exact markers are parsed by the harness):
   - If no mismatch: print exactly `SIMULATION PASSED`
   - Else: print exactly `SIMULATION FAILED - x MISMATCHES DETECTED, FIRST AT TIME y`
7) Trace-friendly summary lines (ALWAYS print these near end-of-sim, even if x=0):
   - Print exactly: `Mismatches: x in y samples`
   - For each DUT output port named `sig`, print exactly:
       `Hint: Output 'sig' has n mismatches. First mismatch occurred at time t.`
     where:
       - n = number of samples where this output mismatched (use 0 if never mismatched)
       - t = the earliest time (as integer) where this output mismatched (use 0 if never mismatched)
8) Sampling:
   - For posedge-sequential designs, compute/update expectations on posedge and check on negedge to avoid races.
   - For pure combinational designs (no clock), check at the moment inputs change (after a tiny delay if needed).

In `reasoning`, write a short, practical summary (no step-by-step chain-of-thought).

{examples_prompt}

Please also follow the display prompt below:
{display_prompt}
"""

GOLDEN_TB_PROMPT = r"""
You are given:
1) An input spec describing a DUT to implement;
2) A JSON contract written by the Architect agent;
3) A golden testbench that MUST be treated as ground truth for interface and timing.

Task:
1) Write the DUT IO interface (module header only; no implementation);
2) Improve the golden testbench by adding more helpful displays, while preserving its functionality.

Hard rules:
- The golden testbench is the source of truth when it contradicts the spec/contract.
- Maintain the exact original behavior, interface, module instantiation, and error counting.
- Do not remove existing `$dumpfile`/`$dumpvars`. If missing, add `$dumpfile("wave.vcd")` and `$dumpvars(...)`.
- If the contract disagrees with the golden TB, prefer the golden TB and do NOT "fix" the TB to match the contract.
- Verilator target: do not introduce unsupported SV/SVA features; if adding assertions, keep them non-intrusive and do not change timing/behavior.

<contract_json>
{contract_json}
</contract_json>

<input_spec>
{input_spec}
</input_spec>

Below is the golden testbench code for the module generated with the given natural language specification.
<golden_testbench>
{golden_testbench}
</golden_testbench>

Additions required:
1) Add richer displays on every check (inputs, outputs, expected) without changing timing or behavior.
2) Print the required end-of-sim summary:
   - If no mismatch: print exactly `SIMULATION PASSED`
   - Else: print exactly `SIMULATION FAILED - x MISMATCHES DETECTED, FIRST AT TIME y`

In `reasoning`, write a short, practical summary (no step-by-step chain-of-thought).

Please also follow the display prompt below:
{display_prompt}
"""

DISPLAY_MOMENT_PROMPT = r"""
1. When the first mismatch occurs, display the input signals, output signals and expected output signals at that time.
2. For multiple-bit signals displayed in HEX format, also display the BINARY format if its width <= 64.
"""

DISPLAY_QUEUE_PROMPT = r"""
Verilator compatibility (important):
- Avoid `sequence ... endsequence` declarations (unsupported by Verilator).
- Avoid SVA repetition/abbrev operators like `[*]` (unsupported by Verilator).
- If you need multi-signal history, use multiple queues (one per signal) of simple types (`logic`, `logic [N:0]`, `time`, `int`) as shown in the example below. Avoid queues of structs/packed structs.
- If you use assertions, prefer immediate assertions or simple `assert/assume/cover property` forms that Verilator supports (no advanced SVA features).

1. If module to test is sequential logic (like including an FSM):
    1.1. Store input signals, output signals, expected output signals and reset signals in queues with MAX_QUEUE_SIZE (one queue per signal; do not bundle into a struct);
        When the first mismatch occurs, display the queue contents after storing them. Make sure the mismatched signal can be displayed.
    1.2. MAX_QUEUE_SIZE should be set according to the requirement of the module.
        For example, if the module has a 3-bit state, MAX_QUEUE_SIZE should be at least 2 ** 3 = 8.
        And if the module was to detect a pattern of 8 bits, MAX_QUEUE_SIZE should be at least (8 + 1) = 9.
        However, to control log size, NEVER set MAX_QUEUE_SIZE > 10.
    1.3. The clocking of queue and display should be same with the clocking of tb_match detection.
        For example, if 'always @(posedge clk or negedge clk)' is used to detect mismatch,
        It should also be used to push queue and display first error.
2. If module to test is combinational logic:
    When the first mismatch occurs, display the input signals, output signals and expected output signals at that time.
3. For multiple-bit signals displayed in HEX format, also display the BINARY format if its width <= 64.

<display_queue_example>
// Queue-based simulation mismatch display

reg [INPUT_WIDTH-1:0] input_queue [$];
reg [OUTPUT_WIDTH-1:0] got_output_queue [$];
reg [OUTPUT_WIDTH-1:0] golden_queue [$];
reg reset_queue [$];

localparam MAX_QUEUE_SIZE = 5;

always @(posedge clk or negedge clk) begin
    if (input_queue.size() >= MAX_QUEUE_SIZE - 1) begin
        input_queue.delete(0);
        got_output_queue.delete(0);
        golden_queue.delete(0);
        reset_queue.delete(0);
    end

    input_queue.push_back(input_data);
    got_output_queue.push_back(got_output);
    golden_queue.push_back(golden_output);
    reset_queue.push_back(rst);

    // Check for first mismatch
    if (got_output !== golden_output) begin
        $display("Mismatch detected at time %t", $time);
        $display("\nLast %d cycles of simulation:", input_queue.size());


        for (int i = 0; i < input_queue.size(); i++) begin
            if (got_output_queue[i] === golden_queue[i]) begin
                $display("Got Match at");
            end else begin
                $display("Got Mismatch at");
            end
            $display("Cycle %d, reset %b, input %h, got output %h, exp output %h",
                i,
                reset_queue[i],
                input_queue[i],
                got_output_queue[i],
                golden_queue[i]
            );
        end
    end

end
</display_queue_example>
"""


EXAMPLE_OUTPUT_FORMAT = """<reasoning>
Concise rationale + key assumptions (no step-by-step chain-of-thought)
</reasoning>
<interface>
SystemVerilog module header only (no implementation)
</interface>
<testbench>
Complete SystemVerilog testbench module
</testbench>
"""


class TBOutputFormat(BaseModel):
    reasoning: str
    interface: str
    testbench: str


EXTRA_ORDER_GOLDEN_TB_PROMPT = r"""
Golden TB mode reminders:
- If golden TB contradicts the spec/contract, follow the golden TB.
- Preserve the original behavior, instantiation, and mismatch counting.
- Always print the required end-of-sim summary lines.
- Always generate the complete testbench, even if long.
- Generate the interface according to the golden TB. Declare all ports as `logic`.
"""

EXTRA_ORDER_NON_GOLDEN_TB_PROMPT = r"""
Non-golden mode reminders:
- If the spec is ambiguous about output latency, follow the contract's `timing` section.
- For pattern detectors (if relevant), the common convention is to assert `detected` on the cycle AFTER the pattern completes, unless specified otherwise in the contract/spec.
"""

class TBGenerator:
    def __init__(
        self,
        cfg: OpenAIConfig,
    ):
        self._cfg = cfg
        self._agent = SafeReActAgent(
            name="Verifier",
            sys_prompt=SYSTEM_PROMPT,
            model=make_openai_model(cfg),
            formatter=make_formatter(cfg.model),
            memory=InMemoryMemory(),
            max_iters=10,
        )
        self.failed_trial: List[str] = []
        self.golden_tb_path: str | None = None
        self.parse_max_trial = 5
        self.gen_display_queue = True
        self.last_prompt: str = ""
        self.last_raw_output: str = ""

    async def reset(self) -> None:
        await clear_memory_safely(self._agent)

    def set_golden_tb_path(self, golden_tb_path: str | None) -> None:
        self.golden_tb_path = golden_tb_path

    def set_failed_trial(
        self, failed_sim_log: str, previous_code: str, previous_tb: str
    ) -> None:
        cur_failed_trial = FAILED_TRIAL_PROMPT.format(
            failed_sim_log=failed_sim_log,
            previous_code=add_lineno(previous_code),
            previous_tb=add_lineno(previous_tb),
        )
        self.failed_trial.append(cur_failed_trial)

    def set_tb_lint_error(self, *, lint_log: str, previous_tb: str) -> None:
        cur = TB_LINT_FAILED_PROMPT.format(
            lint_log=lint_log.strip(),
            previous_tb_with_lineno=add_lineno(previous_tb),
        )
        self.failed_trial.append(cur)

    def get_init_prompt_messages(self, input_spec: str, *, contract_json: str) -> List[Msg]:
        display_prompt = (
            DISPLAY_QUEUE_PROMPT if self.gen_display_queue else DISPLAY_MOMENT_PROMPT
        )
        if self.golden_tb_path:
            with open(self.golden_tb_path, "r") as f:
                golden_testbench = f.read()
            generation_content = GOLDEN_TB_PROMPT.format(
                input_spec=input_spec,
                golden_testbench=golden_testbench,
                display_prompt=display_prompt,
                contract_json=contract_json,
            )
        else:
            generation_content = NON_GOLDEN_TB_PROMPT.format(
                input_spec=input_spec,
                examples_prompt=TB_4_SHOT_EXAMPLES,
                display_prompt=display_prompt,
                contract_json=contract_json,
            )
        parts: list[str] = [generation_content]
        parts.extend(self.failed_trial)
        return [Msg("user", "\n\n".join(parts), role="user")]

    def get_order_prompt_messages(self) -> List[Msg]:
        if self.golden_tb_path:
            order_prompt_message = Msg(
                "user",
                TAG_ORDER_PROMPT.format(output_format=EXAMPLE_OUTPUT_FORMAT)
                + EXTRA_ORDER_GOLDEN_TB_PROMPT,
                "user",
            )
        else:
            order_prompt_message = Msg(
                "user",
                TAG_ORDER_PROMPT.format(output_format=EXAMPLE_OUTPUT_FORMAT)
                + EXTRA_ORDER_NON_GOLDEN_TB_PROMPT,
                "user",
            )

        return [order_prompt_message]

    def parse_output(self, response_text: str) -> TBOutputFormat:
        try:
            interface = strip_markdown_code_fences(extract_xml_tag(response_text, "interface")).strip()
            testbench = strip_markdown_code_fences(extract_xml_tag(response_text, "testbench")).strip()
            reasoning = extract_xml_tag(response_text, "reasoning", required=False).strip()
            if not interface or not testbench:
                raise ValueError("Empty <interface> or <testbench> block")
            ret = TBOutputFormat(reasoning=reasoning, interface=interface, testbench=testbench)
        except Exception as e:  # noqa: BLE001
            ret = TBOutputFormat(
                reasoning=f"Parse Error: {type(e).__name__}: {e}",
                interface="",
                testbench="",
            )
        return ret

    async def chat(self, input_spec: str, *, contract_json: str) -> Tuple[str, str]:
        await self.reset()
        init = self.get_init_prompt_messages(input_spec, contract_json=contract_json)[0].content
        order = self.get_order_prompt_messages()[0].content
        prompt = f"{init}\n\n{order}"

        response_text = ""
        resp_obj = TBOutputFormat(reasoning="", interface="", testbench="")
        for _ in range(self.parse_max_trial):
            self.last_prompt = prompt
            msg = await self._agent(Msg("user", prompt, role="user"))
            response_text = msg.get_text_content() or ""
            self.last_raw_output = response_text
            resp_obj = self.parse_output(response_text)
            if not resp_obj.reasoning.startswith("Parse Error"):
                break
            repair = PARSE_REPAIR_PROMPT.format(
                parse_error=resp_obj.reasoning,
                bad_output=clip_text(response_text, max_chars=6000),
            )
            prompt = f"{repair}\n\n{order}"
        if resp_obj.reasoning.startswith("Parse Error"):
            raise ValueError(
                f"Parse error when decoding model output: {response_text}"
            )
        return (resp_obj.testbench, resp_obj.interface)
