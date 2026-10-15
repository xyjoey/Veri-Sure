from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import shutil
from typing import Callable, List, Tuple

from .bash_tools import CommandResult, run_bash_command
from .architect_agent import ArchitectAgent
from .contract_linter import lint_contract_json, render_contract_issues
from .config import OpenAIConfig
from .rtl_editor import RTLEditor
from .model import get_model_usage
from .rtl_generator import RTLGenerator
from .sim_reviewer import SimReviewer
from .tb_generator import TBGenerator


@dataclass(frozen=True)
class TopAgentConfig:
    sim_max_retry: int = 4
    is_ablation: bool = False
    contract_only: bool = True
    debug_max_trials: int = 30


@dataclass(frozen=True)
class TopAgentResult:
    output_dir_per_run: str
    rtl_path: str
    tb_path: str
    if_path: str
    is_sim_pass: bool
    rtl_code: str
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class TopAgent:
    """Multi-agent RTL generation loop (Architect/Verifier/Coder/Debugger)."""

    def __init__(self, cfg: OpenAIConfig, *, config: TopAgentConfig | None = None) -> None:
        self.cfg = cfg
        self.config = config or TopAgentConfig()

    def _write_output(self, *, output_dir_per_run: Path, file_name: str, content: str) -> None:
        output_dir_per_run.mkdir(parents=True, exist_ok=True)
        (output_dir_per_run / file_name).write_text(content, encoding="utf-8")

    def _augment_dumpvars_with_dut_scope(self, testbench: str, *, module_name: str = "TopModule") -> str:
        """Best-effort: dump DUT internals into wave.vcd for trace-based debugging."""
        m = re.search(rf"\b{re.escape(module_name)}\b\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", testbench)
        if not m:
            return testbench
        inst = m.group(1)

        if re.search(rf"\$dumpvars\s*\(\s*0\s*,\s*{re.escape(inst)}\s*\)", testbench):
            return testbench

        lines = testbench.splitlines()
        insert_after = None
        for i, line in enumerate(lines):
            if "$dumpvars" in line:
                insert_after = i
                break
        if insert_after is None:
            for i, line in enumerate(lines):
                if "$dumpfile" in line:
                    insert_after = i
                    break
        if insert_after is None:
            return testbench

        indent = re.match(r"\s*", lines[insert_after]).group(0)  # type: ignore[union-attr]
        lines.insert(insert_after + 1, f"{indent}$dumpvars(0, {inst});")
        return "\n".join(lines) + ("\n" if testbench.endswith("\n") else "")

    def _tb_has_syntax_error(self, *, tb_path: Path) -> bool:
        """Return True if `tb_path` contains a real syntax error (not just missing module defs)."""
        if shutil.which("verilator") is None:
            # If the environment lacks Verilator, don't force golden TB fallback.
            return False

        cmd = f"verilator --lint-only --sv --timing -Wall -Wno-fatal --assert {shlex.quote(str(tb_path))}"
        _ok, out = run_bash_command(cmd, timeout=60, cwd=str(tb_path.parent))
        try:
            obj = CommandResult.model_validate_json(out)
        except Exception:  # noqa: BLE001
            return False
        stdout = obj.stdout or ""
        stderr = obj.stderr or ""
        text = f"{stdout}\n{stderr}".lower()

        # Ignore the expected "TopModule is missing" error when linting a testbench
        # file in isolation. We only care about whether the testbench itself is broken
        # (or uses unsupported SV features) for golden TB fallback.
        error_lines = [
            line.strip()
            for line in f"{stdout}\n{stderr}".splitlines()
            if line.lstrip().startswith("%Error")
        ]
        real_error_lines = [
            line
            for line in error_lines
            if ("-MODMISSING:" not in line)
            and (not line.startswith("%Error: Exiting due to"))
        ]
        if real_error_lines:
            return True

        # If there were no "real" errors, treat syntax errors as broken TB.
        return ("syntax error" in text) or ("malformed statement" in text)

    def _tb_lint_report(self, *, tb_path: Path) -> tuple[bool, str, str]:
        """Return (is_ok, excerpt, raw_json) from `verilator --lint-only` on the testbench.

        This lints the TB in isolation; missing DUT module definitions are expected and ignored.
        """
        if shutil.which("verilator") is None:
            return True, "", ""

        cmd = f"verilator --lint-only --sv --timing -Wall -Wno-fatal --assert {shlex.quote(str(tb_path))}"
        _ok, out = run_bash_command(cmd, timeout=60, cwd=str(tb_path.parent))
        try:
            obj = CommandResult.model_validate_json(out)
        except Exception:  # noqa: BLE001
            return False, out, out

        stdout = obj.stdout or ""
        stderr = obj.stderr or ""
        text = f"{stdout}\n{stderr}"

        error_lines = [
            line.strip()
            for line in text.splitlines()
            if line.lstrip().startswith("%Error")
        ]
        real_error_lines = [
            line
            for line in error_lines
            if ("-MODMISSING:" not in line)
            and (not line.startswith("%Error: Exiting due to"))
        ]

        is_ok = not real_error_lines and ("syntax error" not in text.lower()) and ("malformed statement" not in text.lower())
        excerpt = "\n".join(real_error_lines).strip() or text.strip()
        if len(excerpt) > 6000:
            excerpt = excerpt[:6000] + "\n...<snip>...\n"
        return is_ok, excerpt + ("\n" if excerpt and not excerpt.endswith("\n") else ""), out

    async def _build_contract_json(
        self,
        *,
        architect: ArchitectAgent,
        spec: str,
        golden_tb_path: str | None,
        output_dir_per_run: Path,
        max_repairs: int = 2,
    ) -> str:
        """Generate contract.json and (best-effort) repair it until lint passes."""
        contract = await architect.chat(spec, golden_tb_path=golden_tb_path)
        contract_json = contract.model_dump_json(indent=2, exclude_none=True) + "\n"

        for repair_idx in range(max(0, int(max_repairs)) + 1):
            issues, _obj = lint_contract_json(contract_json)
            errors = [i for i in issues if i.severity == "error"]
            lint_report = render_contract_issues(issues)

            self._write_output(
                output_dir_per_run=output_dir_per_run,
                file_name="contract_lint.txt",
                content=lint_report or "(no issues)\n",
            )

            if not errors:
                break
            if repair_idx >= max_repairs:
                break

            revised = await architect.revise_contract(
                input_spec=spec,
                contract_json=contract_json,
                lint_errors=lint_report,
                golden_tb_path=golden_tb_path,
            )
            contract_json = revised.model_dump_json(indent=2, exclude_none=True) + "\n"

        self._write_output(output_dir_per_run=output_dir_per_run, file_name="contract.json", content=contract_json)
        return contract_json

    def _contract_only_context(self, contract_json: str) -> str:
        """Compact pseudo-spec for contract-only mode; avoids leaking ambiguous NL downstream."""
        if not self.config.contract_only:
            return ""
        try:
            obj = __import__("json").loads(contract_json)
        except Exception:  # noqa: BLE001
            return ""
        bullets = obj.get("functional_summary")
        if not isinstance(bullets, list) or not bullets:
            return ""
        items = [str(x) for x in bullets if isinstance(x, str) and x.strip()]
        if not items:
            return ""
        return "Contract-only context:\n" + "\n".join(f"- {s}" for s in items[:6]) + "\n"

    async def _run_instance(
        self,
        *,
        spec: str,
        output_dir_per_run: Path,
        golden_tb_path: str | None,
        golden_rtl_blackbox_path: str | None,
        usage_fns: List[Callable[[], Tuple[int, int]]],
    ) -> Tuple[bool, str, int, int]:
        architect = ArchitectAgent(self.cfg)
        tb_gen = TBGenerator(self.cfg)
        rtl_gen = RTLGenerator(self.cfg)
        sim_reviewer = SimReviewer(str(output_dir_per_run), golden_rtl_blackbox_path)
        rtl_edit = RTLEditor(self.cfg, sim_reviewer=sim_reviewer, max_trials=int(self.config.debug_max_trials))
        tracked_models = [
            architect._agent.model,
            tb_gen._agent.model,
            rtl_gen._agent.model,
            rtl_edit._agent.model,
        ]

        def usage_totals() -> Tuple[int, int]:
            total_in = 0
            total_out = 0
            # Include the Debugger's BooleanProofer/Asserter helper agents (created lazily).
            for model in tracked_models + rtl_edit.helper_models():
                input_tokens, output_tokens = get_model_usage(model)
                total_in += input_tokens
                total_out += output_tokens
            return total_in, total_out

        usage_fns.append(usage_totals)

        def finish(is_sim_pass: bool, rtl_code: str) -> Tuple[bool, str, int, int]:
            input_tokens, output_tokens = usage_totals()
            return is_sim_pass, rtl_code, input_tokens, output_tokens

        await architect.reset()
        contract_json = await self._build_contract_json(
            architect=architect,
            spec=spec,
            golden_tb_path=golden_tb_path,
            output_dir_per_run=output_dir_per_run,
        )
        try:
            contract_obj = __import__("json").loads(contract_json)
            module_name = str(contract_obj.get("module_name") or "TopModule")
        except Exception:  # noqa: BLE001
            module_name = "TopModule"
        try:
            self._write_output(
                output_dir_per_run=output_dir_per_run,
                file_name="architect_prompt.txt",
                content=(getattr(architect, "last_prompt", "") or "") + "\n",
            )
            self._write_output(
                output_dir_per_run=output_dir_per_run,
                file_name="architect_raw_output.txt",
                content=(getattr(architect, "last_raw_output", "") or "") + "\n",
            )
        except Exception:  # noqa: BLE001
            pass

        await tb_gen.reset()
        tb_gen.set_golden_tb_path(golden_tb_path)
        tb_input_spec = self._contract_only_context(contract_json) if self.config.contract_only else spec
        testbench, interface = await tb_gen.chat(tb_input_spec, contract_json=contract_json)
        testbench = self._augment_dumpvars_with_dut_scope(testbench, module_name=module_name)
        self._write_output(output_dir_per_run=output_dir_per_run, file_name="tb.sv", content=testbench)
        self._write_output(output_dir_per_run=output_dir_per_run, file_name="if.sv", content=interface)
        # If we were given a golden TB and the LLM accidentally broke its syntax,
        # fall back to the original golden TB (plus our safe dumpvars augmentation).
        tb_path = output_dir_per_run / "tb.sv"
        if golden_tb_path and self._tb_has_syntax_error(tb_path=tb_path):
            golden_text = Path(golden_tb_path).read_text(encoding="utf-8")
            golden_text = self._augment_dumpvars_with_dut_scope(golden_text, module_name=module_name)
            self._write_output(output_dir_per_run=output_dir_per_run, file_name="tb.sv", content=golden_text)
            testbench = golden_text
        elif not golden_tb_path:
            # Non-golden mode: lint the generated TB early and give the Verifier one chance to repair it.
            tb_ok, tb_lint_excerpt, tb_lint_json = self._tb_lint_report(tb_path=tb_path)
            if not tb_ok:
                self._write_output(output_dir_per_run=output_dir_per_run, file_name="tb_lint_failed_log.json", content=tb_lint_json)
                self._write_output(output_dir_per_run=output_dir_per_run, file_name="tb_lint_failed_excerpt.txt", content=tb_lint_excerpt)

                tb_gen.set_tb_lint_error(lint_log=tb_lint_excerpt, previous_tb=testbench)
                testbench, interface = await tb_gen.chat(tb_input_spec, contract_json=contract_json)
                testbench = self._augment_dumpvars_with_dut_scope(testbench, module_name=module_name)
                self._write_output(output_dir_per_run=output_dir_per_run, file_name="tb.sv", content=testbench)
                self._write_output(output_dir_per_run=output_dir_per_run, file_name="if.sv", content=interface)

                tb_ok2, tb_lint_excerpt2, tb_lint_json2 = self._tb_lint_report(tb_path=tb_path)
                if not tb_ok2:
                    self._write_output(output_dir_per_run=output_dir_per_run, file_name="tb_lint_failed_log_retry.json", content=tb_lint_json2)
                    self._write_output(output_dir_per_run=output_dir_per_run, file_name="tb_lint_failed_excerpt_retry.txt", content=tb_lint_excerpt2)
                    try:
                        self._write_output(
                            output_dir_per_run=output_dir_per_run,
                            file_name="verifier_prompt.txt",
                            content=(getattr(tb_gen, "last_prompt", "") or "") + "\n",
                        )
                        self._write_output(
                            output_dir_per_run=output_dir_per_run,
                            file_name="verifier_raw_output.txt",
                            content=(getattr(tb_gen, "last_raw_output", "") or "") + "\n",
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    return finish(False, "")
        try:
            self._write_output(
                output_dir_per_run=output_dir_per_run,
                file_name="verifier_prompt.txt",
                content=(getattr(tb_gen, "last_prompt", "") or "") + "\n",
            )
            self._write_output(
                output_dir_per_run=output_dir_per_run,
                file_name="verifier_raw_output.txt",
                content=(getattr(tb_gen, "last_raw_output", "") or "") + "\n",
            )
        except Exception:  # noqa: BLE001
            pass

        await rtl_gen.reset()
        rtl_path = str(output_dir_per_run / "rtl.sv")
        rtl_input_spec = self._contract_only_context(contract_json) if self.config.contract_only else spec
        is_syntax_pass, rtl_code = await rtl_gen.chat(
            input_spec=rtl_input_spec,
            testbench=testbench,
            interface=interface,
            rtl_path=rtl_path,
            contract_json=contract_json,
        )
        if not is_syntax_pass:
            try:
                self._write_output(
                    output_dir_per_run=output_dir_per_run,
                    file_name="coder_prompt.txt",
                    content=(getattr(rtl_gen, "last_prompt", "") or "") + "\n",
                )
                self._write_output(
                    output_dir_per_run=output_dir_per_run,
                    file_name="coder_raw_output.txt",
                    content=(getattr(rtl_gen, "last_raw_output", "") or "") + "\n",
                )
            except Exception:  # noqa: BLE001
                pass
            return finish(False, rtl_code)
        self._write_output(output_dir_per_run=output_dir_per_run, file_name="rtl.sv", content=rtl_code)
        try:
            self._write_output(
                output_dir_per_run=output_dir_per_run,
                file_name="coder_prompt.txt",
                content=(getattr(rtl_gen, "last_prompt", "") or "") + "\n",
            )
            self._write_output(
                output_dir_per_run=output_dir_per_run,
                file_name="coder_raw_output.txt",
                content=(getattr(rtl_gen, "last_raw_output", "") or "") + "\n",
            )
        except Exception:  # noqa: BLE001
            pass

        sim_log = ""
        is_sim_pass = False
        sim_mismatch_cnt = 0
        did_rtl_regen_after_mismatch = False

        # Retry simulation + RTL debug for mismatch-driven failures.
        # If a golden testbench is provided (e.g., VerilogEval), treat it as fixed and only debug RTL.
        remaining_debug_trials = int(self.config.debug_max_trials)
        did_fallback_to_golden_tb = False
        for _ in range(max(1, int(self.config.sim_max_retry))):
            is_sim_pass, sim_mismatch_cnt, sim_log = await asyncio.to_thread(sim_reviewer.review)
            if is_sim_pass:
                return finish(True, rtl_code)

            if sim_mismatch_cnt <= 0:
                if golden_tb_path and not did_fallback_to_golden_tb:
                    # The Verifier may have introduced SV features unsupported by Verilator
                    # (or broken the TB) even if it remains syntactically valid.
                    # In that case, fall back to the original golden TB and retry.
                    try:
                        tb_before = (output_dir_per_run / "tb.sv").read_text(encoding="utf-8")
                        self._write_output(
                            output_dir_per_run=output_dir_per_run,
                            file_name="tb_before_golden_fallback.sv",
                            content=tb_before,
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        self._write_output(
                            output_dir_per_run=output_dir_per_run,
                            file_name="sim_failed_log_before_tb_fallback.json",
                            content=sim_log,
                        )
                    except Exception:  # noqa: BLE001
                        pass

                    golden_text = Path(golden_tb_path).read_text(encoding="utf-8")
                    golden_text = self._augment_dumpvars_with_dut_scope(golden_text, module_name=module_name)
                    self._write_output(output_dir_per_run=output_dir_per_run, file_name="tb.sv", content=golden_text)
                    testbench = golden_text
                    did_fallback_to_golden_tb = True
                    continue

                # Non-mismatch failures (e.g., harness timeout or runtime error) don't fit the
                # mismatch-driven debug loop; return the last RTL without crashing.
                self._write_output(
                    output_dir_per_run=output_dir_per_run,
                    file_name="sim_failed_log.json",
                    content=sim_log,
                )
                return finish(False, rtl_code)

            if remaining_debug_trials <= 0:
                self._write_output(
                    output_dir_per_run=output_dir_per_run,
                    file_name="debug_budget_exhausted.txt",
                    content=f"Debugger budget exhausted (debug_max_trials={self.config.debug_max_trials}).\n",
                )
                return finish(False, rtl_code)

            # Before trace-based block editing, give the Coder one shot to regenerate RTL using the failure log.
            # This often fixes systematic contract/timing misunderstandings faster than local patching.
            if not did_rtl_regen_after_mismatch:
                did_rtl_regen_after_mismatch = True
                try:
                    rtl_gen.set_failed_trial(sim_log, rtl_code, testbench)
                    is_syntax_pass2, rtl_code2 = await rtl_gen.chat(
                        input_spec=rtl_input_spec,
                        testbench=testbench,
                        interface=interface,
                        rtl_path=rtl_path,
                        contract_json=contract_json,
                    )
                    if is_syntax_pass2:
                        rtl_code = rtl_code2
                        self._write_output(output_dir_per_run=output_dir_per_run, file_name="rtl_regen_after_mismatch.sv", content=rtl_code)
                        self._write_output(output_dir_per_run=output_dir_per_run, file_name="rtl.sv", content=rtl_code)
                        try:
                            self._write_output(
                                output_dir_per_run=output_dir_per_run,
                                file_name="coder_prompt_regen_after_mismatch.txt",
                                content=(getattr(rtl_gen, "last_prompt", "") or "") + "\n",
                            )
                            self._write_output(
                                output_dir_per_run=output_dir_per_run,
                                file_name="coder_raw_output_regen_after_mismatch.txt",
                                content=(getattr(rtl_gen, "last_raw_output", "") or "") + "\n",
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        continue
                except Exception:  # noqa: BLE001
                    pass
                # Regen failed: rtl_gen.chat writes every attempt to rtl.sv, so restore the last
                # syntax-correct RTL before the Debugger (which edits rtl.sv on disk) takes over.
                self._write_output(output_dir_per_run=output_dir_per_run, file_name="rtl.sv", content=rtl_code)

            # Snapshot the failing state for inspection/debugging.
            self._write_output(
                output_dir_per_run=output_dir_per_run,
                file_name="rtl_before_debug.sv",
                content=rtl_code,
            )
            # Write sim failure log as JSON (CommandResult model).
            self._write_output(
                output_dir_per_run=output_dir_per_run,
                file_name="sim_failed_log.json",
                content=sim_log,
            )

            await rtl_edit.reset()
            is_sim_pass, rtl_code, used_trials = await rtl_edit.chat(
                spec=(self._contract_only_context(contract_json) if self.config.contract_only else spec),
                output_dir_per_run=str(output_dir_per_run),
                sim_failed_log=sim_log,
                sim_mismatch_cnt=sim_mismatch_cnt,
                contract_json=contract_json,
                max_trials=remaining_debug_trials,
            )
            remaining_debug_trials = max(0, remaining_debug_trials - int(used_trials))
            if is_sim_pass:
                return finish(True, rtl_code)

        # Last check, just in case the final edit improved but didn't re-run.
        is_sim_pass, _, _ = await asyncio.to_thread(sim_reviewer.review)
        return finish(is_sim_pass, rtl_code)

    async def _run_instance_ablation(
        self,
        *,
        spec: str,
        output_dir_per_run: Path,
        usage_fns: List[Callable[[], Tuple[int, int]]],
    ) -> Tuple[bool, str, int, int]:
        rtl_gen = RTLGenerator(self.cfg, ablation=True)
        usage_fns.append(lambda: get_model_usage(rtl_gen._agent.model))
        await rtl_gen.reset()
        rtl_path = str(output_dir_per_run / "rtl.sv")
        is_syntax_pass, rtl_code = await rtl_gen.ablation_chat(input_spec=spec, rtl_path=rtl_path)
        self._write_output(output_dir_per_run=output_dir_per_run, file_name="rtl.sv", content=rtl_code)
        input_tokens, output_tokens = get_model_usage(rtl_gen._agent.model)
        return is_syntax_pass, rtl_code, input_tokens, output_tokens

    async def run(
        self,
        *,
        spec: str,
        output_dir_per_run: Path,
        golden_tb_path: str | None = None,
        golden_rtl_blackbox_path: str | None = None,
    ) -> TopAgentResult:
        output_dir_per_run = output_dir_per_run.expanduser().resolve()
        output_dir_per_run.mkdir(parents=True, exist_ok=True)

        tb_path = str(output_dir_per_run / "tb.sv")
        if_path = str(output_dir_per_run / "if.sv")
        rtl_path = str(output_dir_per_run / "rtl.sv")

        tag = output_dir_per_run / "properly_finished.tag"
        if tag.exists():
            tag.unlink()

        # Lets the error path still report tokens spent before the exception.
        usage_fns: List[Callable[[], Tuple[int, int]]] = []
        try:
            if re.search(r"\s", str(output_dir_per_run)):
                # Verilator's generated Makefiles refuse such paths; fail clearly instead of
                # surfacing it later as an RTL syntax/simulation failure.
                raise ValueError(
                    f"Output directory path contains whitespace, which Verilator cannot build in: {output_dir_per_run}"
                )
            if self.config.is_ablation:
                is_sim_pass, rtl_code, input_tokens, output_tokens = await self._run_instance_ablation(
                    spec=spec, output_dir_per_run=output_dir_per_run, usage_fns=usage_fns
                )
            else:
                is_sim_pass, rtl_code, input_tokens, output_tokens = await self._run_instance(
                    spec=spec,
                    output_dir_per_run=output_dir_per_run,
                    golden_tb_path=golden_tb_path,
                    golden_rtl_blackbox_path=golden_rtl_blackbox_path,
                    usage_fns=usage_fns,
                )
            tag.write_text("1", encoding="utf-8")
            return TopAgentResult(
                output_dir_per_run=str(output_dir_per_run),
                rtl_path=rtl_path,
                tb_path=tb_path,
                if_path=if_path,
                is_sim_pass=is_sim_pass,
                rtl_code=rtl_code,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as e:  # noqa: BLE001
            input_tokens, output_tokens = 0, 0
            for fn in usage_fns:
                try:
                    i, o = fn()
                except Exception:  # noqa: BLE001
                    continue
                input_tokens += i
                output_tokens += o
            return TopAgentResult(
                output_dir_per_run=str(output_dir_per_run),
                rtl_path=rtl_path,
                tb_path=tb_path,
                if_path=if_path,
                is_sim_pass=False,
                rtl_code="",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=f"{type(e).__name__}: {e}",
            )
