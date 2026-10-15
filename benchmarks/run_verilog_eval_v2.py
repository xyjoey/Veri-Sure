from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import time

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eda_agent.config import add_llm_arguments, llm_config_from_args
from eda_agent.verilator_judge import make_timestamped_dir, run_verilator_testbench_sync
from eda_agent.top_agent import TopAgent, TopAgentConfig


PASSFAIL_MISMATCH_RE = re.compile(r"^\s*Mismatches:\s*(\d+)\s*in\s*(\d+)\s*samples\s*$", re.MULTILINE)


@dataclass(frozen=True)
class SampleOutcome:
    passed: bool
    passfail: str
    mismatches: int | None
    timed_out: bool


@dataclass(frozen=True)
class WorkItem:
    problem: str
    sample_idx: int
    prompt_text: str
    prompt_path: Path
    test_path: Path
    ref_path: Path


def _filter_verilator_warnings(log: str) -> str:
    """Remove Verilator `%Warning-*` blocks from a combined compile+run log.

    Verilator warnings are typically multi-line blocks:
      - A header line starting with "%Warning-..."
      - Followed by context/notes lines that are indented (or start with "...").
    """
    out_lines: list[str] = []
    skipping_block = False
    for line in log.splitlines(keepends=True):
        if line.startswith("%Warning"):
            skipping_block = True
            continue
        if skipping_block:
            # Continuation/context lines are indented or are "...".
            if line.startswith((" ", "\t")) or line.lstrip().startswith("..."):
                continue
            # End of this warning block; handle this line normally.
            skipping_block = False
        out_lines.append(line)
    return "".join(out_lines)


def _read_problems(dataset_dir: Path, *, limit: int | None) -> list[str]:
    problems_file = dataset_dir / "problems.txt"
    if problems_file.exists():
        problems = [
            line.strip()
            for line in problems_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        # Drop repeated entries (keeping order): duplicates would run twice into the same output
        # directory, racing each other under --jobs > 1 and double-counting tokens.
        problems = list(dict.fromkeys(problems))
    else:
        problems = sorted(
            {
                p.name[: -len("_prompt.txt")]
                for p in dataset_dir.glob("*_prompt.txt")
            },
        )
    if limit is not None:
        return problems[:limit]
    return problems


def _classify_passfail(compile_log: str, verilog_text: str) -> tuple[str, int | None]:
    # First priority: if the golden TB prints a mismatch summary, treat it as source of truth.
    mismatches: int | None = None
    for line in compile_log.splitlines():
        m = PASSFAIL_MISMATCH_RE.match(line)
        if m:
            mismatches = int(m.group(1))
            if mismatches == 0:
                return ".", 0

    error_C = False
    error_p = False
    no_mismatch = False
    saw_timeout = False

    for line in compile_log.splitlines():
        if "syntax error" in line:
            return "S", None
        # Verilator/clang errors; avoid matching substrings like "errors_q".
        if line.lstrip().startswith("%Error"):
            error_C = True
        if re.search(r":\s*error:\s", line, flags=re.IGNORECASE):
            error_C = True
        if "error: This assignment requires an explicit cast" in line:
            return "e", None
        if "error: Sized numeric constant must have a size greater than zero" in line:
            return "0", None
        if "warning: always_comb process has no sensitivities" in line:
            return "n", None
        if "found no sensitivities so it will never trigger" in line:
            return "n", None
        if "is declared here as wire" in line:
            return "w", None
        if "Unknown module type" in line:
            return "m", None
        if "Unable to bind wire/reg" in line:
            error_p = True
        if "Unable to bind wire/reg/memory `clk'" in line:
            return "c", None
        if "TIMEOUT" in line:
            saw_timeout = True

        if mismatches is None:
            m = PASSFAIL_MISMATCH_RE.match(line)
            if m:
                mismatches = int(m.group(1))
                if mismatches == 0:
                    no_mismatch = True

    if error_p:
        return "p", mismatches
    if error_C:
        return "C", mismatches
    if no_mismatch:
        return ".", 0
    if saw_timeout:
        return "T", mismatches

    # Runtime issues / unknown. Try to detect reset-sensitivity pitfalls.
    for line in verilog_text.splitlines():
        if "posedge reset" in line or "negedge reset" in line or "posedge r)" in line:
            return "r", mismatches
    return "R", mismatches


def _render_row(passfails: list[str]) -> str:
    parts: list[str] = []
    for idx, ch in enumerate(passfails):
        if idx != 0 and idx % 5 == 0:
            parts.append(" ")
        parts.append(ch)
    return "".join(parts)


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verilog-eval-dir",
        default="benchmarks/verilogeval-v2-ext",
        help="Path to verilogeval-v2-ext checkout",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Override dataset directory path (absolute or relative); "
        "skips --verilog-eval-dir/--task resolution when provided",
    )
    parser.add_argument(
        "--task",
        choices=["spec-to-rtl", "code-complete-iccad2023"],
        default="spec-to-rtl",
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--sim-max-retry", type=int, default=4)
    parser.add_argument(
        "--debug-max-trials",
        type=int,
        default=30,
        help="Max Debugger (RTLEditor) edit attempts per failing simulation.",
    )
    parser.add_argument("--ablation", action="store_true", help="Ablation mode: only RTL generation + syntax check")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=None, help="Only run first N problems")
    parser.add_argument(
        "--problem",
        action="append",
        default=None,
        help="Only run a specific problem name (repeatable), e.g. --problem Prob004_vector2",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Max number of concurrent (problem,sample) evaluations",
    )

    parser.add_argument("--out-root", default="runs/verilog-eval-v2", help="Root directory for outputs")
    add_llm_arguments(parser)

    args = parser.parse_args(argv)

    cfg = llm_config_from_args(args)
    missing = cfg.missing_credentials_message()
    if missing:
        raise SystemExit(missing)
    # Record the effective (provider-resolved) settings in the run name and metadata below.
    args.temperature, args.top_p, args.max_completion_tokens = cfg.temperature, cfg.top_p, cfg.max_tokens

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    else:
        verilog_eval_dir = Path(args.verilog_eval_dir).expanduser().resolve()
        dataset_dir = verilog_eval_dir / f"dataset_{args.task}"
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset dir not found: {dataset_dir}")

    problems = _read_problems(dataset_dir, limit=args.limit)
    if args.problem:
        wanted = set(args.problem)
        problems = [p for p in problems if p in wanted]
        missing = sorted(wanted - set(problems))
        if missing:
            raise SystemExit(f"Unknown problem(s): {', '.join(missing)}")
    if not problems:
        raise SystemExit(f"No problems found under {dataset_dir}")

    out_root = Path(args.out_root).expanduser().resolve()
    run_dir = make_timestamped_dir(
        out_root,
        "verilogeval-v2-ext_"
        f"{args.task}_t{args.temperature}_p{args.top_p}_n{args.samples}"
        f"_r{args.sim_max_retry}"
        f"{'_ablation' if args.ablation else ''}",
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "task": args.task,
                "samples": args.samples,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_completion_tokens": args.max_completion_tokens,
                "sim_max_retry": args.sim_max_retry,
                "debug_max_trials": args.debug_max_trials,
                "ablation": args.ablation,
                "timeout_s": args.timeout_s,
                "dataset_dir": str(dataset_dir),
                "provider": cfg.provider,
                "model": cfg.model,
                "base_url": cfg.base_url,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    results: dict[str, list[SampleOutcome]] = {}

    items: list[WorkItem] = []
    for problem in problems:
        prompt_path = dataset_dir / f"{problem}_prompt.txt"
        test_path = dataset_dir / f"{problem}_test.sv"
        ref_path = dataset_dir / f"{problem}_ref.sv"

        if not prompt_path.exists():
            raise SystemExit(f"Missing prompt: {prompt_path}")
        if not test_path.exists():
            raise SystemExit(f"Missing test: {test_path}")
        if not ref_path.exists():
            raise SystemExit(f"Missing ref: {ref_path}")

        prompt_text = prompt_path.read_text(encoding="utf-8")
        for sample_idx in range(1, args.samples + 1):
            items.append(
                WorkItem(
                    problem=problem,
                    sample_idx=sample_idx,
                    prompt_text=prompt_text,
                    prompt_path=prompt_path,
                    test_path=test_path,
                    ref_path=ref_path,
                ),
            )

    sem = asyncio.Semaphore(max(1, args.jobs))

    async def eval_one(item: WorkItem) -> tuple[str, int, SampleOutcome, int, int]:
        async with sem:
            problem = item.problem
            sample_idx = item.sample_idx
            sample_str = f"{sample_idx:02}"

            problem_dir = run_dir / problem
            problem_dir.mkdir(parents=True, exist_ok=True)

            sample_sv_path = problem_dir / f"{problem}_sample{sample_str}.sv"
            sample_generate_log = problem_dir / f"{problem}_sample{sample_str}-sv-generate.log"
            sample_test_log = problem_dir / f"{problem}_sample{sample_str}-sv-vl-test.log"
            run_sample_dir = problem_dir / f"_run_sample{sample_str}"
            run_sample_dir.mkdir(parents=True, exist_ok=True)
            golden_dir = problem_dir / f"_golden_sample{sample_str}"
            golden_dir.mkdir(parents=True, exist_ok=True)

            try:
                agent = TopAgent(
                    cfg,
                    config=TopAgentConfig(
                        sim_max_retry=args.sim_max_retry,
                        debug_max_trials=args.debug_max_trials,
                        is_ablation=args.ablation,
                    ),
                )

                run_result = await agent.run(
                    spec=item.prompt_text,
                    output_dir_per_run=run_sample_dir,
                    golden_tb_path=str(item.test_path),
                    golden_rtl_blackbox_path=str(item.ref_path),
                )
                input_tokens = int(getattr(run_result, "input_tokens", 0) or 0)
                output_tokens = int(getattr(run_result, "output_tokens", 0) or 0)
                rtl = run_result.rtl_code.strip()
                trace_report_path = run_sample_dir / "trace_report.json"
                trace_report_summary = ""
                if trace_report_path.exists():
                    try:
                        trace_data = json.loads(trace_report_path.read_text(encoding="utf-8"))
                        trace_report_summary = json.dumps(
                            {
                                "fail_time": trace_data.get("fail_time"),
                                "total_mismatches": trace_data.get("total_mismatches"),
                                "vcd_available": (trace_data.get("notes") or {}).get("vcd_available"),
                                "suspect_blocks": len(trace_data.get("suspect_blocks") or []),
                            },
                            ensure_ascii=False,
                        )
                    except Exception:  # noqa: BLE001
                        trace_report_summary = ""

                sample_generate_log.write_text(
                    "\n".join(
                        [
                            f"problem     = {problem}",
                            f"model       = {cfg.model}",
                            f"temperature = {args.temperature}",
                            f"top_p       = {args.top_p}",
                            f"max_tokens  = {args.max_completion_tokens}",
                            f"prompt_tokens = {input_tokens}",
                            f"resp_tokens   = {output_tokens}",
                            f"total_tokens  = {input_tokens + output_tokens}",
                            "",
                            f"sim_max_retry         = {args.sim_max_retry}",
                            f"ablation              = {args.ablation}",
                            "",
                            "prompt:",
                            item.prompt_text,
                            "",
                            f"output_dir = {run_result.output_dir_per_run}",
                            f"error      = {run_result.error or ''}",
                            f"trace_report = {trace_report_path if trace_report_path.exists() else ''}",
                            f"trace_summary = {trace_report_summary}",
                        ],
                    )
                    + "\n",
                    encoding="utf-8",
                )

                if run_result.error or not rtl:
                    sample_sv_path.write_text(
                        "// ERROR: failed to produce RTL\n",
                        encoding="utf-8",
                    )
                    sample_test_log.write_text(
                        f"ERROR: {run_result.error}\n",
                        encoding="utf-8",
                    )
                    return problem, sample_idx, SampleOutcome(
                        passed=False,
                        passfail="E",
                        mismatches=None,
                        timed_out=False,
                    ), input_tokens, output_tokens

                sample_sv_path.write_text(rtl.rstrip() + "\n", encoding="utf-8")

                dut_path = golden_dir / f"{problem}_sample{sample_str}.sv"
                dut_path.write_text(rtl.rstrip() + "\n", encoding="utf-8")

                result = await asyncio.to_thread(
                    run_verilator_testbench_sync,
                    run_dir=golden_dir,
                    top="tb",
                    verilog_files=[dut_path, item.test_path, item.ref_path],
                    timeout_s=args.timeout_s,
                )

                filtered_log = _filter_verilator_warnings(result.log)
                sample_test_log.write_text(filtered_log, encoding="utf-8")

                passfail, mismatches = _classify_passfail(filtered_log, rtl)
                outcome = SampleOutcome(
                    passed=(passfail == "."),
                    passfail=passfail,
                    mismatches=mismatches,
                    timed_out=result.timed_out,
                )
                return problem, sample_idx, outcome, input_tokens, output_tokens

            except Exception as e:  # noqa: BLE001
                sample_sv_path.write_text(
                    "// ERROR: sample generation failed\n",
                    encoding="utf-8",
                )
                sample_generate_log.write_text(
                    f"ERROR: {type(e).__name__}: {e}\n",
                    encoding="utf-8",
                )
                sample_test_log.write_text(
                    f"ERROR: {type(e).__name__}: {e}\n",
                    encoding="utf-8",
                )
                return problem, sample_idx, SampleOutcome(
                    passed=False,
                    passfail="E",
                    mismatches=None,
                    timed_out=False,
                ), 0, 0

    total_units = len(items)
    passed_units = 0
    done_units = 0
    per_problem: dict[str, dict[int, SampleOutcome]] = {}
    total_input_tokens = 0
    total_output_tokens = 0

    pbar = tqdm(total=total_units, desc="verilogeval-v2-ext", unit="sample")
    try:
        tasks = [asyncio.create_task(eval_one(item)) for item in items]
        for fut in asyncio.as_completed(tasks):
            problem, sample_idx, outcome, input_tokens, output_tokens = await fut
            per_problem.setdefault(problem, {})[sample_idx] = outcome
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            done_units += 1
            if outcome.passfail == ".":
                passed_units += 1
            pbar.update(1)
            pbar.set_postfix_str(f"pass={passed_units}/{done_units}")
    finally:
        pbar.close()

    for problem in problems:
        sample_map = per_problem.get(problem, {})
        results[problem] = [
            sample_map.get(i, SampleOutcome(False, "E", None, False))
            for i in range(1, args.samples + 1)
        ]

    # Write summary
    summary_rows: list[str] = []
    summary_csv_lines: list[str] = []
    pass_rate_sum = 0.0

    problem_width = max(len(p) for p in results)
    for problem, outcomes in results.items():
        passfails = [o.passfail for o in outcomes]
        npass = sum(1 for o in outcomes if o.passfail == ".")
        nsamples = len(outcomes)
        pass_rate = int((npass / nsamples) * 100)
        pass_rate_sum += pass_rate

        summary_rows.append(
            f"{problem:{problem_width}} [{npass:02}/{nsamples:02}]({pass_rate:3}%) {_render_row(passfails)}",
        )
        summary_csv_lines.append(
            ",".join([problem, str(npass), str(nsamples), str(pass_rate / 100.0), *passfails]),
        )

    (run_dir / "summary.txt").write_text(
        "\n".join([""] + summary_rows + ["", f"pass_rate = {pass_rate_sum / len(results):.2f}", ""])
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.csv").write_text(
        "\n".join(summary_csv_lines) + "\n",
        encoding="utf-8",
    )
    (run_dir / "token_usage.json").write_text(
        json.dumps(
            {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Done. Results in: {run_dir}")
    print(f"Summary: {run_dir / 'summary.txt'}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
