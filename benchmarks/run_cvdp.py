"""Run the Veri-Sure framework on the NVIDIA CVDP benchmark dataset.

Usage examples:
    # Run on all Veri-Sure-compatible problems from HuggingFace (non-agentic, open-source)
    python benchmarks/run_cvdp.py --stream

    # Run on a specific config
    python benchmarks/run_cvdp.py --config cvdp_agentic_code_generation_no_commercial

    # Filter by task category (cid003 = spec-to-RTL, cid016 = debugging)
    python benchmarks/run_cvdp.py --cid cid003 --cid cid016

    # Filter by difficulty
    python benchmarks/run_cvdp.py --difficulty easy --difficulty medium

    # Use local JSONL file instead of HuggingFace
    python benchmarks/run_cvdp.py --jsonl-path /path/to/cvdp_v1.0.4_nonagentic_code_generation_no_commercial.jsonl

    # Run with Docker-based evaluation (requires Docker)
    python benchmarks/run_cvdp.py --eval-mode docker

    # Run first 10 problems only
    python benchmarks/run_cvdp.py --limit 10

    # Run specific problems by ID
    python benchmarks/run_cvdp.py --problem cvdp_copilot_alu_0001
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import time
from collections import Counter

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.cvdp_adapter import (
    CVDPProblem,
    CVDPEvalResult,
    CID_LABELS,
    VERISURE_COMPATIBLE_CIDS,
    load_cvdp_from_huggingface,
    load_cvdp_from_jsonl,
    filter_problems,
    summarize_dataset,
    setup_cvdp_workspace,
    place_generated_code,
    evaluate_with_cvdp_harness,
)
from eda_agent.config import add_llm_arguments, llm_config_from_args
from eda_agent.verilator_judge import make_timestamped_dir
from eda_agent.top_agent import TopAgent, TopAgentConfig


# ---------------------------------------------------------------------------
# Outcome tracking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProblemOutcome:
    problem_id: str
    cid: str
    difficulty: str
    verisure_passed: bool       # Veri-Sure's internal sim passed
    cvdp_eval: CVDPEvalResult | None  # CVDP harness evaluation result
    passfail: str               # Single-char summary
    input_tokens: int
    output_tokens: int
    error: str | None = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Veri-Sure on the NVIDIA CVDP benchmark dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data source
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--jsonl-path",
        default=None,
        help="Path to a local CVDP JSONL file",
    )
    grp.add_argument(
        "--config",
        default="cvdp_nonagentic_code_generation_no_commercial",
        choices=[
            "cvdp_nonagentic_code_generation_no_commercial",
            "cvdp_nonagentic_code_generation_commercial",
            "cvdp_agentic_code_generation_no_commercial",
            "cvdp_agentic_code_generation_commercial",
        ],
        help="HuggingFace dataset config name (default: nonagentic, no_commercial)",
    )

    # Filtering
    parser.add_argument("--cid", action="append", default=None, help="Filter by CID (repeatable)")
    parser.add_argument("--difficulty", action="append", default=None, help="Filter by difficulty (repeatable)")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N problems")
    parser.add_argument("--problem", action="append", default=None, help="Run specific problem ID(s)")
    parser.add_argument(
        "--verisure-compatible-only",
        action="store_true",
        default=True,
        help="Only run tasks where Veri-Sure is applicable (default: True)",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Include all task types (including TB/assertion generation)",
    )

    # Veri-Sure config
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--sim-max-retry", type=int, default=4)
    parser.add_argument("--debug-max-trials", type=int, default=30)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--jobs", type=int, default=1, help="Max concurrent evaluations")

    # Evaluation
    parser.add_argument(
        "--eval-mode",
        choices=["docker", "direct", "none"],
        default="none",
        help="CVDP harness evaluation mode (default: none = Veri-Sure internal only)",
    )
    parser.add_argument("--eval-timeout", type=float, default=300.0)

    # Output
    parser.add_argument("--out-root", default="runs/cvdp", help="Root output directory")

    # Model config
    add_llm_arguments(parser)

    args = parser.parse_args(argv)

    # ------- Load model config -------
    cfg = llm_config_from_args(args)
    missing = cfg.missing_credentials_message()
    if missing:
        raise SystemExit(missing)
    # Record the effective (provider-resolved) settings in the run name and metadata below.
    args.temperature, args.top_p, args.max_completion_tokens = cfg.temperature, cfg.top_p, cfg.max_tokens

    # ------- Load dataset -------
    print("Loading CVDP dataset...")
    if args.jsonl_path:
        problems = load_cvdp_from_jsonl(args.jsonl_path)
        config_name = Path(args.jsonl_path).stem
    else:
        config_name = args.config
        problems = load_cvdp_from_huggingface(config=config_name)

    # ------- Filter -------
    verisure_only = args.verisure_compatible_only and not args.all_tasks
    cids = set(args.cid) if args.cid else None
    difficulties = set(args.difficulty) if args.difficulty else None

    problems = filter_problems(
        problems,
        cids=cids,
        difficulties=difficulties,
        verisure_compatible_only=verisure_only,
        limit=args.limit,
    )

    if args.problem:
        wanted = set(args.problem)
        problems = [p for p in problems if p.id in wanted]
        missing = sorted(wanted - {p.id for p in problems})
        if missing:
            print(f"Warning: problem IDs not found: {', '.join(missing)}")

    if not problems:
        raise SystemExit("No problems to evaluate after filtering.")

    print(f"\n{summarize_dataset(problems)}\n")

    # ------- Set up output directory -------
    out_root = Path(args.out_root).expanduser().resolve()
    tag_parts = [
        config_name,
        f"t{args.temperature}",
        f"n{args.samples}",
        f"r{args.sim_max_retry}",
    ]
    if args.ablation:
        tag_parts.append("ablation")
    run_dir = make_timestamped_dir(out_root, "_".join(tag_parts))
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run config.
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "config": config_name,
                "cids": list(cids) if cids else None,
                "difficulties": list(difficulties) if difficulties else None,
                "samples": args.samples,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_completion_tokens": args.max_completion_tokens,
                "sim_max_retry": args.sim_max_retry,
                "debug_max_trials": args.debug_max_trials,
                "ablation": args.ablation,
                "eval_mode": args.eval_mode,
                "provider": cfg.provider,
                "model": cfg.model,
                "base_url": cfg.base_url,
                "total_problems": len(problems),
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    # ------- Run evaluation -------
    sem = asyncio.Semaphore(max(1, args.jobs))
    outcomes: list[ProblemOutcome] = []

    async def eval_one(prob: CVDPProblem, sample_idx: int) -> ProblemOutcome:
        async with sem:
            problem_dir = run_dir / f"{prob.id}_sample{sample_idx:02d}"
            problem_dir.mkdir(parents=True, exist_ok=True)

            # Write problem info.
            (problem_dir / "problem_info.json").write_text(
                json.dumps(
                    {
                        "id": prob.id,
                        "cid": prob.cid,
                        "cid_label": prob.cid_label,
                        "difficulty": prob.difficulty,
                        "is_agentic": prob.is_agentic,
                        "output_targets": list(prob.output_targets.keys()),
                    },
                    indent=2,
                    ensure_ascii=False,
                ) + "\n",
                encoding="utf-8",
            )

            # Write the prompt for debugging.
            spec = prob.build_verisure_spec()
            (problem_dir / "verisure_spec.txt").write_text(spec, encoding="utf-8")

            # Run Veri-Sure TopAgent.
            run_sample_dir = problem_dir / "_verisure_run"
            run_sample_dir.mkdir(parents=True, exist_ok=True)

            try:
                agent = TopAgent(
                    cfg,
                    config=TopAgentConfig(
                        sim_max_retry=args.sim_max_retry,
                        debug_max_trials=args.debug_max_trials,
                        is_ablation=args.ablation,
                    ),
                )

                # CVDP problems don't have Verilator-compatible golden TBs,
                # so we run without golden_tb_path / golden_rtl_blackbox_path.
                # Veri-Sure will generate its own TB via the Verifier agent.
                run_result = await agent.run(
                    spec=spec,
                    output_dir_per_run=run_sample_dir,
                )

                input_tokens = int(getattr(run_result, "input_tokens", 0) or 0)
                output_tokens = int(getattr(run_result, "output_tokens", 0) or 0)
                rtl = run_result.rtl_code.strip()

                if run_result.error or not rtl:
                    return ProblemOutcome(
                        problem_id=prob.id,
                        cid=prob.cid,
                        difficulty=prob.difficulty,
                        verisure_passed=False,
                        cvdp_eval=None,
                        passfail="E",
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        error=run_result.error,
                    )

                # Save generated RTL.
                (problem_dir / "generated_rtl.sv").write_text(
                    rtl.rstrip() + "\n", encoding="utf-8"
                )

                verisure_passed = run_result.is_sim_pass

                # CVDP harness evaluation (optional).
                cvdp_eval = None
                if args.eval_mode != "none":
                    workspace = problem_dir / "_cvdp_workspace"
                    setup_cvdp_workspace(prob, workspace)
                    place_generated_code(prob, workspace, rtl)

                    cvdp_eval = await asyncio.to_thread(
                        evaluate_with_cvdp_harness,
                        prob,
                        workspace,
                        use_docker=(args.eval_mode == "docker"),
                        timeout_s=args.eval_timeout,
                    )

                    (problem_dir / "cvdp_eval.json").write_text(
                        json.dumps(
                            {
                                "passed": cvdp_eval.passed,
                                "score": cvdp_eval.score,
                                "target_score": cvdp_eval.target_score,
                                "error": cvdp_eval.error,
                            },
                            indent=2,
                            ensure_ascii=False,
                        ) + "\n",
                        encoding="utf-8",
                    )
                    (problem_dir / "cvdp_eval_log.txt").write_text(
                        cvdp_eval.log, encoding="utf-8"
                    )

                # Determine passfail character.
                if cvdp_eval is not None:
                    passfail = "." if cvdp_eval.passed else "F"
                else:
                    passfail = "." if verisure_passed else "F"

                return ProblemOutcome(
                    problem_id=prob.id,
                    cid=prob.cid,
                    difficulty=prob.difficulty,
                    verisure_passed=verisure_passed,
                    cvdp_eval=cvdp_eval,
                    passfail=passfail,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            except Exception as e:  # noqa: BLE001
                return ProblemOutcome(
                    problem_id=prob.id,
                    cid=prob.cid,
                    difficulty=prob.difficulty,
                    verisure_passed=False,
                    cvdp_eval=None,
                    passfail="E",
                    input_tokens=0,
                    output_tokens=0,
                    error=f"{type(e).__name__}: {e}",
                )

    # Build work items.
    items: list[tuple[CVDPProblem, int]] = []
    for prob in problems:
        for s in range(1, args.samples + 1):
            items.append((prob, s))

    total = len(items)
    passed = 0
    done = 0
    total_input_tokens = 0
    total_output_tokens = 0

    pbar = tqdm(total=total, desc="cvdp-bench", unit="problem")
    try:
        tasks = [asyncio.create_task(eval_one(prob, s)) for prob, s in items]
        for fut in asyncio.as_completed(tasks):
            outcome = await fut
            outcomes.append(outcome)
            total_input_tokens += outcome.input_tokens
            total_output_tokens += outcome.output_tokens
            done += 1
            if outcome.passfail == ".":
                passed += 1
            pbar.update(1)
            pbar.set_postfix_str(f"pass={passed}/{done}")
    finally:
        pbar.close()

    # ------- Generate summary -------
    _write_summary(run_dir, outcomes, problems, args, total_input_tokens, total_output_tokens)

    print(f"\nDone. Results in: {run_dir}")
    print(f"Summary: {run_dir / 'summary.txt'}")
    return 0


def _write_summary(
    run_dir: Path,
    outcomes: list[ProblemOutcome],
    problems: list[CVDPProblem],
    args: argparse.Namespace,
    total_input_tokens: int,
    total_output_tokens: int,
) -> None:
    """Write summary files: summary.txt, summary.csv, per-category breakdown, token usage."""

    lines: list[str] = ["CVDP Benchmark Results", "=" * 60, ""]

    # Overall pass rate.
    n_total = len(outcomes)
    n_passed = sum(1 for o in outcomes if o.passfail == ".")
    overall_rate = (n_passed / n_total * 100) if n_total > 0 else 0
    lines.append(f"Overall: {n_passed}/{n_total} ({overall_rate:.1f}%)")
    lines.append("")

    # Per-category breakdown.
    cid_outcomes: dict[str, list[ProblemOutcome]] = {}
    for o in outcomes:
        cid_outcomes.setdefault(o.cid, []).append(o)

    lines.append("Per-Category Results:")
    lines.append(f"{'CID':<10} {'Task':<35} {'Pass':>6} {'Total':>6} {'Rate':>8}")
    lines.append("-" * 70)
    for cid in sorted(cid_outcomes):
        outs = cid_outcomes[cid]
        p = sum(1 for o in outs if o.passfail == ".")
        t = len(outs)
        rate = (p / t * 100) if t > 0 else 0
        label = CID_LABELS.get(cid, cid)
        lines.append(f"{cid:<10} {label:<35} {p:>6} {t:>6} {rate:>7.1f}%")

    lines.append("")

    # Per-difficulty breakdown.
    diff_outcomes: dict[str, list[ProblemOutcome]] = {}
    for o in outcomes:
        diff_outcomes.setdefault(o.difficulty, []).append(o)

    lines.append("Per-Difficulty Results:")
    lines.append(f"{'Difficulty':<12} {'Pass':>6} {'Total':>6} {'Rate':>8}")
    lines.append("-" * 35)
    for diff in ["easy", "medium", "hard"]:
        if diff in diff_outcomes:
            outs = diff_outcomes[diff]
            p = sum(1 for o in outs if o.passfail == ".")
            t = len(outs)
            rate = (p / t * 100) if t > 0 else 0
            lines.append(f"{diff:<12} {p:>6} {t:>6} {rate:>7.1f}%")

    lines.append("")

    # Per-problem detail.
    lines.append("Per-Problem Results:")
    lines.append(f"{'Problem ID':<60} {'CID':<8} {'Diff':<8} {'Result':<8}")
    lines.append("-" * 90)
    for o in sorted(outcomes, key=lambda x: x.problem_id):
        status = "PASS" if o.passfail == "." else ("ERROR" if o.passfail == "E" else "FAIL")
        lines.append(f"{o.problem_id:<60} {o.cid:<8} {o.difficulty:<8} {status:<8}")

    # Write summary.txt
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Write summary.csv
    csv_lines = ["problem_id,cid,difficulty,passfail,verisure_passed,cvdp_score,input_tokens,output_tokens"]
    for o in sorted(outcomes, key=lambda x: x.problem_id):
        cvdp_score = f"{o.cvdp_eval.score:.3f}" if o.cvdp_eval else ""
        csv_lines.append(
            f"{o.problem_id},{o.cid},{o.difficulty},{o.passfail},"
            f"{o.verisure_passed},{cvdp_score},{o.input_tokens},{o.output_tokens}"
        )
    (run_dir / "summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    # Write token usage.
    (run_dir / "token_usage.json").write_text(
        json.dumps(
            {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    # Write per-category summary as JSON (useful for paper tables).
    cat_summary = {}
    for cid in sorted(cid_outcomes):
        outs = cid_outcomes[cid]
        p = sum(1 for o in outs if o.passfail == ".")
        t = len(outs)
        cat_summary[cid] = {
            "label": CID_LABELS.get(cid, cid),
            "passed": p,
            "total": t,
            "pass_rate": round(p / t * 100, 2) if t > 0 else 0,
        }
    (run_dir / "category_summary.json").write_text(
        json.dumps(cat_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
