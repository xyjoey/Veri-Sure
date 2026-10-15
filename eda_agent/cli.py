from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .config import add_llm_arguments, llm_config_from_args
from .verilator_judge import make_timestamped_dir
from .top_agent import TopAgent, TopAgentConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eda-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("run", help="Run multi-agent RTL loop")
    m.add_argument("--prompt", required=True, help="Natural language spec")
    m.add_argument("--golden-tb", default=None, help="Optional golden testbench path (e.g. VerilogEval *_test.sv)")
    m.add_argument(
        "--golden-ref",
        default=None,
        help="Optional golden reference/blackbox path (e.g. VerilogEval *_ref.sv defining RefModule)",
    )
    m.add_argument("--runs-root", default="runs", help="Directory to store artifacts")
    m.add_argument("--sim-max-retry", type=int, default=4)
    m.add_argument(
        "--debug-max-trials",
        type=int,
        default=30,
        help="Max Debugger (RTLEditor) edit attempts per failing simulation.",
    )
    m.add_argument("--ablation", action="store_true", help="Ablation mode: only RTL generation + syntax check")
    add_llm_arguments(m)

    args = parser.parse_args(argv)

    if args.cmd == "run":
        cfg = llm_config_from_args(args)
        missing = cfg.missing_credentials_message()
        if missing:
            raise SystemExit(missing)

        runs_root = Path(args.runs_root)
        run_dir = make_timestamped_dir(runs_root, "run")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "prompt.txt").write_text(args.prompt + "\n", encoding="utf-8")

        agent = TopAgent(
            cfg,
            config=TopAgentConfig(
                sim_max_retry=args.sim_max_retry,
                debug_max_trials=args.debug_max_trials,
                is_ablation=args.ablation,
            ),
        )
        result = asyncio.run(
            agent.run(
                spec=args.prompt,
                output_dir_per_run=run_dir,
                golden_tb_path=args.golden_tb,
                golden_rtl_blackbox_path=args.golden_ref,
            )
        )

        status = "PASS" if result.is_sim_pass and not result.error else "FAIL"
        print(status)
        print(f"model: {cfg.provider}/{cfg.model}")
        print(f"run_dir: {result.output_dir_per_run}")
        print(f"rtl: {result.rtl_path}")
        print(f"testbench: {result.tb_path}")
        if result.error:
            print(f"error: {result.error}")
        return 0 if status == "PASS" else 2

    raise AssertionError("unreachable")
