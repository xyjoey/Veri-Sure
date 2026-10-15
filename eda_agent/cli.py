from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .config import load_openai_config
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
    m.add_argument("--model", default=None, help="OpenAI model name (default: env OPENAI_MODEL)")
    m.add_argument("--base-url", default=None, help="OpenAI base_url (default: env OPENAI_BASE_URL)")
    m.add_argument("--api-key", default=None, help="OpenAI API key (default: env OPENAI_API_KEY)")
    m.add_argument(
        "--extra-body",
        default=None,
        help="OpenAI extra_body JSON string (default: env OPENAI_EXTRA_BODY)",
    )
    m.add_argument("--runs-root", default="runs", help="Directory to store artifacts")
    m.add_argument(
        "--reasoning-effort",
        default=None,
        help="OpenAI reasoning_effort for reasoning models (default: provider default)",
    )
    m.add_argument("--temperature", type=float, default=0.0)
    m.add_argument("--top-p", type=float, default=1.0)
    m.add_argument("--stream", action="store_true", help="Enable streaming model output")
    m.add_argument("--sim-max-retry", type=int, default=4)
    m.add_argument(
        "--debug-max-trials",
        type=int,
        default=30,
        help="Max Debugger (RTLEditor) edit attempts per failing simulation.",
    )
    m.add_argument("--ablation", action="store_true", help="Ablation mode: only RTL generation + syntax check")
    m.add_argument(
        "--max-completion-tokens",
        "--max-tokens",
        dest="max_completion_tokens",
        type=int,
        default=2000,
        help="Max completion tokens (OpenAI: max_completion_tokens)",
    )

    args = parser.parse_args(argv)

    if args.cmd == "run":
        cfg = load_openai_config(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            extra_body=args.extra_body,
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
            top_p=args.top_p,
            max_completion_tokens=args.max_completion_tokens,
            stream=args.stream,
        )
        if not cfg.api_key:
            raise SystemExit(
                "Missing OPENAI_API_KEY (set env or pass --api-key).",
            )

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
        print(f"run_dir: {result.output_dir_per_run}")
        print(f"rtl: {result.rtl_path}")
        print(f"testbench: {result.tb_path}")
        if result.error:
            print(f"error: {result.error}")
        return 0 if status == "PASS" else 2

    raise AssertionError("unreachable")
