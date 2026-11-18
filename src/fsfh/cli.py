"""Command line interface: ``python -m fsfh {fly,campaign,fit,triage,results,plot} ...``."""

from __future__ import annotations

import argparse
from pathlib import Path

from fsfh.autopilot import AutopilotGains
from fsfh.campaign import run_campaign
from fsfh.detectors import fit
from fsfh.plot import plot_run
from fsfh.results import make_results
from fsfh.runlog import read_run, write_run
from fsfh.scenarios import SCENARIOS, get_scenario
from fsfh.sensors import DEFAULT_SENSORS, derive_sensors
from fsfh.sim import fly


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fsfh", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fly = sub.add_parser("fly", help="fly a scripted scenario and log truth + sensors")
    p_fly.add_argument("--scenario", default="basic", choices=sorted(SCENARIOS))
    p_fly.add_argument("--seed", type=int, default=1, help="seed for sensor noise")
    p_fly.add_argument("--out", type=Path, default=Path("runs"), help="output root")
    p_fly.add_argument(
        "--duration", type=float, default=None, help="truncate the scenario to this many seconds"
    )

    p_camp = sub.add_parser("campaign", help="fly a mix of nominal and faulted runs")
    p_camp.add_argument("--n", type=int, default=60)
    p_camp.add_argument("--seed", type=int, default=7, help="campaign seed (drives everything)")
    p_camp.add_argument("--scenario", default="basic", choices=sorted(SCENARIOS))
    p_camp.add_argument("--out", type=Path, default=Path("runs"))
    p_camp.add_argument("--duration", type=float, default=None)

    p_plot = sub.add_parser("plot", help="plot truth vs sensors for one run")
    p_plot.add_argument("run_dir", type=Path)
    p_plot.add_argument("--out", type=Path, default=None, help="PNG path (default: in run dir)")

    p_fit = sub.add_parser("fit", help="fit detectors on the nominal runs of a campaign")
    p_fit.add_argument("--manifest", type=Path, default=Path("runs/campaign7-manifest.csv"))
    p_fit.add_argument("--models", type=Path, default=Path("models"))
    p_fit.add_argument("--target-fa-per-hour", type=float, default=1.0)

    p_tri = sub.add_parser("triage", help="run the LLM triage agent over a campaign's alarms")
    p_tri.add_argument("--manifest", type=Path, default=Path("runs/campaign7-manifest.csv"))
    p_tri.add_argument("--models", type=Path, default=Path("models"))
    p_tri.add_argument("--backend", choices=["anthropic", "ollama"], default="anthropic")
    p_tri.add_argument("--model", default=None, help="model id for the backend")
    p_tri.add_argument("--cache", type=Path, default=Path("triage_cache"))
    p_tri.add_argument("--offline", action="store_true", help="cache only, never call the network")
    p_tri.add_argument("--out", type=Path, default=Path("results"))

    p_res = sub.add_parser("results", help="regenerate results/*.md and the README tables")
    p_res.add_argument("--runs", type=Path, default=Path("runs"))
    p_res.add_argument("--readme", type=Path, default=Path("README.md"))
    p_res.add_argument("--no-readme", action="store_true", help="do not touch the README")
    p_res.add_argument("--out", type=Path, default=Path("results"))
    p_res.add_argument("--manifest", type=Path, default=Path("runs/campaign7-manifest.csv"))
    p_res.add_argument("--models", type=Path, default=Path("models"))
    p_res.add_argument("--triage-cache", type=Path, default=Path("triage_cache"))
    p_res.add_argument("--triage-backend", default="anthropic")
    p_res.add_argument("--triage-model", default=None)
    return parser


def cmd_fly(args: argparse.Namespace) -> int:
    scenario = get_scenario(args.scenario)
    if args.duration is not None:
        scenario = scenario.truncated(args.duration)
    gains = AutopilotGains()
    result = fly(scenario, gains)
    sensors = derive_sensors(result.truth, args.seed, DEFAULT_SENSORS)
    run_dir = write_run(args.out, scenario, args.seed, result, sensors, gains, DEFAULT_SENSORS)
    last = result.truth.iloc[-1]
    print(f"wrote {run_dir} ({len(result.truth)} samples)")
    print(
        f"final: t={last.t_s:.1f}s alt={last.altitude_ft:.0f}ft "
        f"hdg={last.heading_deg:.1f}deg ias={last.ias_kts:.1f}kts"
    )
    return 0


def cmd_campaign(args: argparse.Namespace) -> int:
    scenario = get_scenario(args.scenario)
    if args.duration is not None:
        scenario = scenario.truncated(args.duration)
    manifest = run_campaign(scenario, args.n, args.seed, args.out)
    print(f"wrote {manifest}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    out = args.out or args.run_dir / "truth_vs_sensors.png"
    print(f"wrote {plot_run(args.run_dir, out)}")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    import csv

    rows = list(csv.DictReader(args.manifest.open()))
    nominal = {
        r["run_id"]: read_run(Path(r["path"])).sensors for r in rows if r["fault_type"] == "nominal"
    }
    model = fit(nominal, args.models, args.target_fa_per_hour)
    print(f"fitted on {len(model.train_runs)} train / {len(model.val_runs)} val nominal runs")
    print("thresholds:", {k: round(v, 3) for k, v in model.thresholds.items()})
    return 0


def cmd_triage(args: argparse.Namespace) -> int:
    from fsfh.triage.backends import make_backend
    from fsfh.triage.evaluate import triage_campaign, write_report

    backend = make_backend(args.backend, args.model, args.cache, args.offline)
    df = triage_campaign(args.manifest, args.models, backend)
    print(write_report(df, args.out, backend), end="")
    return 0


def cmd_results(args: argparse.Namespace) -> int:
    models = args.models if (args.models / "detectors.json").is_file() else None
    table = make_results(
        args.runs, None if args.no_readme else args.readme, args.out, args.manifest, models,
        triage=(args.triage_cache, args.triage_backend, args.triage_model),
    )  # fmt: skip
    print(table, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fly":
        return cmd_fly(args)
    if args.command == "campaign":
        return cmd_campaign(args)
    if args.command == "plot":
        return cmd_plot(args)
    if args.command == "fit":
        return cmd_fit(args)
    if args.command == "triage":
        return cmd_triage(args)
    if args.command == "results":
        return cmd_results(args)
    raise AssertionError(f"unhandled command {args.command}")
