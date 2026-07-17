"""Command-line entry point for the automated MOHAWK pipeline."""

from __future__ import annotations

import argparse
import sys

from .config import GraphConfig, MohawkConfig, load_config
from .pipeline import run_pipeline, save_results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="mohawk",
        description="Automated MOHAWK hdEEG brain-network pipeline (Python/MNE).",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="EEG file (RAW/MFF/EDF/VHDR/SET/FIF). Omit with --demo.",
    )
    parser.add_argument("-n", "--basename", default=None,
                        help="Base name for outputs (default: input stem).")
    parser.add_argument("-o", "--outdir", default=".", help="Output directory.")
    parser.add_argument("--montage", default=None,
                        help="Override montage (e.g. standard_1005).")
    parser.add_argument("--trials", type=int, default=60,
                        help="Number of epochs to retain (default 60; 0 = all).")
    parser.add_argument("--no-graph", action="store_true",
                        help="Skip (slow) graph-theory metrics.")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation.")
    parser.add_argument("--heuristic", type=int, default=None,
                        help="Louvain repetitions for graph metrics (default 50).")
    parser.add_argument("--config", default=None,
                        help="YAML config file (e.g. config/paper.yml for paper-exact settings).")
    parser.add_argument("--faithful", action="store_true",
                        help="Faithful mode: no automatic artefact rejection; "
                             "use the reviewed --bad-channels / --ica-exclude instead.")
    parser.add_argument("--bad-channels", default=None,
                        help="Comma-separated channels to interpolate (manual).")
    parser.add_argument("--ica-exclude", default=None,
                        help="Comma-separated ICA component indices to remove (manual).")
    parser.add_argument("--demo", action="store_true",
                        help="Run on built-in synthetic data instead of a file.")
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else MohawkConfig()
    if args.heuristic is not None:
        config.graph = GraphConfig(heuristic=args.heuristic)

    bad_channels = args.bad_channels.split(",") if args.bad_channels else ()
    ica_exclude = (
        [int(i) for i in args.ica_exclude.split(",")] if args.ica_exclude else None
    )

    if args.demo:
        from .datasets import synthetic_raw
        source = synthetic_raw()
        basename = args.basename or "demo"
    elif args.input:
        source = args.input
        import os
        basename = args.basename or os.path.splitext(os.path.basename(args.input))[0]
    else:
        parser.error("provide an input file or --demo")
        return 2

    result = run_pipeline(
        source,
        basename=basename,
        outdir=args.outdir,
        montage=args.montage,
        config=config,
        set_trials=None if args.trials == 0 else args.trials,
        compute_graph=not args.no_graph,
        make_figures=not args.no_figures,
        auto_reject=not args.faithful,
        bad_channels=bad_channels,
        ica_exclude=ica_exclude,
    )
    path = save_results(result, args.outdir)
    print(f"MOHAWK: done. ICA excluded {len(result.ica_excluded)} components.")
    print(f"Results: {path}")
    for name, fig in result.figures.items():
        print(f"  figure[{name}]: {fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
