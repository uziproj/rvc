"""Demonstrate per-call f0_method override.

Shows that f0_method is set ONCE on Config, but can be overridden per-call
by passing it explicitly to rvc.run(f0_method=...). Useful for A/B testing
different pitch extractors on the same audio.

Usage:
    python examples/compare_f0.py input.wav model.pth ./outputs
"""

import argparse
import os
import sys
from pathlib import Path

from rvc import Config, RVClass, F0_METHODS


def compare_f0_methods(input_path, model_path, output_dir,
                       methods=("rmvpe", "fcpe", "crepe-large", "harvest"),
                       embedder_model="contentvec_base", pitch=0,
                       log_level="warning"):
    """Convert the same input with different F0 extraction methods.

    Args:
        input_path: Input audio file.
        model_path: Path to .pth voice model.
        output_dir: Where to write outputs (created if missing).
        methods: Iterable of F0 method names to try.
        embedder_model: Embedder model (set on Config).
        pitch: Pitch shift in semitones.
        log_level: Log level — 'warning' here to keep output clean since
            we print our own progress messages.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Default F0 method is rmvpe — this is what gets used when rvc.run()
    # does NOT pass f0_method explicitly.
    config = Config(
        embedder_model=embedder_model,
        f0_method="rmvpe",
        log_level=log_level,
    )

    print(f"Available F0 methods: {F0_METHODS}")
    print(f"Comparing: {list(methods)}")
    print()

    with RVClass(config=config, pth_path=model_path) as rvc:
        for method in methods:
            if method not in F0_METHODS:
                print(f"  [SKIP] Unknown F0 method: {method}")
                continue

            # Replace characters that aren't filesystem-safe
            safe_name = method.replace("+", "plus").replace("[", "").replace("]", "").replace(":", "_")
            output_path = os.path.join(output_dir, f"output_f0_{safe_name}.wav")
            print(f"--- F0 method: {method} ---")

            # Override f0_method for THIS call only — Config's value
            # stays as 'rmvpe' for subsequent calls that don't pass it.
            rvc.run(
                input_path=input_path,
                output_path=output_path,
                pitch=pitch,
                f0_method=method,
            )

    print(f"\nAll comparisons complete. Outputs in: {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="RVC F0 method comparison (one model, many extractors)"
    )
    parser.add_argument("input", type=str, help="Input audio file")
    parser.add_argument("model", type=str, help="Path to .pth voice model")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["rmvpe", "fcpe", "crepe-large", "harvest"],
                        help="F0 methods to compare (default: rmvpe fcpe crepe-large harvest)")
    parser.add_argument("--embedder", type=str, default="contentvec_base",
                        help='Embedder model (default: "contentvec_base")')
    parser.add_argument("--pitch", "-p", type=int, default=0,
                        help="Pitch shift in semitones (default: 0)")
    parser.add_argument("--log-level", type=str, default="warning",
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Log level (default: warning — keeps output clean)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"Error: model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    compare_f0_methods(
        input_path=args.input,
        model_path=args.model,
        output_dir=args.output_dir,
        methods=args.methods,
        embedder_model=args.embedder,
        pitch=args.pitch,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
