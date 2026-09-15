"""Convert the same audio with multiple pitch shifts.

Demonstrates that RVClass loads the model ONCE and reuses it across
many .run() calls — no need to reload the .pth between conversions.

Usage:
    python examples/multi_pitch.py input.wav model.pth ./outputs
    python examples/multi_pitch.py input.wav model.pth ./outputs --pitches -12 0 12
"""

import argparse
import os
import sys
from pathlib import Path

from rvc import Config, RVClass


def infer_multi_pitch(input_path, model_path, output_dir, pitches=(-12, 0, 12),
                      f0_method="rmvpe", embedder_model="contentvec_base",
                      index_path=None, index_rate=0.5, log_level="info"):
    """Convert the same input multiple times with different pitch shifts.

    Args:
        input_path: Input audio file.
        model_path: Path to .pth voice model.
        output_dir: Where to write outputs (created if missing).
        pitches: Iterable of pitch shifts in semitones (e.g. (-12, 0, +12)).
        f0_method: F0 method (set on Config).
        embedder_model: Embedder model (set on Config).
        index_path: Optional .index file.
        index_rate: Feature retrieval ratio.
        log_level: Log level.
    """
    os.makedirs(output_dir, exist_ok=True)

    config = Config(
        embedder_model=embedder_model,
        f0_method=f0_method,
        log_level=log_level,
    )

    # Model loaded ONCE here, reused for every pitch
    with RVClass(config=config, pth_path=model_path) as rvc:
        for pitch in pitches:
            output_path = os.path.join(
                output_dir,
                f"output_pitch{pitch:+d}.wav".replace("+", "p").replace("-", "n"),
            )
            print(f"\n--- Converting with pitch = {pitch:+d} ---")

            rvc.run(
                input_path=input_path,
                output_path=output_path,
                pitch=pitch,
                index_path=index_path,
                index_rate=index_rate,
            )

    print(f"\nAll conversions complete. Outputs in: {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="RVC multi-pitch conversion (one model, many pitch shifts)"
    )
    parser.add_argument("input", type=str, help="Input audio file")
    parser.add_argument("model", type=str, help="Path to .pth voice model")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--pitches", type=int, nargs="+", default=[-12, 0, 12],
                        help="Pitch shifts in semitones (default: -12 0 12)")
    parser.add_argument("--f0-method", type=str, default="rmvpe",
                        help='F0 method (default: "rmvpe")')
    parser.add_argument("--embedder", type=str, default="contentvec_base",
                        help='Embedder model (default: "contentvec_base")')
    parser.add_argument("--index", type=str, default=None,
                        help="Path to .index file")
    parser.add_argument("--index-rate", type=float, default=0.5,
                        help="Index rate (default: 0.5)")
    parser.add_argument("--log-level", type=str, default="info",
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Log level (default: info)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"Error: model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    infer_multi_pitch(
        input_path=args.input,
        model_path=args.model,
        output_dir=args.output_dir,
        pitches=args.pitches,
        f0_method=args.f0_method,
        embedder_model=args.embedder,
        index_path=args.index,
        index_rate=args.index_rate,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
