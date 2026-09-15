"""Basic single-file voice conversion.

Usage:
    python examples/infer.py input.wav model.pth output.wav
    python examples/infer.py input.wav model.pth output.wav --pitch 12
    python examples/infer.py input.wav model.pth output.wav --pitch -5 --f0-method crepe-large
"""

import argparse
import sys
from pathlib import Path

from rvc import Config, RVClass


def infer(input_path, model_path, output_path, pitch=0, f0_method="rmvpe",
          embedder_model="contentvec_base", index_path=None, index_rate=0.5,
          log_level="info"):
    """Convert a single audio file using RVC.

    Args:
        input_path: Path to input audio (wav, mp3, flac, ogg, m4a, ...).
        model_path: Path to .pth voice model.
        output_path: Path to write converted audio.
        pitch: Pitch shift in semitones (e.g. 12 = +1 octave).
        f0_method: F0 extraction method. Set on Config; override per-call
            by passing it to rvc.run(f0_method=...).
        embedder_model: Voice embedder model. Set on Config.
        index_path: Optional .index file for feature retrieval.
        index_rate: Feature retrieval ratio (0.0-1.0).
        log_level: 'debug' | 'info' | 'warning' | 'error' | 'critical'.
    """
    # f0_method + embedder_model set ONCE here in Config — single source of truth
    config = Config(
        embedder_model=embedder_model,
        f0_method=f0_method,
        log_level=log_level,
    )

    # RVClass inherits them — no need to repeat
    with RVClass(config=config, pth_path=model_path) as rvc:
        # Only per-conversion params here
        rvc.run(
            input_path=input_path,
            output_path=output_path,
            pitch=pitch,
            index_path=index_path,
            index_rate=index_rate,
        )

    print(f"Done! Output saved to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="RVC single-file inference")
    parser.add_argument("input", type=str, help="Input audio file")
    parser.add_argument("model", type=str, help="Path to .pth voice model")
    parser.add_argument("output", type=str, nargs="?", default="output.wav",
                        help="Output audio path (default: output.wav)")
    parser.add_argument("--pitch", "-p", type=int, default=0,
                        help="Pitch shift in semitones (default: 0)")
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

    infer(
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        pitch=args.pitch,
        f0_method=args.f0_method,
        embedder_model=args.embedder,
        index_path=args.index,
        index_rate=args.index_rate,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
