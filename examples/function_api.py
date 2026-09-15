"""Use the function-style API (backwards-compatible).

Shows that the original ``run_inference_script`` function still works —
it's now a thin wrapper that instantiates RVClass and calls .run()
internally. Useful if you have existing code that uses the function API
and don't want to migrate.

Usage:
    python examples/function_api.py input.wav model.pth output.wav
    python examples/function_api.py input.wav model.pth output.wav --pitch 12
"""

import argparse
import sys
from pathlib import Path

from rvc import Config, run_inference_script


def infer(input_path, model_path, output_path, pitch=0, f0_method="rmvpe",
          embedder_model="contentvec_base", index_path=None, index_rate=0.5,
          log_level="info"):
    """Convert a single audio file using the function-style API.

    Note: this is the SAME as examples/infer.py functionally — the only
    difference is the API style. New code should prefer the class form
    (RVClass) so the model can be reused across multiple conversions.
    """
    config = Config(
        embedder_model=embedder_model,
        f0_method=f0_method,
        log_level=log_level,
    )

    # One-shot function call. Internally creates an RVClass, runs the
    # conversion, and cleans up — but the model is loaded only once.
    run_inference_script(
        config=config,
        input_path=input_path,
        output_path=output_path,
        pth_path=model_path,
        pitch=pitch,
        f0_method=f0_method,
        index_path=index_path,
        index_rate=index_rate,
    )

    print(f"Done! Output saved to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="RVC inference (function-style API, backwards-compatible)"
    )
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
