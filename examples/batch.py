"""Batch voice conversion.

Convert every audio file in a directory using a single loaded model.
The model is loaded ONCE (in RVClass.__init__) and reused for every file.

Usage:
    python examples/batch.py ./audio_in model.pth
    python examples/batch.py ./audio_in model.pth --pitch 12 --f0-method rmvpe
    python examples/batch.py ./audio_in model.pth --output-format flac
"""

import argparse
import sys
from pathlib import Path

from rvc import Config, RVClass


def infer_batch(input_dir, model_path, pitch=0, f0_method="rmvpe",
                embedder_model="contentvec_base", index_path=None,
                index_rate=0.5, output_format="wav", clean_audio=False,
                clean_strength=0.7, log_level="info"):
    """Batch-convert every audio file in ``input_dir``.

    Args:
        input_dir: Directory of audio files (wav, mp3, flac, ogg, m4a, ...).
        model_path: Path to .pth voice model.
        pitch: Pitch shift in semitones.
        f0_method: F0 extraction method (set on Config, single source of truth).
        embedder_model: Voice embedder model (set on Config).
        index_path: Optional .index file.
        index_rate: Feature retrieval ratio.
        output_format: Output format (wav, flac, mp3, ogg).
        clean_audio: Apply noise reduction.
        clean_strength: Noise reduction strength (0.0-1.0).
        log_level: Log level.

    Outputs are written next to each input as ``<name>_output.<ext>``.
    """
    config = Config(
        embedder_model=embedder_model,
        f0_method=f0_method,
        log_level=log_level,
    )

    with RVClass(config=config, pth_path=model_path) as rvc:
        # rvc.run() auto-detects directory input and switches to batch mode.
        # Output files are written next to each input as <name>_output.<ext>.
        rvc.run(
            input_path=input_dir,
            pitch=pitch,
            index_path=index_path,
            index_rate=index_rate,
            export_format=output_format,
            clean_audio=clean_audio,
            clean_strength=clean_strength,
        )

    print(f"Batch conversion complete. Outputs in: {input_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="RVC batch inference")
    parser.add_argument("input_dir", type=str, help="Directory of audio files")
    parser.add_argument("model", type=str, help="Path to .pth voice model")
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
    parser.add_argument("--output-format", type=str, default="wav",
                        choices=["wav", "flac", "mp3", "ogg"],
                        help="Output format (default: wav)")
    parser.add_argument("--clean-audio", action="store_true",
                        help="Apply noise reduction")
    parser.add_argument("--clean-strength", type=float, default=0.7,
                        help="Noise reduction strength (default: 0.7)")
    parser.add_argument("--log-level", type=str, default="info",
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Log level (default: info)")
    args = parser.parse_args()

    if not Path(args.input_dir).is_dir():
        print(f"Error: input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"Error: model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    infer_batch(
        input_dir=args.input_dir,
        model_path=args.model,
        pitch=args.pitch,
        f0_method=args.f0_method,
        embedder_model=args.embedder,
        index_path=args.index,
        index_rate=args.index_rate,
        output_format=args.output_format,
        clean_audio=args.clean_audio,
        clean_strength=args.clean_strength,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
