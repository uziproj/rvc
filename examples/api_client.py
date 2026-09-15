"""Use the REST API server programmatically.

Shows how to start the RVC REST API server, load a model, convert audio,
and unload the model — all from Python.

Usage:
    # Terminal 1: start the server
    rvc-api --host 0.0.0.0 --port 8000

    # Terminal 2: run this script
    python examples/api_client.py input.wav model.pth output.wav
"""

import argparse
import sys
from pathlib import Path

import requests


API_BASE = "http://localhost:8000/api/v1"


def infer(input_path, model_path, output_path, pitch=0, f0_method="rmvpe",
          embedder_model="contentvec_base", index_rate=0.5):
    """Convert audio via the REST API.

    Workflow:
        1. Load the .pth model (server returns a model_id).
        2. Upload audio + parameters -> receive converted audio bytes.
        3. Unload the model (free GPU memory).
    """
    # --- 1. Load the model ---
    print(f"[1/3] Loading model: {model_path}")
    resp = requests.post(
        f"{API_BASE}/models/load",
        json={"model_path": model_path},
        timeout=60,
    )
    resp.raise_for_status()
    model_id = resp.json()["model_id"]
    print(f"      model_id = {model_id}")

    try:
        # --- 2. Convert audio (file upload) ---
        print(f"[2/3] Converting: {input_path} -> {output_path}")
        with open(input_path, "rb") as f:
            resp = requests.post(
                f"{API_BASE}/convert",
                files={"audio": f},
                data={
                    "model_id": model_id,
                    "pitch": str(pitch),
                    "f0_method": f0_method,
                    "embedder_model": embedder_model,
                    "index_rate": str(index_rate),
                    "output_format": "wav",
                },
                stream=True,
                timeout=300,
            )
        resp.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"      Saved: {output_path}")

    finally:
        # --- 3. Unload the model (even if conversion failed) ---
        print(f"[3/3] Unloading model: {model_id}")
        requests.delete(f"{API_BASE}/models/{model_id}", timeout=30)

    print("\nDone!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="RVC REST API client example (requires server running)"
    )
    parser.add_argument("input", type=str, help="Input audio file")
    parser.add_argument("model", type=str, help="Path to .pth voice model (server-side path)")
    parser.add_argument("output", type=str, nargs="?", default="output.wav",
                        help="Output audio path (default: output.wav)")
    parser.add_argument("--pitch", "-p", type=int, default=0,
                        help="Pitch shift in semitones (default: 0)")
    parser.add_argument("--f0-method", type=str, default="rmvpe",
                        help='F0 method (default: "rmvpe")')
    parser.add_argument("--embedder", type=str, default="contentvec_base",
                        help='Embedder model (default: "contentvec_base")')
    parser.add_argument("--index-rate", type=float, default=0.5,
                        help="Index rate (default: 0.5)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Check server is up
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(
            f"Error: cannot reach RVC API at {API_BASE}.\n"
            f"Start it first with: rvc-api --host 0.0.0.0 --port 8000",
            file=sys.stderr,
        )
        sys.exit(1)

    infer(
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        pitch=args.pitch,
        f0_method=args.f0_method,
        embedder_model=args.embedder,
        index_rate=args.index_rate,
    )


if __name__ == "__main__":
    main()
