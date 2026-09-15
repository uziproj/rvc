"""Configure logging at different verbosity levels.

Shows how to control RVC's log output via:
  - Config(log_level=...)
  - Config.set_log_level(...) at runtime
  - set_log_level() standalone function
  - RVClass(log_level=...) per-instance override

Usage:
    python examples/logging_levels.py input.wav model.pth
"""

import sys
from pathlib import Path

from rvc import Config, RVClass, set_log_level, get_log_level, get_logger, LOG_LEVELS


def demo_logging_levels(input_path, model_path):
    """Show every way to control log verbosity in RVC."""

    # A logger for THIS example script (under the 'rvc' namespace)
    my_logger = get_logger("example.logging")
    my_logger.info(f"Available log levels: {list(LOG_LEVELS.keys())}")

    # ------------------------------------------------------------------
    # 1. Default level (info)
    # ------------------------------------------------------------------
    print("\n=== 1. Default level (info) ===")
    my_logger.info(f"Current level: {get_log_level()}")
    config = Config()  # uses default log_level='info'
    my_logger.info(f"After Config(): {get_log_level()}")

    # ------------------------------------------------------------------
    # 2. Set via Config constructor
    # ------------------------------------------------------------------
    print("\n=== 2. Set via Config(log_level='warning') ===")
    config = Config(log_level="warning")
    my_logger.warning(f"Current level: {get_log_level()} (only warnings+ will show)")
    my_logger.info("This INFO won't appear because level is 'warning'")

    # ------------------------------------------------------------------
    # 3. Change at runtime via Config.set_log_level()
    # ------------------------------------------------------------------
    print("\n=== 3. Runtime change: Config.set_log_level('debug') ===")
    config.set_log_level("debug")
    my_logger.info(f"Current level: {get_log_level()}")
    my_logger.debug("Now DEBUG messages appear — including GPU cache clears")

    # ------------------------------------------------------------------
    # 4. Standalone set_log_level()
    # ------------------------------------------------------------------
    print("\n=== 4. Standalone set_log_level('error') ===")
    set_log_level("error")
    my_logger.error(f"Current level: {get_log_level()} (only errors+ will show)")
    my_logger.warning("This WARNING won't appear")
    my_logger.info("This INFO won't appear either")

    # ------------------------------------------------------------------
    # 5. RVClass per-instance override
    # ------------------------------------------------------------------
    print("\n=== 5. RVClass(log_level='info') override ===")
    set_log_level("error")  # global is 'error'

    config = Config(log_level="error")  # Config sets global to 'error'
    my_logger.error(f"After Config(log_level='error'): {get_log_level()}")

    # But this RVClass instance overrides to 'info'
    with RVClass(config=config, pth_path=model_path, log_level="info") as rvc:
        my_logger.info("Now INFO appears again — RVClass overrode to 'info'")
        my_logger.debug("DEBUG still hidden (RVClass set info, not debug)")

        # ------------------------------------------------------------------
        # 6. Run a conversion — log output reflects the level set above
        # ------------------------------------------------------------------
        print("\n=== 6. Run a conversion ===")
        from pathlib import Path
        output_path = "./output_logging_demo.wav"
        if Path(output_path).exists():
            Path(output_path).unlink()

        rvc.run(
            input_path=input_path,
            output_path=output_path,
            pitch=0,
        )

    print(f"\nDone! Output: {output_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python examples/logging_levels.py <input.wav> <model.pth>")
        sys.exit(1)
    input_path = sys.argv[1]
    model_path = sys.argv[2]

    if not Path(input_path).exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not Path(model_path).exists():
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    demo_logging_levels(input_path, model_path)


if __name__ == "__main__":
    main()
