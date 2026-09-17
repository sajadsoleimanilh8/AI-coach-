"""Train the 'calibration' model. Same as `python -m training.train calibration`; see training/cli.py."""

from training.cli import main

if __name__ == "__main__":
    raise SystemExit(main(model="calibration"))
