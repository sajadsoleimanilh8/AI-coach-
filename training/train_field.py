"""Train the 'field' model. Same as `python -m training.train field`; see training/cli.py."""

from training.cli import main

if __name__ == "__main__":
    raise SystemExit(main(model="field"))
