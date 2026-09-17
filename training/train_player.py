"""Train the 'player' model. Same as `python -m training.train player`; see training/cli.py."""

from training.cli import main

if __name__ == "__main__":
    raise SystemExit(main(model="player"))
