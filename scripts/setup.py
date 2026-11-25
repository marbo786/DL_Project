"""Install the constrained development environment into the active interpreter."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    if sys.platform != "darwin":
        pip("install", "--index-url", "https://download.pytorch.org/whl/cpu",
            "torch==2.12.0+cpu", "torchvision==0.27.0+cpu")
    pip("install", "-c", "constraints.txt", "-e", ".[research,dev]")
    pip("check")
