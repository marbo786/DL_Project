"""Run the existing single-image receipt protocol from explicitly supplied inputs."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DATASET_CARD = "https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v2#paired-authentic-images"
CHECKPOINT = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"


def stages(archive, package, weights, output, threads):
    imported = output / "aiforge"
    assets = imported / "assets"
    dedup = imported / "manifest-deduplicated.csv"
    protocol = output / "protocol"
    contained = output / "contained"
    manifest = contained / "manifest.csv"
    study = output / "study"
    retrospective = output / "retrospective"
    jobs = [
        ("research.aiforge", archive, imported),
        ("research.manifest", imported / "manifest.csv", assets, "--deduplicated-output", dedup),
        ("research.receipt.protocol", dedup, assets, protocol, "--package", package),
        ("research.receipt.candidate_review", protocol, assets, contained),
        ("research.receipt.study", "--manifest", manifest, "--root", assets, "--weights", weights,
         "--output", study, "--threads", threads, "--curve"),
        ("research.receipt.retrospective", "--manifest", manifest, "--root", assets, "--study", study,
         "--output", retrospective, "--threads", threads),
        ("research.receipt.forms", "--package", package, "--manifest", manifest, "--receipt-root", assets,
         "--study", study, "--lock", retrospective / "freeze.json", "--output", output / "xfund",
         "--threads", threads),
    ]
    return [[sys.executable, "-m", *map(str, job)] for job in jobs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="official paired AIForge-Doc v2 ZIP")
    parser.add_argument("--package", type=Path, help="extracted full AIForge-Doc v2 package")
    parser.add_argument("--weights", type=Path, help="official MobileNetV3-Small ImageNet V1 checkpoint")
    parser.add_argument("--output", type=Path, help="fresh output directory, for example runs/receipt-study")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--plan", action="store_true", help="print commands without writing files or starting work")
    args = parser.parse_args()
    if not any((args.archive, args.package, args.weights, args.output)):
        print(f"Obtain the full paired archive from the publisher: {DATASET_CARD}")
        print(f"Obtain the official pretrained checkpoint: {CHECKPOINT}")
        print("Extract the package privately, then pass --archive, --package, --weights, and --output. Use --help for options.")
        return
    if not all((args.archive, args.package, args.weights, args.output)) or args.threads < 1:
        parser.error("supply all four paths and a positive thread count")
    archive, package, weights, output = (value.resolve() for value in (args.archive, args.package, args.weights, args.output))
    commands = stages(archive, package, weights, output, args.threads)
    if args.plan:
        for command in commands:
            print(subprocess.list2cmdline(command))
        return
    if not archive.is_file() or not weights.is_file():
        parser.error("the paired archive and pretrained checkpoint must exist")
    for file in ("metadata.jsonl", "source_mapping.csv", "dataset_info.json"):
        if not (package / file).is_file():
            parser.error(f"the extracted package is missing {file}")
    if output.is_relative_to(package) or any(output.is_relative_to(ROOT / name) for name in ("data", "artifacts", "results")):
        parser.error("choose an output directory outside the dataset and saved evidence")
    output.mkdir(parents=True, exist_ok=False)
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
