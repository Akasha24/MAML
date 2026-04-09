#!/usr/bin/env python3
"""
Run the full pipeline: generate data -> train -> test

Usage examples:
  python run_all.py                 # generate new random data, train, test
  python run_all.py --seed 42       # reproducible generation
  python run_all.py --skip-train    # only generate data and test (if model exists)
"""
import argparse
import subprocess
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="Run generate -> train -> test pipeline")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for data generation (omit for random)")
    parser.add_argument("--train-tasks", type=int, default=100, help="Number of training tasks to generate")
    parser.add_argument("--test-tasks", type=int, default=20, help="Number of test tasks to generate")
    parser.add_argument("--n-support", type=int, default=8, help="Support set size per task")
    parser.add_argument("--n-query", type=int, default=64, help="Query set size per task")
    parser.add_argument("--skip-train", action="store_true", help="Skip the training step")
    parser.add_argument("--skip-test", action="store_true", help="Skip the testing step")
    args = parser.parse_args()

    repo_dir = os.path.dirname(__file__)
    python = sys.executable

    # 1) Generate data
    gen_cmd = [python, "generate_data.py",
               "--train-tasks", str(args.train_tasks),
               "--test-tasks", str(args.test_tasks),
               "--n-support", str(args.n_support),
               "--n-query", str(args.n_query)]
    if args.seed is not None:
        gen_cmd += ["--seed", str(args.seed)]

    print("==> Generating dataset")
    subprocess.run(gen_cmd, cwd=repo_dir, check=True)

    # 2) Train
    if not args.skip_train:
        print("==> Training model")
        subprocess.run([python, "train.py"], cwd=repo_dir, check=True)
    else:
        print("==> Skipping training step")

    # 3) Test
    if not args.skip_test:
        print("==> Running tests/evaluation")
        subprocess.run([python, "test.py"], cwd=repo_dir, check=True)
    else:
        print("==> Skipping test step")

    print("\nPipeline finished.")


if __name__ == '__main__':
    main()
