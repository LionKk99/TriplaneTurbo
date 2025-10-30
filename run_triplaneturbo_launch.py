import os
import sys
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Python wrapper for TriplaneTurbo single-prompt export (launch backend)")
    parser.add_argument("--prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--output_path", type=str, required=True, help="output file path, e.g. /path/to/out.obj")
    parser.add_argument("--output_type", type=str, choices=["obj", "ply", "glb"], default="obj", help="export format")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA device ids, e.g. '0' or '0,1'")
    parser.add_argument("--config", type=str, default="configs/TriplaneTurbo_v1_acc-2.yaml", help="config yaml for launch")
    parser.add_argument("--ckpt", type=str, default="pretrained/triplane_turbo_sd_v1.pth", help="adapter checkpoint path")
    args = parser.parse_args()

    # resolve repo dir and script path
    here = os.path.abspath(os.path.dirname(__file__))
    sh_path = os.path.join(here, "scripts", "run_single_prompt.sh")
    if not os.path.exists(sh_path):
        raise FileNotFoundError(f"Cannot find script: {sh_path}")

    cmd = [
        "bash", sh_path,
        "--gpu", args.gpu,
        "--prompt", args.prompt,
        "--fmt", args.output_type,
        "--output", args.output_path,
        "--config", args.config,
        "--ckpt", args.ckpt,
    ]
    # Inherit current env; caller ensures correct Python env
    print(f"[run_triplaneturbo_launch] Running: {' '.join(cmd)}")
    env = os.environ.copy()
    # ensure launch.py uses the SAME python interpreter as this wrapper
    env["PYTHON_BIN"] = sys.executable
    result = subprocess.run(cmd, cwd=here, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
