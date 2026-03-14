from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parents[2]
LEANBENCH_ROOT = Path("/Users/alokbeniwal/LeanBench")
REMOTE_REPO = "/root/tinygrad"
REMOTE_OUT = "/root/modal-out"
RUN_NAME = "linear_triton_bias2_scale_relu"

app = modal.App("tinygrad-lean-linear-triton")
out_vol = modal.Volume.from_name("tinygrad-lean-modal-runs", create_if_missing=True)

image = (
  modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
  .apt_install(
    "bash",
    "build-essential",
    "ca-certificates",
    "clang",
    "curl",
    "git",
    "pkg-config",
    "libgmp-dev",
    "xz-utils",
  )
  .run_commands(
    "curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y",
    env={"HOME": "/root"},
  )
  .env(
    {
      "HOME": "/root",
      "PATH": "/root/.elan/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    }
  )
  .add_local_dir(
    str(REPO_ROOT),
    remote_path=REMOTE_REPO,
    copy=False,
    ignore=[
      ".git",
      ".lake",
      "LeanBench",
      "__pycache__",
      "*.pyc",
      ".venv",
      ".mypy_cache",
      ".pytest_cache",
      ".ruff_cache",
      "_out",
      "wandb",
      "tinygrad.egg-info",
    ],
  )
  .add_local_dir(str(LEANBENCH_ROOT), remote_path="/root/LeanBench", copy=False)
)


def _run(cmd: str):
  import subprocess

  proc = subprocess.run(
    ["bash", "-lc", cmd],
    check=False,
    capture_output=True,
    text=True,
  )
  return {
    "cmd": cmd,
    "returncode": proc.returncode,
    "stdout": proc.stdout,
    "stderr": proc.stderr,
  }


@app.function(gpu="L4", image=image, timeout=60 * 60, cpu=4, volumes={REMOTE_OUT: out_vol})
def run_linear_triton_bias2_scale_relu() -> str:
  import json
  import os
  from datetime import datetime, timezone

  run_id = os.environ.get("MODAL_TASK_ID", "unknown")
  out_dir = Path(REMOTE_OUT) / RUN_NAME / run_id
  out_dir.mkdir(parents=True, exist_ok=True)

  steps = [
    _run("nvidia-smi"),
    _run(f"cd {REMOTE_REPO} && ~/.elan/bin/lake build linear_triton_bias2_scale_relu_smoke"),
    _run(f"cd {REMOTE_REPO} && ./.lake/build/bin/linear_triton_bias2_scale_relu_smoke"),
  ]

  overall_rc = 0
  log_lines: list[str] = []
  for step in steps:
    log_lines.append(f"$ {step['cmd']}\n")
    if step["stdout"]:
      log_lines.append(step["stdout"])
    if step["stderr"]:
      log_lines.append(step["stderr"])
    if step["returncode"] != 0 and overall_rc == 0:
      overall_rc = step["returncode"]

  (out_dir / "combined.log").write_text("\n".join(log_lines))
  (out_dir / "status.json").write_text(
    json.dumps(
      {
        "run_id": run_id,
        "status": "ok" if overall_rc == 0 else "failed",
        "returncode": overall_rc,
        "finished_at": datetime.now(timezone.utc).isoformat(),
      },
      indent=2,
    )
  )
  out_vol.commit()
  if overall_rc != 0:
    raise RuntimeError(f"remote run failed, see {out_dir}/combined.log")
  return str(out_dir)


@app.local_entrypoint()
def main():
  print(run_linear_triton_bias2_scale_relu.remote())
