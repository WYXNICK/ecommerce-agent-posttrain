#!/usr/bin/env python3
"""Detached SFT → epoch-checkpoint dev selection → HDFS archive supervisor."""

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from sft_data import NEXT, selected_data
from artifacts import persist
from shopping_agent.protocol import load_config, sha256, validate_execution
from shopping_agent.runs import write_json

HERE = Path(__file__).resolve().parent


def entry_files():
    files = [NEXT / "configs/sft.json", NEXT / "configs/data_selection.json", *HERE.glob("*.py")]
    return {str(p.relative_to(NEXT)): sha256(p) for p in sorted(files)}


def completed_training(output, work, settings):
    """A committed completion marker must be backed by resumable saved weights."""
    path = output / "train_summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    batch = settings["world_size"] * settings["micro_batch_size"] * settings["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(summary["train_examples"] / batch)
    if summary["status"] != "complete" or summary["global_step"] != steps_per_epoch * settings["epochs"]:
        raise ValueError("training completion marker has an incomplete update count")
    expected = [work / "checkpoints" / f"checkpoint-{steps_per_epoch*i}" for i in range(1, settings["epochs"]+1)]
    if summary["epoch_checkpoints"] != list(map(str, expected)):
        raise ValueError("training marker points to different checkpoint directories")
    for checkpoint in expected:
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if state["global_step"] != int(checkpoint.name.split("-")[-1]):
            raise ValueError("saved checkpoint has a different update count")
        for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt"):
            if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
                raise ValueError(f"completed checkpoint is missing {name}")
    if not (work / "checkpoints/final-adapter/adapter_model.safetensors").is_file():
        raise ValueError("final adapter has not been saved")
    return summary


def validate_post_training_update(original, current, training):
    """Repair export/supervision after completion without changing the training identity."""
    if training is None or training["status"] != "complete":
        raise ValueError("source updates require a committed complete training run")
    if {k:v for k,v in original.items() if k != "entry_files"} != {k:v for k,v in current.items() if k != "entry_files"}:
        raise ValueError("post-training repair changed data, settings, weights or runtime")
    changed = {k for k in original["entry_files"] | current["entry_files"]
               if original["entry_files"].get(k) != current["entry_files"].get(k)}
    allowed = {"scripts/sft/supervise_sft.py", "scripts/sft/export_model.py"}
    if not changed <= allowed:
        raise ValueError("post-training repair changed the frozen training implementation")
    return sorted(changed)


def selection_key(item):
    summary = item["summary"]
    if (summary["status"] != "complete" or summary["completed_tasks"] != 300
            or summary["expected_tasks"] != 300 or summary["unresolved_infrastructure_tasks"]):
        raise ValueError("checkpoint selection requires the full, infrastructure-valid 300-task dev set")
    return (summary["strict_successes"], summary["mean_final_reward"], -summary["average_steps"], -item["step"])


def report_result(result):
    training, best = result["training"], result["selected"]
    losses = {round(item["epoch"]): item["eval_loss"] for item in training["log_history"] if "eval_loss" in item}
    lines = ["# S0 冷启动 SFT 结果", "", "训练及开发集选模已完成；未运行最终测试集。HDFS 归档状态单独见本地 `archive.json`。", "",
             "| 检查点 | 验证 loss | 开发集严格成功 | 平均 Reward v3 | 平均步骤 |",
             "|---|---:|---:|---:|---:|"]
    for epoch, item in enumerate(result["candidates"], 1):
        s = item["summary"]
        lines.append(f"| step {item['step']} | {losses[epoch]:.4f} | {s['strict_successes']}/300（{s['strict_success_rate']:.2%}） | {s['mean_final_reward']:.4f} | {s['average_steps']:.3f} |")
    lines += ["", f"按预先固定规则选中 **step {best['step']}**，后续失败回流从该独立模型开始。",
              "开发集参与选模，其分数不能作为最终泛化成绩。训练用 Astra 2,000 条、独立验证 200 条；测试仍锁定。", "",
              f"实际更新 {training['global_step']} 次；各卡峰值分配显存（GiB）：{training['peak_gpu_memory_gib_by_rank']}。",
              "[完整指标、配置及逐题证据哈希](../../results/sft/astra-s0/summary.json)。运行方法见 [SFT 总结](README.md)。"]
    (NEXT / "docs/sft/results.md").write_text("\n".join(lines) + "\n")


def terminate(process):
    if process is not None:
        # A torchrun/vLLM leader can exit while a child still owns GPU memory.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="worker-local /tmp directory")
    parser.add_argument("--output", type=Path, required=True, help="small shared logs and evidence")
    parser.add_argument("--keep-on", type=Path, required=True)
    parser.add_argument("--keep-off", type=Path, required=True)
    parser.add_argument("--archive", required=True, help="HDFS destination, unique to this run")
    parser.add_argument("--environment-port", type=int, default=5705)
    parser.add_argument("--model-port", type=int, default=8015)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    output, work = args.output.resolve(), args.work_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    lock = (output / ".supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"status": "preparing", "pid": os.getpid(), "worker_hostname": os.uname().nodename,
             "started_at": datetime.now(timezone.utc).isoformat(), "work_directory": str(work),
             "model": str(args.model), "output": str(output), "archive": args.archive}
    job, environment, serving, monitor = None, None, None, None
    stopped = False

    def publish(**changes):
        state.update(changes)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "status.json", state)

    def interrupt(*_):
        nonlocal stopped
        stopped = True
        if job and job.poll() is None:
            os.killpg(job.pid, signal.SIGTERM)

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupt)

    def keep(on, *gpus):
        subprocess.run(["bash", str(args.keep_on if on else args.keep_off), *map(str, gpus)], check=True)

    def launch(command, name, extra_env=None):
        env = {**os.environ, **(extra_env or {})}
        with (output / name).open("a") as log:
            return subprocess.Popen(list(map(str, command)), env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def run(command, name):
        nonlocal job
        if stopped:
            raise InterruptedError("stop requested")
        job = launch(command, name)
        publish(child_pid=job.pid)
        completed_at = None
        while job.poll() is None:
            if stopped:
                raise InterruptedError("stop requested")
            if monitor is not None and monitor.poll() is not None:
                raise RuntimeError("GPU monitor stopped; inspect gpu-monitor.log")
            if name == "train.log":
                if completed_at is None and completed_training(output, work, settings) is not None:
                    completed_at = time.monotonic()
                    publish(status="training_finalizing")
                if completed_at is not None and time.monotonic() - completed_at > 90:
                    terminate(job)
                    write_json(output / "post_training_shutdown.json", {
                        "reason": "NCCL shutdown exceeded grace period after complete artifacts were verified",
                        "global_step": json.loads((output / "train_summary.json").read_text())["global_step"],
                        "train_summary_sha256": sha256(output / "train_summary.json")})
                    job = None
                    return
            time.sleep(5)
        if stopped:
            raise InterruptedError("stop requested")
        if job.returncode:
            raise RuntimeError(f"command exited {job.returncode}; see {name}")
        job = None

    def ready(process, url, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if stopped or process.poll() is not None:
                raise RuntimeError("service stopped before readiness; inspect service log")
            try:
                with urlopen(url, timeout=2):
                    return
            except HTTPError as exc:
                if exc.code == 405:  # ShopSimulator POST endpoint.
                    return
                raise
            except (URLError, OSError):
                time.sleep(2)
        raise TimeoutError(f"service readiness timed out: {url}")

    def archive(paths):
        run(["hdfs", "dfs", "-mkdir", "-p", args.archive], "archive.log")
        for path in paths:
            run(["hdfs", "dfs", "-put", "-f", str(path), args.archive + "/"], "archive.log")
        write_json(output / "archive.json", {"status": "complete", "destination": args.archive,
            "files": {str(p): {str(f.relative_to(p)): sha256(f) for f in p.rglob("*") if f.is_file()}
                      if p.is_dir() else sha256(p) for p in paths}})

    try:
        keep(True, 0, 1)
        publish()
        monitor = launch([sys.executable, HERE / "monitor_gpu.py", output], "gpu-monitor.log")
        _, manifest_path, manifest = selected_data()
        settings = json.loads((NEXT / "configs/sft.json").read_text())
        spec = {"execution_fingerprint": validate_execution(), "entry_files": entry_files(),
                "settings": settings,
                "student": load_config()["student"], "teacher": manifest["teacher"],
                "runtime": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate", "vllm")},
                "dataset_manifest_sha256": sha256(manifest_path),
                "base_files": {p.name: sha256(p) for p in args.model.iterdir() if p.is_file()},
                "selection": "dev300: strict successes, mean reward, fewer actions, earlier checkpoint",
                "test_evaluation": "not scheduled"}
        if {k: v for k, v in spec["base_files"].items() if k.endswith(".safetensors")} != spec["settings"]["base_model_weights"]:
            raise ValueError("base weights differ from the frozen original Baseline model")
        spec_path = output / "run.json"
        original_spec = json.loads(spec_path.read_text()) if spec_path.exists() else spec
        if original_spec != spec:
            changes = validate_post_training_update(original_spec, spec, completed_training(output, work, settings))
            write_json(output / "post_training_run.json", {
                "scope": "post-training supervision/export only; no additional optimization",
                "training_run_sha256": sha256(spec_path), "changed_files": changes, "specification": spec})
        elif not spec_path.exists():
            write_json(spec_path, spec)
        if not (work / "prepared/prepared.json").exists():
            run([sys.executable, HERE / "sft_data.py", "--model", args.model,
                 "--output", work / "prepared"], "prepare.log")
        prepared = json.loads((work / "prepared/prepared.json").read_text())
        if prepared["dataset_manifest_sha256"] != spec["dataset_manifest_sha256"]:
            raise ValueError("prepared data has a different source")
        write_json(output / "prepared.json", prepared)
        if completed_training(output, work, settings) is None:
            if (work / "checkpoints").exists() and not args.resume:
                raise ValueError("existing weights require an explicit --resume checkpoint")
            keep(False, 0, 1)
            publish(status="training")
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
                HERE / "train_sft.py", "--model", args.model, "--prepared", work / "prepared",
                "--weights", work / "checkpoints", "--output", output]
            if args.resume:
                command += ["--resume", args.resume]
            run(command, "train.log")
            keep(True, 0, 1)
        if entry_files() != spec["entry_files"] or validate_execution() != spec["execution_fingerprint"]:
            raise ValueError("SFT or execution source changed during training")
        training = completed_training(output, work, settings)
        publish(status="selecting_on_dev")
        env_url, model_url = f"http://127.0.0.1:{args.environment_port}", f"http://127.0.0.1:{args.model_port}/v1"
        environment = launch(["bash", NEXT / "scripts/start_environment.sh"], "environment.log",
            {"SHOPSIM_PORT": str(args.environment_port), "SHOP_RUNTIME_DIR": str(work / "environment")})
        ready(environment, env_url + "/api/shop_agent", 180)
        candidates = []
        for checkpoint in map(Path, training["epoch_checkpoints"]):
            step = int(checkpoint.name.split("-")[-1])
            model_dir, dev_dir = work / f"epoch-{step}", output / f"dev/step-{step}"
            publish(dev_checkpoint=step)
            if not (model_dir / "merge_manifest.json").exists():
                run([sys.executable, HERE / "export_model.py", "--base-model", args.model,
                     "--adapter", checkpoint, "--output", model_dir], f"export-{step}.log")
            persist(model_dir)
            keep(False, 0)
            serving = launch(["bash", NEXT / "scripts/serve_model.sh", model_dir], f"serve-{step}.log",
                {"CUDA_VISIBLE_DEVICES": "0", "LLM_PORT": str(args.model_port), "TENSOR_PARALLEL_SIZE": "1"})
            ready(serving, model_url + "/models", 1200)
            run([sys.executable, NEXT / "scripts/agent.py", "evaluate", "--split", "dev",
                 "--checkpoint", model_dir, "--output", dev_dir, "--environment-url", env_url,
                 "--model-url", model_url], f"dev-{step}.log")
            terminate(serving)
            serving = None
            keep(True, 0)
            summary = json.loads((dev_dir / "summary.json").read_text())
            candidate = {"step": step, "model": str(model_dir), "checkpoint": str(checkpoint), "summary": summary,
                         "evidence_sha256": {p.name: sha256(p) for p in dev_dir.iterdir() if p.suffix in {".json", ".jsonl"}}}
            selection_key(candidate)
            candidates.append(candidate)
        best = max(candidates, key=selection_key)
        result = {"status": "complete", "run": original_spec, "post_training_execution": spec, "training": training,
                  "candidates": candidates, "selected": best,
                  "note": "Development-set selection only. Final test1000 not run."}
        write_json(output / "summary.json", result)
        result_dir = NEXT / "results/sft/astra-s0"
        result_dir.mkdir(parents=True, exist_ok=True)
        write_json(result_dir / "summary.json", result)
        report_result(result)
        if os.environ.get("SHOP_ARTIFACT_QUEUE"):
            write_json(output / "archive.json", {"status": "queued", "destination": args.archive,
                       "devbox_receipts": os.environ["SHOP_ARTIFACT_QUEUE"] + "/receipts",
                       "hdfs_receipts": os.environ["SHOP_ARTIFACT_QUEUE"] + "/hdfs"})
            publish(status="complete", selected_step=best["step"], archive_status="queued")
            return 0
        publish(status="archiving", selected_step=best["step"])
        checkpoints = {Path(best["checkpoint"]), Path(training["epoch_checkpoints"][-1])}
        paths = [Path(best["model"]), *sorted(checkpoints), output / "run.json", output / "summary.json",
                 output / "train_summary.json", output / "prepared.json", output / "metrics.jsonl", output / "dev"]
        paths += [p for p in (output / "post_training_run.json", output / "shutdown-recovery.json") if p.exists()]
        archive(paths)
        publish(status="complete")
    except Exception as exc:
        publish(status="interrupted" if stopped else "failed", error=f"{type(exc).__name__}: {exc}")
        if state.get("selected_step"):
            write_json(output / "archive.json", {"status": "pending", "error": state["error"],
                       "destination": args.archive, "work_directory": str(work)})
    finally:
        for process in (job, serving, environment, monitor):
            terminate(process)
        # This runs on both success and failure, independently of the SSH session.
        try:
            keep(True, 0, 1)
            publish(keepalive="running on GPUs 0 and 1")
        except Exception as exc:
            publish(keepalive=f"restore failed: {exc}")
        lock.close()
    return 0 if state["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
