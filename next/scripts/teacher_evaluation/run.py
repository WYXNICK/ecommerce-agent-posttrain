#!/usr/bin/env python3
"""Detach-safe teacher evaluation supervisor with a collection completion gate."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from evaluate import NEXT, entry_files, teacher_config
from shopping_agent.protocol import sha256, validate_execution
from shopping_agent.runs import write_json
from shopping_agent.tasks import TaskPool


def collection_complete(directory, config, fingerprint):
    """Do not schedule evaluation from partial, failed or different-teacher data."""
    path = directory / "status.json"
    if not path.exists():
        return False
    state = json.loads(path.read_text())
    if state["teacher"] != config["teacher"] or state["execution_fingerprint"] != fingerprint:
        raise ValueError("collection dependency has a different teacher or execution contract")
    if state["status"] != "complete":
        return False
    for role, key in (("train", "training_target"), ("validation", "validation_target")):
        summary = json.loads((directory / role / "summary.json").read_text())
        if (summary["status"] != "complete" or summary["target"] != config["collection"][key]
                or summary["accepted_unique_tasks"] != summary["target"]):
            raise ValueError("collection reports complete without its full accepted datasets")
    return True


def save_progress(output, state):
    write_json(output / "status.json", state)
    split, count = state["split"], state["expected_tasks"]
    summary_path = output / split / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    lines = [f"# {state['teacher_name']} 教师 {split} 评测", "",
             f"状态：**{state['status']}**。{split} 固定 {count} 题，模型行为失败保留在分母。", "",
             f"完成：{summary.get('completed_tasks', 0)}/{count}；严格成功：{summary.get('strict_successes', 0)}/{count}。",
             "运行中指标仅为进度，不是完整成绩。", "",
             f"`run.log`：监督进程；`{split}.log`：逐题进度；`{split}/raw.jsonl`：原始证据；",
             f"`{split}/run.json`：冻结配置；`{split}/summary.json`：指标、缺失任务与 usage。"]
    if state.get("error"):
        lines += ["", "中断原因：" + state["error"]]
    (output / "progress.md").write_text("\n".join(lines) + "\n")


def export_result(output, teacher, split="test"):
    evidence = output / split
    tasks = TaskPool().tasks(split)
    count = len(tasks)
    summary = json.loads((evidence / "summary.json").read_text())
    if (summary["status"] != "complete" or summary["completed_tasks"] != count
            or summary["expected_tasks"] != count or summary["unresolved_infrastructure_tasks"]):
        raise ValueError("cannot export an incomplete teacher benchmark")
    spec = json.loads((evidence / "run.json").read_text())
    if spec["split"] != split or spec["candidate_task_ids"] != [task["task_id"] for task in tasks]:
        raise ValueError("evaluation evidence does not match the complete frozen split")
    directory = NEXT / "results/teacher-evaluation" / teacher
    if split == "dev":
        directory /= "dev"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "summary.json", {
        "specification": spec, "summary": summary,
        "evidence_sha256": {name: sha256(evidence / name)
                            for name in ("run.json", "raw.jsonl", "summary.json")},
    })
    document = NEXT / "docs/teacher-evaluation/results" / (f"{teacher}.md" if split == "test" else f"{teacher}-dev.md")
    result_relative = f"{teacher}/summary.json" if split == "test" else f"{teacher}/dev/summary.json"
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(
        f"# {spec['teacher']['expected_response_model']} {split} 评测结果\n\n"
        f"状态：**complete**。冻结 {split} {count} 题，单次有效模型行为，不按成功筛选或重采。\n\n"
        f"- 严格成功：{summary['strict_successes']}/{count}（{summary['strict_success_rate']:.2%}）。\n"
        f"- 平均 Reward v3：{summary['mean_final_reward']:.4f}；平均步骤：{summary['average_steps']:.3f}。\n"
        "- 缺失任务及未解决基础设施错误：0；模型失败均保留在完整分母。\n"
        f"- [完整指标、配置及证据哈希](../../../results/teacher-evaluation/{result_relative})。\n\n"
        "教师采用采集时同一 high 推理与网关默认采样；学生使用确定性本地推理。"
        "两者共享可见执行协议，但推理计算预算不同，本表作为教师参考，不作为算法消融。\n",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", choices=("sol", "astra"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--after-collection", type=Path, required=True)
    parser.add_argument("--environment-port", type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config, fingerprint = teacher_config(args.teacher), validate_execution()
    state = {"status": "waiting_collection", "teacher_name": args.teacher, "split": args.split,
             "expected_tasks": len(TaskPool().tasks(args.split)),
             "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
             "teacher": config["teacher"], "pid": os.getpid(), "entry_files": entry_files(),
             "execution_fingerprint": fingerprint, "collection_dependency": str(args.after_collection.resolve())}
    stopped, job, server = False, None, None

    def interrupt(*_):
        nonlocal stopped
        stopped = True
        if job and job.poll() is None:
            job.send_signal(signal.SIGTERM)

    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, interrupt)
    try:
        save_progress(output, state)
        while not collection_complete(args.after_collection, config, fingerprint):
            if stopped:
                raise InterruptedError("stop requested")
            time.sleep(10)
        if stopped:
            raise InterruptedError("stop requested")
        if entry_files() != state["entry_files"] or validate_execution() != fingerprint:
            raise ValueError("frozen evaluator changed while waiting")
        state["status"] = "starting"
        save_progress(output, state)
        env = os.environ.copy()
        env.update(SHOPSIM_PORT=str(args.environment_port), SHOP_RUNTIME_DIR=str(output / "environment-runtime"))
        with (output / "environment.log").open("a") as log:
            server = subprocess.Popen(["bash", str(NEXT / "scripts/start_environment.sh")], env=env,
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        url = f"http://127.0.0.1:{args.environment_port}"
        deadline = time.monotonic() + 180
        while True:
            if stopped or server.poll() is not None:
                raise RuntimeError("environment stopped before becoming ready")
            try:
                with urlopen(url + "/api/shop_agent", timeout=2):
                    break
            except HTTPError as exc:
                if exc.code == 405:
                    break
                raise
            except (URLError, OSError):
                if time.monotonic() > deadline:
                    raise TimeoutError("environment readiness timed out")
                time.sleep(2)
        command = [sys.executable, str(Path(__file__).with_name("evaluate.py")),
                   "--teacher", args.teacher, "--split", args.split, "--environment-url", url, "--output", str(output / args.split)]
        with (output / f"{args.split}.log").open("a") as log:
            job = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        state.update(status="running", evaluator_pid=job.pid, environment_pid=server.pid)
        while job.poll() is None:
            save_progress(output, state)
            time.sleep(10)
        if stopped:
            raise InterruptedError("stop requested")
        if job.returncode:
            raise RuntimeError(f"evaluator exited {job.returncode}; inspect {args.split}.log and {args.split}/summary.json")
        export_result(output, args.teacher, args.split)
        state["status"] = "complete"
    except Exception as exc:
        state.update(status="interrupted" if stopped else "failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if job and job.poll() is None:
            job.send_signal(signal.SIGTERM)
            job.wait()
        if server and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        save_progress(output, state)
        lock.close()
    return 0 if state["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
