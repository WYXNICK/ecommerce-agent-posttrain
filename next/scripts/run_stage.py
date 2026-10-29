#!/usr/bin/env python3
"""Own stage services, preserve logs, write concise results, and restore GPU keepalive."""

import argparse
from datetime import datetime
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

NEXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NEXT / "src"))
from shopping_agent.protocol import load_config, teacher_profiles, validate_execution
from shopping_agent.runs import write_json


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def report(path, stage, state, output):
    lines = [f"# {'Baseline 正式评测' if stage == 'baseline' else '教师轨迹采集'}", "",
             f"状态：**{state['status']}**。开始：{state['started_at']}（北京时间）；结束：{state.get('ended_at', '尚未结束')}。", "",
             "## 方法", "",
             f"- 执行指纹：`{state['execution_fingerprint']}`。配置、任务与完整命令见同目录 `status.json`、各任务 `run.json`。",
             "- Environment v2.1 / Reward v3；固定学生模板、守卫及预算。原始轨迹追加落盘，可续跑。"]
    if stage == "baseline":
        lines += ["- 冻结 revision 的原始 Qwen3.5-2B，关闭 thinking、temperature=0；先完整 300-dev，再完整 1,000-test。",
                  "- dev 与 test 使用同一权重与参数；测试轨迹不进入教师采集、训练或修复。缺失/基础设施失败不算完整正式结果。",
                  "- vLLM 使用 Triton GDN 预填充、最多 32 个并发序列；避免首次 FlashInfer 大量内核编译。模型、编译缓存均在 /tmp。",
                  "- 单 H100 执行推理，另一张保活；退出后自动恢复双卡保活。保活不计入评测算力利用率。"]
    else:
        teacher = state["teacher"]
        lines += [f"- {teacher['model']}，响应快照 `{teacher['expected_response_model']}`，{teacher['reasoning_effort']} 推理；严格 JSON 动作经共享守卫执行真实环境，未使用隐藏 gold 或私有思维链。",
                  "- 4,000 候选目标 2,000 个唯一可训练任务；独立 800 验证候选目标 200 条。按领域验收配额，每题至多两次模型尝试。",
                  "- 严格 gold 成功后检查学生模板、完整长度与监督 mask；留出校验强制执行。配额不足会明确报告，不自动换题。"]
        lines += [f"- 并发 {state['workers']}；429 每请求最多尝试 {teacher.get('rate_limit_attempts', 3)} 次，初始等待 {teacher.get('rate_limit_initial_delay_seconds', 10)} 秒、逐次翻倍、上限 {teacher.get('rate_limit_max_delay_seconds', 120)} 秒。其他临时传输错误最多 3 次，间隔 5/10 秒；鉴权失败立即停止。耗尽后保存进度停止。"]
        if state.get("seed_from"):
            lines += [f"- 从 `{state['seed_from']}` 继承同教师的已有轨迹；`source.json` 锁定原 run/raw 哈希，原数据只读保留，新 raw 仅追加新轨迹，成品与统计包含两者。"]
        if (output / "transport-update.json").exists():
            lines += ["- 已有轨迹保留原始源码指纹，本轮新增轨迹使用修正后的指纹；`transport-update.json` 校验原配置与旧 raw 前缀，数据划分、教师、动作、评分和预算保持不变。"]
    lines += ["", "## 当前结果", ""]
    for name in (("dev", "test") if stage == "baseline" else ("train", "validation")):
        summary = output / name / "summary.json"
        if not summary.exists():
            lines += [f"- **{name}**：尚未产生结果。"]
            continue
        value = json.loads(summary.read_text())
        if stage == "baseline":
            lines += [f"- **{name}**：{value['status']}；完成 {value['completed_tasks']}/{value['expected_tasks']}；"
                      f"严格成功 {value['strict_successes']}/{value['expected_tasks']}（{value['strict_success_rate']:.2%}）；"
                      f"平均奖励 {value['mean_final_reward']:.4f}；平均步骤 {value['average_steps']:.2f}。"]
        else:
            lines += [f"- **{name}**：{value['status']}；可训练独立任务 {value['accepted_unique_tasks']}/{value['target']}；"
                      f"已执行 {value['attempts']} 条轨迹；领域缺口 `{json.dumps(value['shortfall_by_domain'], ensure_ascii=False)}`。"]
        lines += [f"  - usage：`{json.dumps(value['usage']['reported_tokens'], ensure_ascii=False)}`；"
                  f"未报告 usage 的调用 {value['usage']['calls_without_usage']}。"]
    if state.get("error"):
        lines += [f"- 中断原因：{state['error']}。"]
    lines += ["", "## 证据与查看", "",
              f"- 输出目录：`{output}`。主进程日志 `run.log`，环境日志 `environment.log`；Baseline 另有 `model.log`。",
              "- 每项任务：`run.json` 配置身份、`raw.jsonl` 原始轨迹、`summary.json` 当前/最终汇总；采集另有成品和拒收原因。",
              "- 运行中指标仅是进度，不能当作最终成绩；以任务与阶段均完成后的汇总为准。",
              "- usage 仅统计 API 已报告 token，网络重试可能产生未报告费用。"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("baseline", "teacher"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--keep-on", type=Path)
    parser.add_argument("--keep-off", type=Path)
    parser.add_argument("--environment-port", type=int, default=5701)
    parser.add_argument("--teacher-profile", choices=teacher_profiles())
    parser.add_argument("--seed-from", type=Path)
    args = parser.parse_args()
    if args.stage == "baseline" and not all((args.checkpoint, args.keep_on, args.keep_off)):
        parser.error("baseline requires checkpoint and existing keepalive on/off scripts")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = {"stage": args.stage, "status": "starting", "started_at": now(),
             "execution_fingerprint": validate_execution(), "commands": [], "processes": {},
             "pid": os.getpid(), "hostname": os.uname().nodename}
    if args.stage == "teacher":
        config = load_config(teacher_profile=args.teacher_profile)
        state.update(teacher=config["teacher"], workers=config["collection"]["workers"],
                     teacher_profile=args.teacher_profile,
                     seed_from=str(args.seed_from.resolve()) if args.seed_from else None)
    services, job = [], None
    stopped, keep_stopped = False, False

    def publish():
        write_json(output / "status.json", state)
        report(args.report, args.stage, state, output)

    def interrupt(*_):
        nonlocal stopped
        stopped = True
        if job and job.poll() is None:
            job.send_signal(signal.SIGTERM)

    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, interrupt)

    def start(command, log_name, env=None, service=False):
        state["commands"].append(command)
        with (output / log_name).open("a") as log:
            process = subprocess.Popen(command, cwd=NEXT.parent, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state["processes"][log_name] = process.pid
        if service:
            services.append(process)
        publish()
        return process

    def ready(url, process, seconds=1800):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if stopped:
                raise InterruptedError("stop requested")
            if process.poll() is not None:
                raise RuntimeError(f"service exited ({process.returncode}); inspect its log")
            try:
                with urlopen(url, timeout=2) as response:
                    return response.status
            except HTTPError as exc:
                if exc.code == 405 and url.endswith("/api/shop_agent"):
                    return 200
                time.sleep(2)
            except (URLError, OSError):
                time.sleep(2)
        raise TimeoutError(f"service readiness timeout: {url}")

    try:
        publish()
        environment = os.environ.copy()
        environment.update(SHOPSIM_PORT=str(args.environment_port), SHOP_RUNTIME_DIR=str(output / "environment-runtime"))
        server = start(["bash", str(NEXT / "scripts/start_environment.sh")], "environment.log", environment, True)
        ready(f"http://127.0.0.1:{args.environment_port}/api/shop_agent", server)
        if args.stage == "baseline":
            subprocess.run(["bash", str(args.keep_off), "0"], check=True)
            keep_stopped = True
            serving = os.environ.copy()
            serving.update(CUDA_VISIBLE_DEVICES="0", LLM_PORT="8000")
            model = start(["bash", str(NEXT / "scripts/serve_model.sh"), str(args.checkpoint.resolve())], "model.log", serving, True)
            ready("http://127.0.0.1:8000/health", model)
        state["status"] = "running"
        publish()
        for name in (("dev", "test") if args.stage == "baseline" else ("validation", "train")):
            if stopped:
                raise InterruptedError("stop requested")
            state["active_task"] = name
            command = [sys.executable, str(NEXT / "scripts/agent.py")]
            if args.stage == "baseline":
                command += ["evaluate", "--split", name, "--checkpoint", str(args.checkpoint.resolve())]
            else:
                command += ["collect", "--role", name]
                if args.teacher_profile:
                    command += ["--teacher-profile", args.teacher_profile]
                if args.seed_from:
                    command += ["--seed-from", str((args.seed_from / name).resolve())]
            command += ["--environment-url", f"http://127.0.0.1:{args.environment_port}", "--output", str(output / name)]
            job = start(command, f"{name}.log")
            while job.poll() is None:
                publish()
                time.sleep(10)
            if job.returncode:
                raise RuntimeError(f"{name} exited {job.returncode}; inspect {name}/summary.json and {name}.log")
        if stopped:
            raise InterruptedError("stop requested")
        state["status"] = "complete"
    except Exception as exc:
        state.update(status="interrupted" if stopped else "failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        for process in reversed(services):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        if keep_stopped:
            result = subprocess.run(["bash", str(args.keep_on), "0", "1"], check=False)
            state["keepalive_restore_returncode"] = result.returncode
            if result.returncode:
                state.update(status="failed", error="GPU keepalive restoration failed")
        state["ended_at"] = now()
        publish()
    return 0 if state["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
