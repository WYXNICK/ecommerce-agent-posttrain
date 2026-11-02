#!/usr/bin/env python3
"""Record actual GPU/SM samples and their trailing one-minute averages."""

from collections import defaultdict, deque
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from sft_data import NEXT
from shopping_agent.runs import write_json


def main():
    output = Path(sys.argv[1])
    history = defaultdict(lambda: deque(maxlen=12))
    with subprocess.Popen(["nvidia-smi", "dmon", "-s", "u", "-d", "5"], stdout=subprocess.PIPE,
                          text=True) as process, (output / "gpu.jsonl").open("a") as log:
        try:
            for line in process.stdout:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                fields = line.split()
                gpu, sm = int(fields[0]), int(fields[1])
                utilization = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
                    "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True)
                util, memory = map(int, utilization.strip().split(","))
                history[gpu].append((util, sm))
                item = {"time": datetime.now(timezone.utc).isoformat(), "gpu": gpu,
                        "gpu_percent": util, "sm_percent": sm, "memory_mib": memory,
                        "samples": len(history[gpu]),
                        "mean_gpu_percent_60s": sum(x[0] for x in history[gpu])/len(history[gpu]),
                        "mean_sm_percent_60s": sum(x[1] for x in history[gpu])/len(history[gpu])}
                item["below_target_60s"] = item["samples"] == 12 and (
                    item["mean_gpu_percent_60s"] <= 30 or item["mean_sm_percent_60s"] <= 20)
                log.write(json.dumps(item) + "\n")
                log.flush()
                write_json(output / f"gpu-{gpu}.json", item)
                if item["below_target_60s"]:
                    print("GPU_LOW_UTILIZATION " + json.dumps(item), flush=True)
        finally:
            process.terminate()
            process.wait()


if __name__ == "__main__":
    main()
