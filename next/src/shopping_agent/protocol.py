"""One execution contract for collection, evaluation and future training adapters."""

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

from shopping_agent.environment.projection import project_observation
from shopping_agent.environment.tools import SHOP_TOOL_SCHEMAS
from shopping_agent.training.sft.dataset import normalize_messages_for_chat_template

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = ROOT.parent
CONFIG_PATH = ROOT / "configs/execution.json"
MANIFEST_PATH = ROOT / "configs/execution_manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def teacher_profiles():
    return sorted(path.stem for path in (ROOT / "configs/teachers").glob("*.json"))


def load_config(path=CONFIG_PATH, *, teacher_profile=None):
    config = json.loads(Path(path).read_text())
    if teacher_profile is not None:
        if teacher_profile not in teacher_profiles():
            raise ValueError("unknown frozen teacher profile")
        config["teacher"].update(json.loads((ROOT / "configs/teachers" / f"{teacher_profile}.json").read_text()))
    return config


def execution_files():
    files = [CONFIG_PATH, ROOT / "data/task_pool/metadata.json", ROOT / "configs/task_pool.json"]
    files += list((ROOT / "configs/teachers").glob("*.json"))
    files += list((ROOT / "src").rglob("*.py"))
    files += list((ROOT / "scripts").glob("*.py")) + list((ROOT / "scripts").glob("*.sh"))
    shop = REFERENCE_ROOT / "environments/ShopSimulator/shop_env"
    files += list(shop.rglob("*.py")) + list((shop / "configs").glob("*.json"))
    files += [REFERENCE_ROOT / "data/environment.json", REFERENCE_ROOT / "data/evaluation/tasks.jsonl"]
    return sorted(set(files))


def runtime_identity():
    packages = ("transformers", "tokenizers", "huggingface_hub", "jinja2")
    identity = {name: metadata.version(name) for name in packages}
    identity["python"] = f"{sys.version_info.major}.{sys.version_info.minor}"
    source = metadata.distribution("transformers").read_text("direct_url.json")
    identity["transformers_source"] = json.loads(source) if source else None
    return identity


def freeze_execution():
    config = load_config()
    manifest = {
        "version": config["version"],
        "runtime": runtime_identity(),
        "files": {str(p.relative_to(REFERENCE_ROOT)): sha256(p) for p in execution_files()},
        "tools_sha256": hashlib.sha256(json.dumps(SHOP_TOOL_SCHEMAS, sort_keys=True).encode()).hexdigest(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def validate_execution():
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest["runtime"] != runtime_identity():
        raise ValueError("execution dependencies changed; restore the frozen runtime or review and freeze again")
    for name, expected in manifest["files"].items():
        if sha256(REFERENCE_ROOT / name) != expected:
            raise ValueError(f"execution contract changed: {name}; review and freeze again")
    current = {str(p.relative_to(REFERENCE_ROOT)) for p in execution_files()}
    if current != set(manifest["files"]):
        raise ValueError("execution source inventory changed; review and freeze again")
    return sha256(MANIFEST_PATH)


class BudgetExceeded(RuntimeError):
    def __init__(self, status):
        super().__init__(status)
        self.status = status


class StudentTokenizer:
    """All visible budgets and SFT checks use the frozen student's tokenizer."""

    def __init__(self, tokenizer, enable_thinking=False):
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking

    @classmethod
    def load(cls, config, path=None):
        from transformers import AutoTokenizer

        student = config["student"]
        tokenizer = AutoTokenizer.from_pretrained(
            path or student["model"],
            **({} if path else {"revision": student["revision"]}),
        )
        return cls(tokenizer, student["enable_thinking"])

    def __call__(self, text, **kwargs):
        return self.tokenizer(text, **kwargs)

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(
            messages, enable_thinking=self.enable_thinking, **kwargs
        )

    def text_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count(self, messages, generation_prompt=False):
        normalized = normalize_messages_for_chat_template(messages)
        if normalized is None:
            raise ValueError("tool arguments must be JSON objects")
        rendered = self.apply_chat_template(
            normalized, tools=SHOP_TOOL_SCHEMAS, tokenize=False,
            add_generation_prompt=generation_prompt,
        )
        return self.text_tokens(rendered)


@dataclass
class EpisodeBudget:
    config: dict
    tokenizer: StudentTokenizer
    initial_tokens: int

    def state(self, messages):
        total = self.tokenizer.count(messages)
        return {
            "context_tokens": self.tokenizer.count(messages, generation_prompt=True),
            "response_tokens": max(0, total - self.initial_tokens),
        }

    def next_limit(self, messages):
        budget = self.config["budgets"]
        state = self.state(messages)
        remaining = budget["response_tokens"] - state["response_tokens"]
        if remaining <= 0:
            raise BudgetExceeded("response_budget_exhausted")
        limit = min(remaining, budget["assistant_tokens_per_turn"])
        if state["context_tokens"] + limit + budget["context_safety_margin"] > budget["context_tokens"]:
            raise BudgetExceeded("context_budget_exhausted")
        return limit, state

    def validate_assistant(self, before, after, limit):
        cost = max(0, self.tokenizer.count(after) - self.tokenizer.count(before, generation_prompt=True))
        if cost > limit or self.state(after)["response_tokens"] > self.config["budgets"]["response_tokens"]:
            raise BudgetExceeded("response_budget_exhausted")


def visible_observation(config, tokenizer, name, observation, parameters=None):
    settings = config["observation"]
    text, meta = project_observation(
        tool_name=name, observation=observation, parameters=parameters,
        count_tokens=tokenizer.text_tokens, **settings,
    )
    return text, meta.to_dict()
