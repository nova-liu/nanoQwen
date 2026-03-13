import argparse
import json
import logging
import mlx.core as mx


from pathlib import Path


from mlx_lm.server import run, ModelProvider

def main():
    config_path = Path("config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config_data = json.loads(config_path.read_text(encoding="utf-8"))

    if not isinstance(config_data, dict):
        raise ValueError("Config root must be a JSON object")

    defaults = {
        "model": None,
        "adapter_path": None,
        "host": "127.0.0.1",
        "port": 8080,
        "draft_model": None,
        "num_draft_tokens": 3,
        "trust_remote_code": False,
        "log_level": "INFO",
        "chat_template": "",
        "use_default_chat_template": False,
        "temp": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "max_tokens": 512,
        "chat_template_args": {},
        "decode_concurrency": 32,
        "prompt_concurrency": 8,
        "prefill_step_size": 2048,
        "prompt_cache_size": 10,
        "prompt_cache_bytes": None,
        "pipeline": False,
        "host": "127.0.0.1",
        "port": 8080,
    }
    merged_config = {**defaults, **config_data}

    if isinstance(merged_config.get("chat_template_args"), str):
        merged_config["chat_template_args"] = json.loads(
            merged_config["chat_template_args"]
        )

    if isinstance(merged_config.get("prompt_cache_bytes"), str):
        merged_config["prompt_cache_bytes"] = parse_size(
            merged_config["prompt_cache_bytes"]
        )

    if merged_config.get("chat_template_args") is None:
        merged_config["chat_template_args"] = {}

    args = argparse.Namespace(**merged_config)
    if mx.metal.is_available():
        wired_limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(wired_limit)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run(args.host, args.port, ModelProvider(args))


if __name__ == "__main__":
    main()
