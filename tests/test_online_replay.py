import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import online_replay


def _job(enable_kv_evict: bool, *, forward: bool):
    line = json.dumps(
        {
            "ts": 1,
            "conv_id": "conv-1",
            "body": {
                "prompt": [{"role": "user", "content": "hello"}],
                "enable_kv_evict": enable_kv_evict,
                "kv_transfer_params": {
                    "conversation_id": "spoofed",
                    "truncated": True,
                },
            },
        }
    )
    config = {
        "api_base": "http://localhost:8000/v1",
        "api_key": "test",
        "model": "model",
        "use_chat": True,
        "max_tokens": 1,
        "temperature": 0.0,
        "prefer_log_body": False,
        "min_p_supported": False,
        "forward_kv_evict": forward,
    }
    job = online_replay.process_log_line(line, ep_config=config)
    assert job is not None
    return job


def test_enabled_forwarding_uses_top_level_flag_not_kv_transfer_params():
    job = _job(True, forward=True)

    assert online_replay._build_extra_body(job.body) == {
        "enable_kv_evict": True
    }
    assert job.conversation_id == "conv-1"


def test_forwarding_is_disabled_by_default():
    job = _job(True, forward=False)

    assert online_replay._build_extra_body(job.body) == {}


def test_enabled_forwarding_preserves_false_value():
    job = _job(False, forward=True)

    assert online_replay._build_extra_body(job.body) == {
        "enable_kv_evict": False
    }


def test_send_request_preserves_header_and_uses_sdk_extra_body():
    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if hasattr(self, "sent"):
                raise StopAsyncIteration
            self.sent = True
            return SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    class Completions:
        async def create(self, **kwargs):
            self.kwargs = kwargs
            return Stream()

    completions = Completions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )

    result = asyncio.run(
        online_replay.send_request(client, _job(True, forward=True))
    )

    assert result[1] == "OK"
    assert completions.kwargs["extra_headers"]["X-Flow-Conversation-Id"] == "conv-1"
    assert completions.kwargs["extra_body"] == {"enable_kv_evict": True}


def test_forwarding_requires_chat_endpoint():
    result = subprocess.run(
        [
            sys.executable,
            str(Path(online_replay.__file__)),
            "--forward-kv-evict",
            "--disable-min-p",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--forward-kv-evict requires --use-chat" in result.stderr
