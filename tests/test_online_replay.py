import asyncio
import json
import queue
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import online_replay


def test_qps_replay_waits_for_inflight_request_after_reader_exhaustion(
    monkeypatch,
):
    async def scenario():
        release = asyncio.Event()
        request_started = asyncio.Event()

        async def fake_dispatch(_client, job, _sequencer):
            request_started.set()
            await release.wait()
            return (job.request_id, "OK", 0.1, 0.2, 10, 2, None)

        class Collector:
            def __init__(self):
                self.round_start_time = time.perf_counter() - 1
                self.current_round_request_ids = set()
                self.results_queue = queue.Queue()
                self.jobs_processed = 0
                self.round_draining = False
                self.total_requests = 0
                self.successful_requests = 0

            def increment_jobs_processed(self):
                self.jobs_processed += 1

            def collect_results(self):
                while not self.results_queue.empty():
                    result = self.results_queue.get_nowait()
                    self.current_round_request_ids.discard(result[0])

            def check_and_report_metrics(self, qps=None):
                return False

        jobs = queue.Queue()
        jobs.put(
            SimpleNamespace(
                request_id="request-1",
                conversation_id="conv-1",
            )
        )
        done = online_replay.threading.Event()
        done.set()
        monkeypatch.setattr(online_replay, "job_queue", jobs)
        monkeypatch.setattr(online_replay, "reader_done_event", done)
        monkeypatch.setattr(online_replay, "dispatch_request", fake_dispatch)

        replay = asyncio.create_task(
            online_replay.replay_by_qps(
                object(),
                Collector(),
                target_qps=100,
                round_duration=30,
            )
        )
        await request_started.wait()
        await asyncio.sleep(0.15)
        assert not replay.done()

        release.set()
        await asyncio.wait_for(replay, timeout=1)

    asyncio.run(scenario())


def test_conversation_sequencer_waits_for_previous_request():
    async def scenario():
        sequencer = online_replay.ConversationSequencer()
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        second_started = asyncio.Event()

        async def first():
            first_started.set()
            await release_first.wait()
            return "first"

        async def second():
            second_started.set()
            return "second"

        first_task = asyncio.create_task(sequencer.run("conv-1", first))
        await first_started.wait()
        second_task = asyncio.create_task(sequencer.run("conv-1", second))
        await asyncio.sleep(0)

        assert not second_started.is_set()

        release_first.set()
        assert await asyncio.gather(first_task, second_task) == [
            "first",
            "second",
        ]

    asyncio.run(scenario())


def test_conversation_sequencer_keeps_different_conversations_concurrent():
    async def scenario():
        sequencer = online_replay.ConversationSequencer()
        both_started = asyncio.Event()
        release = asyncio.Event()
        started = set()

        async def request(name):
            started.add(name)
            if len(started) == 2:
                both_started.set()
            await release.wait()
            return name

        tasks = [
            asyncio.create_task(
                sequencer.run("conv-1", lambda: request("first"))
            ),
            asyncio.create_task(
                sequencer.run("conv-2", lambda: request("second"))
            ),
        ]
        await asyncio.wait_for(both_started.wait(), timeout=1)
        release.set()

        assert await asyncio.gather(*tasks) == ["first", "second"]

    asyncio.run(scenario())


def test_timed_dispatch_includes_conversation_wait(monkeypatch):
    async def scenario():
        sequencer = online_replay.ConversationSequencer()
        release = asyncio.Event()

        async def fake_send(_client, job):
            if job.request_id == "first":
                await release.wait()
            return (job.request_id, "OK", 0.01, 0.02, 10, 2, "")

        monkeypatch.setattr(online_replay, "send_request", fake_send)
        first = SimpleNamespace(request_id="first", conversation_id="conv")
        second = SimpleNamespace(request_id="second", conversation_id="conv")
        first_task = asyncio.create_task(
            online_replay.dispatch_request(
                object(), first, sequencer, scheduled_at=time.perf_counter()
            )
        )
        await asyncio.sleep(0)
        second_scheduled = time.perf_counter()
        second_task = asyncio.create_task(
            online_replay.dispatch_request(
                object(), second, sequencer, scheduled_at=second_scheduled
            )
        )
        await asyncio.sleep(0.02)
        release.set()
        first_result, second_result = await asyncio.gather(
            first_task, second_task
        )

        assert len(first_result) == 10
        assert second_result[8] - second_result[7] >= 0.02
        assert second_result[9] >= second_result[8]

    asyncio.run(scenario())


def test_continuous_qps_replay_drains_only_after_all_windows(monkeypatch):
    async def scenario():
        calls = []
        reports = []

        async def fake_dispatch(
            _client, job, _sequencer, scheduled_at=None
        ):
            calls.append(scheduled_at)
            await asyncio.sleep(0.04)
            wire_started = scheduled_at + 0.01
            return (
                job.request_id,
                "OK",
                0.01,
                0.04,
                10,
                2,
                "",
                scheduled_at,
                wire_started,
                scheduled_at + 0.04,
            )

        def fake_analysis(*_args, **kwargs):
            reports.append(kwargs["timing_metrics"])

        jobs = queue.Queue()
        for index in range(20):
            jobs.put(
                SimpleNamespace(
                    request_id=f"request-{index}",
                    conversation_id=f"conv-{index}",
                )
            )
        monkeypatch.setattr(online_replay, "job_queue", jobs)
        monkeypatch.setattr(online_replay, "dispatch_request", fake_dispatch)
        monkeypatch.setattr(online_replay, "results_analysis", fake_analysis)
        monkeypatch.setattr(
            online_replay,
            "args",
            SimpleNamespace(json_output=None),
            raising=False,
        )
        collector = online_replay.ResultCollector(
            {"model": "test"},
            round_duration=0.05,
            max_rounds=2,
            round_drain_timeout=1,
        )

        started = time.perf_counter()
        await online_replay.replay_by_qps_continuous(
            object(),
            collector,
            target_qps=100,
            round_duration=0.05,
            max_rounds=2,
        )

        assert len(reports) == 2
        assert any(call - started >= 0.05 for call in calls)
        assert all(report["continuous_qps_window"] for report in reports)

    asyncio.run(scenario())


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


def test_preselected_route_bypasses_legacy_md5_sampling():
    line = json.dumps(
        {
            "ts": 1,
            "conv_id": "conv-not-in-range",
            "body": {"prompt": [{"role": "user", "content": "hello"}]},
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
        "forward_kv_evict": False,
    }

    assert online_replay.process_log_line(
        line, 0.0, 0.0, config, preselected_route=True
    ) is not None


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


def test_preselected_route_rejects_md5_sampling():
    result = subprocess.run(
        [
            sys.executable,
            str(Path(online_replay.__file__)),
            "--preselected-route",
            "--sample-range",
            "0",
            "0.1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "cannot be combined" in result.stderr
