"""Event bus behaviour under load and neglect.

The failure that matters here is a browser tab that stops reading: it must not be
able to block the poller thread that feeds every other client.
"""

from __future__ import annotations

import json
import threading
import time

from sparrow_web.events import (QUEUE_SIZE, EventBus, Subscription, format_sse,
                                sse_stream)


class TestRouting:
    def test_only_matching_topics_receive(self):
        bus = EventBus()
        gpu = bus.subscribe({"gpu"})
        run = bus.subscribe({"run:7"})

        assert bus.publish("gpu", "gpu.sample", {"util": 91}) == 1
        assert gpu.drain(0.1)["data"]["util"] == 91
        assert run.drain(0.1) is None

    def test_wildcard_receives_everything(self):
        bus = EventBus()
        everything = bus.subscribe({"*"})
        bus.publish("gpu", "gpu.sample", {})
        bus.publish("run:1", "run.metrics", {})
        assert everything.drain(0.1) is not None
        assert everything.drain(0.1) is not None

    def test_multiple_topics_on_one_connection(self):
        """One EventSource per tab: browsers cap connections per origin."""
        bus = EventBus()
        sub = bus.subscribe({"gpu", "run:3", "jobs"})
        bus.publish("gpu", "gpu.sample", {"a": 1})
        bus.publish("run:3", "run.metrics", {"b": 2})
        bus.publish("jobs", "job.updated", {"c": 3})
        assert [sub.drain(0.1)["data"][k] for k in "abc"] == [1, 2, 3]

    def test_ids_increase_monotonically(self):
        bus = EventBus()
        sub = bus.subscribe({"gpu"})
        for _ in range(3):
            bus.publish("gpu", "gpu.sample", {})
        ids = [sub.drain(0.1)["id"] for _ in range(3)]
        assert ids == sorted(ids) and len(set(ids)) == 3

    def test_unsubscribe_stops_delivery(self):
        bus = EventBus()
        sub = bus.subscribe({"gpu"})
        bus.unsubscribe(sub)
        assert bus.publish("gpu", "gpu.sample", {}) == 0


class TestCapacity:
    def test_subscriber_limit_is_enforced(self):
        """Abandoned tabs must not exhaust the gthread pool."""
        bus = EventBus(max_clients=2)
        assert bus.subscribe({"gpu"}) is not None
        assert bus.subscribe({"gpu"}) is not None
        assert bus.subscribe({"gpu"}) is None, "accepted a client past the limit"

    def test_unsubscribing_frees_a_slot(self):
        bus = EventBus(max_clients=1)
        first = bus.subscribe({"gpu"})
        assert bus.subscribe({"gpu"}) is None
        bus.unsubscribe(first)
        assert bus.subscribe({"gpu"}) is not None


class TestSlowConsumer:
    def test_publishing_never_blocks_on_a_full_queue(self):
        """The whole point: one wedged tab cannot stall the GPU poller."""
        bus = EventBus()
        bus.subscribe({"gpu"})   # never drained

        start = time.time()
        for i in range(QUEUE_SIZE * 3):
            bus.publish("gpu", "gpu.sample", {"i": i})
        assert time.time() - start < 2.0, "publish blocked on a full subscriber queue"

    def test_oldest_events_are_dropped_and_client_is_flagged(self):
        sub = Subscription({"gpu"}, size=3)
        for i in range(6):
            sub.offer({"id": i, "topic": "gpu", "event": "s", "data": {"i": i}})

        assert sub.lagged is True
        seen = [sub.drain(0.05)["data"]["i"] for _ in range(3)]
        assert seen == [3, 4, 5], "kept the stale events instead of the newest"

    def test_lag_is_surfaced_to_the_client(self):
        """The client needs to know it missed events so it can refetch."""
        sub = Subscription({"gpu"}, size=2)
        for i in range(5):
            sub.offer({"id": i, "topic": "gpu", "event": "s", "data": {"i": i}})

        frames = []
        stream = sse_stream(sub, hello=None, ping_interval=99)
        frames.append(next(stream))
        assert '"lagged": true' in frames[0]

    def test_a_slow_client_does_not_affect_a_fast_one(self):
        bus = EventBus()
        slow = bus.subscribe({"gpu"})
        fast = bus.subscribe({"gpu"})
        for i in range(QUEUE_SIZE + 10):
            bus.publish("gpu", "gpu.sample", {"i": i})
            fast.drain(0.01)
        assert slow.lagged is True
        bus.publish("gpu", "gpu.sample", {"i": "final"})
        assert fast.drain(0.1)["data"]["i"] == "final"


class TestFraming:
    def test_frame_shape(self):
        frame = format_sse({"id": 7, "topic": "gpu", "event": "gpu.sample",
                            "data": {"util": 91}})
        assert frame.startswith("id: 7\nevent: gpu.sample\ndata: ")
        assert frame.endswith("\n\n")
        body = json.loads(frame.split("data: ", 1)[1].strip())
        assert body["util"] == 91 and body["topic"] == "gpu"

    def test_non_serialisable_values_do_not_raise(self, tmp_path):
        frame = format_sse({"id": 1, "topic": "t", "event": "e", "data": {"p": tmp_path}})
        assert str(tmp_path) in json.loads(frame.split("data: ", 1)[1].strip())["p"]


class TestStream:
    def test_hello_is_sent_first(self):
        """So a client never has to race a separate GET against the stream."""
        sub = Subscription({"gpu"})
        stream = sse_stream(sub, hello={"gpus": [], "runs": []}, ping_interval=99)
        first = next(stream)
        assert "event: hello" in first
        assert json.loads(first.split("data: ", 1)[1].strip())["gpus"] == []

    def test_ping_keeps_an_idle_stream_alive(self):
        """Between epochs this stream is silent for many minutes."""
        sub = Subscription({"gpu"})
        stream = sse_stream(sub, hello=None, ping_interval=0.01)
        time.sleep(0.05)
        assert next(stream) == ": ping\n\n"

    def test_events_are_yielded_as_they_arrive(self):
        sub = Subscription({"gpu"})
        stream = sse_stream(sub, hello=None, ping_interval=99)

        def publish_later():
            time.sleep(0.05)
            sub.offer({"id": 1, "topic": "gpu", "event": "gpu.sample", "data": {"util": 42}})

        threading.Thread(target=publish_later, daemon=True).start()
        frame = next(stream)
        assert json.loads(frame.split("data: ", 1)[1].strip())["util"] == 42
