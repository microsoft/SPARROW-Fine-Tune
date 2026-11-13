"""Event bus and SSE framing.

One multiplexed stream, not one per topic. Browsers cap concurrent HTTP/1.1
connections to a single origin at around six, so separate ``/events/gpu``,
``/events/run`` and ``/events/jobs`` endpoints would exhaust the budget with two
or three tabs open and later requests would simply hang.

Two rules that keep a slow client from harming the application:

* Publishing is **never** allowed to block. A subscriber that stops draining its
  queue loses its oldest events and is flagged as lagged; it is not permitted to
  stall the poller thread that feeds everyone else.
* Every event carries the complete current value of whatever it describes, not a
  delta. A client that reconnects, or that dropped events while lagging, is
  correct again on the next tick without any replay protocol.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Set

#: Dropped when a subscriber stops draining. Large enough to ride out a slow
#: render, small enough that a dead tab cannot pin much memory.
QUEUE_SIZE = 256


class Subscription:
    def __init__(self, topics: Set[str], size: int = QUEUE_SIZE):
        self.topics = topics
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=size)
        self.lagged = False
        self.created_at = time.time()

    def matches(self, topic: str) -> bool:
        return topic in self.topics or "*" in self.topics

    def offer(self, event: Dict[str, Any]) -> None:
        """Enqueue without blocking, dropping the oldest event if full."""
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            self.lagged = True
            try:
                self.queue.get_nowait()      # make room by discarding the oldest
                self.queue.put_nowait(event)
            except (queue.Empty, queue.Full):
                pass

    def drain(self, timeout: float = 15.0) -> Optional[Dict[str, Any]]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None


class EventBus:
    def __init__(self, max_clients: int = 16):
        self._subs: List[Subscription] = []
        self._lock = threading.Lock()
        self._seq = 0
        self.max_clients = max_clients

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._subs)

    def subscribe(self, topics: Set[str]) -> Optional[Subscription]:
        """Returns ``None`` when at capacity, so the view can answer 503 rather
        than letting abandoned tabs exhaust the thread pool."""
        with self._lock:
            if len(self._subs) >= self.max_clients:
                return None
            sub = Subscription(topics)
            self._subs.append(sub)
            return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    def publish(self, topic: str, event_type: str, payload: Any) -> int:
        """Fan out to matching subscribers. Returns how many received it."""
        with self._lock:
            self._seq += 1
            event = {"id": self._seq, "topic": topic, "event": event_type, "data": payload}
            targets = [s for s in self._subs if s.matches(topic)]
        for sub in targets:
            sub.offer(event)
        return len(targets)


def format_sse(event: Dict[str, Any]) -> str:
    """Render one SSE frame."""
    payload = dict(event.get("data") or {})
    if isinstance(payload, dict):
        payload.setdefault("topic", event.get("topic"))
    body = json.dumps(payload, default=str)
    return f"id: {event.get('id', 0)}\nevent: {event.get('event', 'message')}\ndata: {body}\n\n"


def sse_stream(sub: Subscription, hello: Optional[Dict[str, Any]] = None,
               ping_interval: float = 15.0) -> Iterator[str]:
    """Generator for a Flask streaming response.

    Sends a ``hello`` frame carrying the full current state so a client never
    has to race a separate GET against the stream, then keeps the connection warm
    with comment pings. The pings matter: between epochs this stream is silent
    for twenty minutes at a time on a large dataset, and proxies and browsers
    will drop an idle connection.
    """
    if hello is not None:
        yield format_sse({"id": 0, "topic": "hello", "event": "hello", "data": hello})

    last_ping = time.time()
    while True:
        event = sub.drain(timeout=min(ping_interval, 5.0))
        if event is not None:
            if sub.lagged:
                # Tell the client it missed events so it can refetch rather than
                # quietly rendering a gap.
                data = dict(event.get("data") or {})
                data["lagged"] = True
                event = {**event, "data": data}
                sub.lagged = False
            yield format_sse(event)
        now = time.time()
        if now - last_ping >= ping_interval:
            last_ping = now
            yield ": ping\n\n"
