"""Bounded caller waits; Python threads and out-of-process COM are never killed.

An unresponsive unit is quarantined with its files. Only its owner thread may
close the driver. A stop request is cooperative and is not proof of termination.
"""
from __future__ import annotations

from contextvars import copy_context
from dataclasses import dataclass, field
import threading
import time
from typing import Any
import uuid


@dataclass
class ExecutionUnit:
    model_path: str
    unit_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    stop_requested: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    value: Any = None
    error: BaseException | None = None
    cleanup_errors: list[str] = field(default_factory=list)
    driver: Any = None
    timed_out: bool = False
    execution_state: str = "not_started"
    cleanup_state: str = "not_started"

    def snapshot(self):
        return {"unit_id": self.unit_id, "model_path": self.model_path,
                "wait_timed_out": self.timed_out,
                "execution_state": self.execution_state,
                "cleanup_state": self.cleanup_state,
                "stop_requested": self.stop_requested.is_set(),
                "primary_error": str(self.error) if self.error else None,
                "cleanup_errors": list(self.cleanup_errors),
                "aspen_process_state": "unknown", "owned_process_ids": []}

    def close_driver(self):
        self.cleanup_state = "cleaning"
        if self.driver is not None:
            fn = getattr(self.driver, "disconnect", None) or getattr(self.driver, "close", None)
            if fn is None and hasattr(self.driver, "__exit__"):
                fn = lambda: self.driver.__exit__(None, None, None)
            if fn:
                try:
                    fn()
                except Exception as exc:
                    self.cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        self.cleanup_state = "incomplete" if self.cleanup_errors else "completed"

    def invoke(self, fn, *, timeout, cleanup_timeout, notify=lambda status: None):
        context = copy_context()
        def work():
            self.execution_state = "running"
            try:
                self.value = fn()
            except BaseException as exc:
                self.error = exc
            finally:
                # Returning from a Python call does not prove a timed-out COM
                # engine stopped. Drivers explicitly expose this distinction.
                unknown = self.timed_out or bool(getattr(self.driver, "needs_recovery", False))
                self.execution_state = "unknown" if unknown else "returned"
                self.close_driver()
                self.done.set()
        thread = threading.Thread(target=lambda: context.run(work), name=f"pao-unit-{self.unit_id}", daemon=True)
        thread.start()
        deadline = time.monotonic() + timeout
        while not self.done.wait(max(0.0, min(.1, deadline - time.monotonic()))):
            if time.monotonic() >= deadline:
                self.timed_out = True
                self.stop_requested.set()
                self.execution_state = "unknown"
                notify("timeout")
                notify("cleaning")
                if not self.done.wait(cleanup_timeout):
                    self.cleanup_state = "incomplete"
                    self.cleanup_errors.append("worker did not return within cleanup deadline; execution may continue; files retained")
                self.execution_state = "unknown"
                if self.error is None:
                    self.error = TimeoutError("caller deadline exceeded; underlying execution stop unconfirmed")
                break
        notify("cleanup_completed" if self.cleanup_state == "completed" else "cleanup_incomplete")
        if self.timed_out:
            raise TimeoutError("caller deadline exceeded; underlying execution stop is unconfirmed")
        if self.error:
            raise self.error
        return self.value
