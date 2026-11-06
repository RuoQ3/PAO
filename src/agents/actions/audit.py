from __future__ import annotations
import json
from dataclasses import dataclass, field
from .models import AuditEvent, AuditEventType

@dataclass
class AuditLog:
    events: list[AuditEvent] = field(default_factory=list)
    def append(self, event: AuditEvent): self.events.append(event)
    def record(self, audit_id, event_type, *, actor="system", status=None, message="", payload=None):
        event = AuditEvent(event_id=__import__('uuid').uuid4().hex, audit_id=audit_id, event_type=event_type, actor=actor, status=status, message=message, payload=payload or {})
        self.append(event); return event
    def to_dict(self): return {"events": [e.to_dict() for e in self.events]}
    def to_json(self, **kwargs): return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, **kwargs)
    @classmethod
    def from_dict(cls, raw): return cls([AuditEvent.from_dict(e) for e in raw.get("events", [])])
    @classmethod
    def from_json(cls, raw): return cls.from_dict(json.loads(raw))
