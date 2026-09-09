"""Structured handling of untrusted text before it reaches model context."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import binascii
import re


@dataclass(frozen=True)
class ScanFinding:
    category: str
    severity: str
    matched: str
    start: int

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


_PATTERNS = (
    ("prior_sanitization", "low", re.compile(r"\[blocked-untrusted-instruction\]", re.I)),
    ("instruction_override", "high", re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?", re.I)),
    ("system_prompt_probe", "high", re.compile(r"(?:reveal|show|print|repeat|give).{0,48}(?:system\s+prompt|developer\s+message|hidden\s+instructions?)", re.I)),
    ("role_redefinition", "high", re.compile(r"(?:you\s+are\s+now|act\s+as|new\s+role|override\s+(?:your\s+)?role)", re.I)),
    ("data_exfiltration", "high", re.compile(r"(?:exfiltrat|send\s+(?:all\s+)?(?:secrets?|tokens?|credentials?)|upload\s+(?:the\s+)?(?:workspace|files?))", re.I)),
    ("tool_authority_escalation", "medium", re.compile(r"(?:run\s+(?:this\s+)?command|disable\s+(?:safety|approval)|bypass\s+(?:approval|policy))", re.I)),
)
_HIDDEN_HTML = re.compile(r"<(?P<tag>[a-z][a-z0-9]*)\b[^>]*(?:\bhidden\b|style\s*=\s*['\"][^'\"]*(?:display\s*:\s*none|visibility\s*:\s*hidden)[^'\"]*['\"])[^>]*>.*?</(?P=tag)\s*>", re.I | re.S)
_BASE64_TOKEN = re.compile(r"(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{24,}={0,2})(?![A-Za-z0-9+/=])")


def scan_and_sanitize(text: str | None, *, max_chars: int = 120_000) -> tuple[str, tuple[ScanFinding, ...]]:
    """Replace suspicious directives while retaining an auditable finding record."""
    raw = text or ""
    findings: list[ScanFinding] = []
    spans: list[tuple[int, int]] = []
    for category, severity, pattern in _PATTERNS:
        for match in pattern.finditer(raw):
            findings.append(ScanFinding(category, severity, match.group(0)[:120], match.start()))
            spans.append((match.start(), match.end()))
    # Hidden DOM text is not a grounded UI affordance. Remove it wholesale so
    # pages cannot smuggle instructions in CSS-hidden containers.
    for match in _HIDDEN_HTML.finditer(raw):
        findings.append(ScanFinding("hidden_html_content", "medium", match.group(0)[:120], match.start()))
        spans.append((match.start(), match.end()))
    # Decode only small base64-like tokens. We do not treat encoded prose as an
    # instruction by itself; it is blocked only when its decoded text matches
    # the same hostile-directive patterns used for visible text.
    for match in _BASE64_TOKEN.finditer(raw):
        token = match.group(1)
        if len(token) > 8192:
            continue
        try:
            decoded = base64.b64decode(token, validate=True).decode("utf-8")
        except (ValueError, binascii.Error, UnicodeDecodeError):
            continue
        if any(pattern.search(decoded) for _, _, pattern in _PATTERNS[1:]):
            findings.append(ScanFinding("encoded_instruction", "high", f"base64:{decoded[:96]}", match.start()))
            spans.append((match.start(), match.end()))
    for start, end in sorted(spans, reverse=True):
        raw = raw[:start] + "[blocked-untrusted-instruction]" + raw[end:]
    return raw[:max_chars], tuple(findings)
