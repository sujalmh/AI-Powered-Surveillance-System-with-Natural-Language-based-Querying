"""Provider-agnostic LLM client for the NL parser.

Two transports, one interface:

``opencode``
    Drives the local **opencode** runtime (https://opencode.ai) with the
    free DeepSeek V4.1 Flash model.  The runtime is asked for a JSON object
    and the reply is decoded with a tolerant extractor, because small
    "flash" models tend to wrap JSON in markdown fences and to add a short
    remark after it.  The prompt is flattened to a single line because the
    Windows command line cannot carry embedded newlines reliably.

``openai`` / ``openrouter`` / ``ollama`` / ``openai_compatible``
    Plain **OpenAI-compatible** ``POST {base_url}/chat/completions`` calls.
    ``openai_compatible`` is the escape hatch for any server that speaks the
    OpenAI chat-completions contract (vLLM, LM Studio, llama.cpp, TGI,
    DeepSeek, OpenRouter, ...).

The module deliberately has no third-party dependency: it uses the standard
library only, so the backend keeps working when LangChain is absent.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]


class LLMError(RuntimeError):
    """Raised when no LLM backend could produce a usable reply."""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def _balanced_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` block, ignoring braces in strings."""
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start:i + 1]
    return None


def _repair(blob: str) -> str:
    """Fix the small syntactic slips small models make when emitting JSON."""
    out = blob
    # ``"key\:"`` -> ``"key":`` and ``"key\"`` -> ``"key"`` (stray escapes).
    out = re.sub(r"\\([:\",])", r"\1", out)
    out = re.sub(r"\\n(?=\s*[}\]])", "", out)
    # Unquoted keys -> quoted keys.
    out = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', out)
    # Trailing commas before a closing bracket.
    out = re.sub(r",(\s*[}\]])", r"\1", out)
    return out


def render_prompt(template: str, **values: Any) -> str:
    """Substitute ``{placeholder}`` tokens, leaving unknown ones untouched.

    The parser prompt carries ``{query}``, ``{schema}``,
    ``{format_instructions}`` and ``{reference_date}``.  A missing value used
    to reach the model literally, which made it resolve relative dates
    against the session date instead of the archive epoch.
    """
    class _Safe(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    try:
        return template.format_map(_Safe(values))
    except (ValueError, IndexError):
        out = template
        for key, val in values.items():
            out = out.replace("{" + key + "}", str(val))
        return out


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract a JSON object from a model reply, tolerating fences and prose.

    Raises:
        LLMError: if no JSON object can be recovered.
    """
    if not text or not text.strip():
        raise LLMError("empty LLM reply")
    candidates: List[str] = []
    for match in _FENCE.finditer(text):
        candidates.append(match.group(1).strip())
    candidates.append(text.strip())
    for candidate in candidates:
        for blob in (candidate, _balanced_object(candidate) or ""):
            if not blob:
                continue
            for attempt in (blob, _repair(blob)):
                try:
                    obj = json.loads(attempt)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(obj, dict):
                    return obj
    raise LLMError(f"no JSON object in LLM reply: {text[:200]!r}")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

@dataclass
class LLMReply:
    text: str
    model: str
    transport: str
    raw: str = ""


class OpenAICompatBackend:
    """Any server that speaks the OpenAI ``/chat/completions`` contract."""

    transport = "openai-compatible"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, prompt: str, system: Optional[str] = None, json_mode: bool = False,
             max_tokens: Optional[int] = None, temperature: float = 0.0) -> LLMReply:
        import urllib.error
        import urllib.request

        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            raise LLMError(f"chat/completions failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - network path
            raise LLMError(f"chat/completions failed: {exc}") from exc
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected chat/completions payload: {str(data)[:200]}") from exc
        return LLMReply(text=text or "", model=self.model, transport=self.transport, raw=str(data)[:2000])


class OpencodeBackend:
    """Drives the local opencode runtime with ``opencode run``.

    The runtime is asked for a single JSON object; the reply is read from the
    NDJSON event stream that ``--format json`` emits.
    """

    transport = "opencode"

    def __init__(self, model: str, server: Optional[str] = None, cli: Optional[str] = None,
                 timeout: int = 600, cwd: Optional[Path] = None,
                 fallbacks: Optional[List[str]] = None) -> None:
        self.model = model
        self.server = server
        self.timeout = timeout
        self.cwd = cwd or ROOT
        self.cli = cli or self._find_cli()
        # Further models to try when the provider refuses the request (the free
        # tier of a model has a daily quota).
        self.fallbacks = [m for m in (fallbacks or []) if m and m != model]

    @staticmethod
    def _find_cli() -> str:
        found = shutil.which("opencode")
        if found:
            return found
        for candidate in (
            Path.home() / "AppData/Roaming/npm/opencode.cmd",
            Path.home() / ".local/bin/opencode",
            Path("/usr/local/bin/opencode"),
        ):
            if candidate.exists():
                return str(candidate)
        raise LLMError(
            "opencode CLI not found; install opencode or set LLM_PROVIDER to an "
            "OpenAI-compatible endpoint"
        )

    @staticmethod
    def _flatten(prompt: str) -> str:
        return " ".join(prompt.split())

    def chat(self, prompt: str, system: Optional[str] = None, json_mode: bool = False,
             max_tokens: Optional[int] = None, temperature: float = 0.0) -> LLMReply:
        """Send one prompt, trying the fallback models if the provider refuses.

        The free tier of a model has a daily quota; when it is exhausted the
        runtime answers with an error event instead of text.  In that case the
        next configured model is used, so the parser keeps working.
        """
        errors = []
        for model in [self.model] + self.fallbacks:
            self.model = model
            try:
                return self._chat_once(prompt, system=system, json_mode=json_mode,
                                       max_tokens=max_tokens, temperature=temperature)
            except LLMError as exc:
                errors.append(f"{model}: {exc}")
        raise LLMError("all opencode models failed -> " + " | ".join(errors))

    def _chat_once(self, prompt: str, system: Optional[str] = None, json_mode: bool = False,
                   max_tokens: Optional[int] = None, temperature: float = 0.0) -> LLMReply:
        """Send one prompt to the opencode runtime.

        The parser prompt (instructions + JSON schema + query) is longer than
        the Windows command-line limit, so it is handed to the runtime as an
        attached file and the message asks for the JSON object it describes.
        """
        cmd = [self.cli, "run"]
        if self.server:
            cmd += ["--server", self.server]
        cmd += ["-m", self.model, "--format", "json"]
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".md", prefix="nl_prompt_",
                                             delete=False, encoding="utf-8") as handle:
                handle.write(prompt)
                tmp_path = Path(handle.name)
            cmd += ["--file", str(tmp_path)]
            cmd.append(
                "The attached file contains the parser instructions, the JSON schema "
                "and the user query. Reply with ONLY the JSON object for that query: "
                "no prose, no markdown fences, no explanation."
            )
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout, cwd=str(self.cwd)
                )
            except subprocess.TimeoutExpired as exc:
                raise LLMError(f"opencode run timed out after {self.timeout}s") from exc
            except OSError as exc:
                raise LLMError(f"opencode run failed: {exc}") from exc
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        texts: List[str] = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "text":
                part = event.get("part") or {}
                if part.get("text"):
                    texts.append(str(part["text"]))
            elif event.get("type") == "error":
                err = event.get("error") or {}
                raise LLMError(f"opencode error: {err.get('message', event)[:200]}")
        if not texts:
            raise LLMError(
                f"opencode produced no text (exit {proc.returncode}): "
                f"{proc.stderr[-300:] or proc.stdout[-300:]}"
            )
        return LLMReply(text="\n".join(texts).strip(), model=self.model,
                         transport=self.transport, raw=proc.stdout[-2000:])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_backend(cfg: Optional[Dict[str, Any]] = None) -> Any:
    """Create the LLM backend described by the active configuration."""
    from backend.app.config import settings

    llm_cfg = cfg or settings.get_active_llm_config()
    provider = (llm_cfg.get("provider") or "").strip().lower()
    model = llm_cfg.get("model") or ""
    api_key = llm_cfg.get("api_key") or ""

    if provider == "opencode":
        # A model id for the opencode runtime is always "provider/model".
        # Anything else (e.g. a stale "gpt-4o-mini" in .env) is ignored in
        # favour of the configured opencode model.
        if "/" not in model:
            model = settings.OPENCODE_MODEL
        fallbacks = [
            m.strip() for m in (settings.OPENCODE_MODEL_FALLBACKS or "").split(",")
            if m.strip()
        ]
        return OpencodeBackend(
            model=model or settings.OPENCODE_MODEL,
            server=settings.OPENCODE_BASE_URL or None,
            cli=settings.OPENCODE_CLI or None,
            fallbacks=fallbacks,
        )
    if provider in ("openai_compatible", "vllm", "lmstudio", "llamacpp", "tgi"):
        base = settings.OPENAI_BASE_URL or llm_cfg.get("base_url") or ""
        if not base:
            raise LLMError("OPENAI_BASE_URL is required for an OpenAI-compatible endpoint")
        return OpenAICompatBackend(base, api_key, model, timeout=settings.NL_TIMEOUT_SECONDS)
    if provider in ("openai", "openrouter", "ollama", ""):
        base = settings.OPENAI_BASE_URL or (
            settings.OPENROUTER_BASE_URL if provider == "openrouter"
            else "https://api.openai.com/v1"
        )
        if provider == "ollama":
            base = settings.OLLAMA_BASE_URL.rstrip("/") + "/v1"
        key = api_key or settings.OPENAI_API_KEY
        if provider == "openai" and not key and not settings.OPENAI_BASE_URL:
            raise LLMError("OPENAI_API_KEY is not set")
        return OpenAICompatBackend(base, key, model or settings.NL_DEFAULT_MODEL,
                                   timeout=settings.NL_TIMEOUT_SECONDS)
    raise LLMError(f"Unsupported LLM provider: '{provider}'")


def chat(
    prompt: str,
    *,
    system: Optional[str] = None,
    json_mode: bool = False,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    backend: Optional[Any] = None,
) -> LLMReply:
    """Single entry point used by the parser and the evaluation scripts."""
    backend = backend or build_backend()
    return backend.chat(prompt, system=system, json_mode=json_mode,
                        max_tokens=max_tokens, temperature=temperature)


def chat_json(prompt: str, *, system: Optional[str] = None, backend: Optional[Any] = None) -> Dict[str, Any]:
    """Chat and decode a JSON object from the reply."""
    reply = chat(prompt, system=system, json_mode=True, backend=backend)
    return extract_json_object(reply.text)


# ---------------------------------------------------------------------------
# On-disk cache
# ---------------------------------------------------------------------------

class ParseCache:
    """JSON-lines cache so a parse run is reproducible and cheap to repeat."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data: Dict[str, Any] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, key: str) -> Optional[Any]:
        return self._data.get(key)

    def put(self, key: str, value: Any) -> None:
        self._data[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False),
                             encoding="utf-8")


def default_cache_path(model: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    return ROOT / "backend" / "benchmarks" / "parsed_cache" / f"{safe}.json"
