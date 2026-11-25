"""Model generation adapters for the bilingual experiment."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import http.client
from ipaddress import ip_address
import json
from math import isfinite
import os
from threading import Lock
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


OLLAMA_OPTIONS = {
    "temperature": 0.2,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "num_ctx": 4096,
    "num_predict": 2048,
}
RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
DEFAULT_RATE_LIMIT_WAIT = 60
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class BackendError(RuntimeError):
    def __init__(self, message, *, retryable=False):
        super().__init__(message)
        self.retryable = retryable


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _is_loopback(host):
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_url(base_url, *, local):
    parts = urlsplit(base_url)
    if not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("model endpoint must omit credentials, query parameters, and fragments")
    loopback = _is_loopback(parts.hostname)
    if local and not loopback:
        raise ValueError("Ollama endpoint must use a loopback address")
    if parts.scheme != "https" and (parts.scheme != "http" or not loopback):
        raise ValueError("model endpoints outside loopback require HTTPS")
    return base_url.rstrip("/")


def _retry_after_seconds(value):
    if not value:
        return DEFAULT_RATE_LIMIT_WAIT
    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (ValueError, OverflowError):
            return DEFAULT_RATE_LIMIT_WAIT
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
    return max(0, seconds) if isfinite(seconds) else DEFAULT_RATE_LIMIT_WAIT


class _JsonClient:
    def __init__(self, base_url, *, timeout, max_attempts, retry_delay, local,
                 min_request_interval_seconds=0):
        if timeout <= 0 or max_attempts < 1 or retry_delay < 0:
            raise ValueError("invalid model timeout or retry configuration")
        if (type(min_request_interval_seconds) not in (int, float)
                or not isfinite(min_request_interval_seconds) or min_request_interval_seconds < 0):
            raise ValueError("min_request_interval_seconds must be a finite, nonnegative number")
        self.base_url = _validate_url(base_url, local=local)
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self.local = local
        self.min_request_interval_seconds = min_request_interval_seconds
        self._request_lock = Lock()
        self._next_request_at = 0
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    def _wait_for_request_slot(self):
        while True:
            with self._request_lock:
                now = time.monotonic()
                wait = self._next_request_at - now
                if wait <= 0:
                    self._next_request_at = now + self.min_request_interval_seconds
                    return
            time.sleep(min(wait, DEFAULT_RATE_LIMIT_WAIT))

    def request(self, path, body=None, token=None):
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(self.base_url + path, data=data, headers=headers)
        self._wait_for_request_slot()
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as error:
            error.close()
            if error.code == 429 and not self.local:
                retry_after = error.headers.get("Retry-After") if error.headers else None
                wait_seconds = _retry_after_seconds(retry_after)
                with self._request_lock:
                    self._next_request_at = max(self._next_request_at, time.monotonic() + wait_seconds)
            raise BackendError(
                f"HTTP {error.code}", retryable=error.code in RETRYABLE_HTTP_CODES,
            ) from None
        except (URLError, TimeoutError, ConnectionError, http.client.HTTPException) as error:
            reason = error.reason if isinstance(error, URLError) else error
            raise BackendError(f"transport error ({type(reason).__name__})", retryable=True) from None
        try:
            if len(raw) <= MAX_RESPONSE_BYTES:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        raise BackendError("model endpoint returned an invalid JSON object")

    def generate(self, path, body, options, parse, *, token=None):
        for attempt in range(1, self.max_attempts + 1):
            try:
                reply = parse(self.request(path, body, token))
                return {**reply, "options": options, "error": None, "attempts": attempt}
            except BackendError as error:
                if not error.retryable or attempt == self.max_attempts:
                    return {
                        "raw_reply": None, "done_reason": None, "options": options,
                        "error": str(error), "attempts": attempt,
                    }
                time.sleep(self.retry_delay)
        raise AssertionError("max_attempts is required to be positive")


def _ollama_tag(model):
    return model if ":" in model.rsplit("/", 1)[-1] else model + ":latest"


def _ollama_reply(body):
    try:
        content = body["message"]["content"]
    except (KeyError, TypeError):
        raise BackendError("Ollama response has no assistant content") from None
    if body.get("done") is not True or not isinstance(content, str):
        raise BackendError("Ollama did not return a complete text reply")
    return {
        "raw_reply": content, "done_reason": body.get("done_reason"),
        "response_model": body.get("model"),
    }


class OllamaBackend:
    def __init__(
        self, model, base_url="http://127.0.0.1:11434", *, timeout=120,
        max_attempts=3, retry_delay=1,
    ):
        if _ollama_tag(model).endswith((":cloud", "-cloud")) or model.endswith("-cloud"):
            raise ValueError("Ollama backend requires a model with local weights")
        self.model = model
        self.client = _JsonClient(
            base_url, timeout=timeout, max_attempts=max_attempts,
            retry_delay=retry_delay, local=True,
        )

    def describe(self):
        models = self.client.request("/api/tags").get("models")
        info = next((
            item for item in models
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and _ollama_tag(item["name"]) == _ollama_tag(self.model)
        ), None) if isinstance(models, list) else None
        if info is None:
            raise BackendError(f"Ollama model is not installed: {self.model}")
        if info.get("remote_host") or info.get("remote_model"):
            raise BackendError(f"Ollama model does not have local weights: {self.model}")
        digest = info.get("digest")
        if not isinstance(digest, str) or not digest:
            raise BackendError(f"Ollama did not report a digest for {self.model}")
        version = self.client.request("/api/version").get("version")
        if not isinstance(version, str) or not version:
            raise BackendError("Ollama did not report its backend version")
        details = info.get("details") or {}
        return {
            "backend": "ollama", "base_url": self.client.base_url,
            "model": self.model, "model_digest": digest,
            "model_version": digest, "backend_version": version,
            "quantization": details.get("quantization_level"),
            "options": dict(OLLAMA_OPTIONS),
        }

    def generate(self, prompt, seed=None):
        options = dict(OLLAMA_OPTIONS)
        if seed is not None:
            options["seed"] = seed
        body = {
            "model": self.model, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": options,
        }
        return self.client.generate("/api/chat", body, options, _ollama_reply)


REMOTE_OPTIONS = {"temperature": 0.2, "top_p": 1.0, "max_tokens": 2048}
REMOTE_OPTION_NAMES = {*REMOTE_OPTIONS, "max_completion_tokens", "reasoning_effort", "seed"}


def _remote_reply(body):
    try:
        choice = body["choices"][0]
        message = choice["message"]
        content = message["content"]
    except (IndexError, KeyError, TypeError):
        raise BackendError("remote response has no assistant content") from None
    if content is None:
        content = message.get("refusal") or ""
    if not isinstance(content, str):
        raise BackendError("remote response does not contain a text reply")
    return {
        "raw_reply": content, "done_reason": choice.get("finish_reason"),
        "response_model": body.get("model"), "system_fingerprint": body.get("system_fingerprint"),
    }


class RemoteBackend:
    """Optional OpenAI-compatible adapter, configured and activated by the caller."""

    def __init__(
        self, model, base_url, *, api_key_env=None, model_version=None, timeout=120,
        max_attempts=3, retry_delay=1, options=None, min_request_interval_seconds=0,
    ):
        self.model = model
        self.model_version = model_version
        self.api_key_env = api_key_env
        self.options = dict(REMOTE_OPTIONS if options is None else options)
        if self.options.keys() - REMOTE_OPTION_NAMES:
            raise ValueError("remote options must contain supported sampling, reasoning, and output parameters only")
        if {"max_tokens", "max_completion_tokens"} <= self.options.keys():
            raise ValueError("configure only one remote output-token limit")
        self.client = _JsonClient(
            base_url, timeout=timeout, max_attempts=max_attempts,
            retry_delay=retry_delay, local=False, min_request_interval_seconds=min_request_interval_seconds,
        )

    def describe(self):
        return {
            "backend": "openai-compatible", "base_url": self.client.base_url,
            "model": self.model, "model_digest": None,
            "model_version": self.model_version, "options": dict(self.options),
        }

    def generate(self, prompt, seed=None):
        options = dict(self.options)
        if "seed" in options and seed is not None:
            options["seed"] = seed
        timestamp = datetime.now(timezone.utc).isoformat()
        token = os.environ.get(self.api_key_env) if self.api_key_env else None
        if self.api_key_env and not token:
            return {
                "raw_reply": None, "done_reason": None, "options": options,
                "error": f"API key environment variable is unset: {self.api_key_env}",
                "attempts": 0, "generated_at": timestamp,
            }
        body = {
            "model": self.model, "stream": False,
            "messages": [{"role": "user", "content": prompt}], **options,
        }
        result = self.client.generate("/chat/completions", body, options, _remote_reply, token=token)
        return {**result, "generated_at": timestamp}
