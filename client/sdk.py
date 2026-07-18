"""Python client SDK for the deployed Document Analyst (Part 3).

`DocumentAnalystClient` wraps the deployed OpenAI-compatible serving endpoint with
production niceties: env-based auth, exponential-backoff retries on 429/503,
clear timeouts, streaming, a READY health check, and typed errors.

    from client.sdk import DocumentAnalystClient
    client = DocumentAnalystClient("my-endpoint")
    print(client.ask("What was the net income in 2023?"))
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator

import httpx


class AnalystClientError(Exception):
    def __init__(self, message: str, status_code=None, request_id=None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


# HTTP statuses that are transient for a scale-to-zero model serving endpoint.
_RETRYABLE = {429, 503}


def _request_id(headers) -> str | None:
    for key in ("x-request-id", "x-databricks-request-id", "databricks-request-id"):
        if key in headers:
            return headers[key]
    return None


def _message_content(message) -> str | None:
    """Pull text out of a chat message (dict or LangChain-serialized)."""
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _extract_answer(data):
    """Extract the assistant's answer from any of the shapes the endpoint may return.

    A models-from-code LangGraph endpoint returns the raw final **state**
    (``[{"messages": [...], "final_answer": ...}]``) — the messages-out contract —
    rather than an OpenAI ``choices[]`` object. We also accept the OpenAI shape and
    MLflow's ``{"predictions": [...]}`` wrapper so the client is portable.
    """
    if isinstance(data, dict) and "predictions" in data:
        data = data["predictions"]
    if isinstance(data, list):
        if not data:
            return None
        data = data[0]
    if isinstance(data, dict):
        # OpenAI-compatible chat completion.
        if data.get("choices"):
            return _message_content(data["choices"][0].get("message", {}))
        # Raw LangGraph state: prefer the last message, fall back to final_answer.
        messages = data.get("messages")
        if messages:
            content = _message_content(messages[-1])
            if content:
                return content
        if data.get("final_answer"):
            return data["final_answer"]
    return None


class DocumentAnalystClient:
    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.endpoint_name = endpoint_name
        self.host = (host or os.environ.get("DATABRICKS_HOST", "")).rstrip("/")
        self.token = token or os.environ.get("DATABRICKS_TOKEN", "")
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.host or not self.token:
            raise AnalystClientError(
                "Missing credentials: pass host/token or set DATABRICKS_HOST and "
                "DATABRICKS_TOKEN in the environment."
            )

    # ─── URLs / headers ──────────────────────────────────────────────────────
    @property
    def _chat_url(self) -> str:
        # OpenAI-compatible surface: the model is selected by the `model` field.
        return f"{self.host}/serving-endpoints/chat/completions"

    @property
    def _status_url(self) -> str:
        return f"{self.host}/api/2.0/serving-endpoints/{self.endpoint_name}"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _payload(self, question: str, stream: bool) -> dict:
        return {
            "model": self.endpoint_name,
            "messages": [{"role": "user", "content": question}],
            "stream": stream,
        }

    # ─── Public API ──────────────────────────────────────────────────────────
    def ask(self, question: str) -> str:
        """Send a question and return the assistant's full answer."""
        response = self._post_with_retries(self._payload(question, stream=False))
        data = response.json()
        answer = _extract_answer(data)
        if answer is None:
            raise AnalystClientError(
                f"Unexpected response shape: {data!r}",
                status_code=response.status_code,
                request_id=_request_id(response.headers),
            )
        return answer

    def ask_streaming(self, question: str) -> Iterator[str]:
        """Yield answer text chunks as they arrive.

        Parses OpenAI-style SSE `data:` lines. A models-from-code endpoint may
        return a single (non-incremental) completion instead of token deltas — in
        that case we still yield the full answer exactly once.
        """
        payload = self._payload(question, stream=True)
        attempt = 0
        start = time.monotonic()
        while True:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    with client.stream(
                        "POST", self._chat_url, headers=self._headers(), json=payload
                    ) as response:
                        if response.status_code in _RETRYABLE and attempt < self.max_retries:
                            response.read()
                            self._sleep_backoff(attempt)
                            attempt += 1
                            continue
                        # A models-from-code endpoint often does not implement
                        # streaming and rejects `stream=true` (or returns a single
                        # non-SSE body). Per the spec this is a valid outcome: fall
                        # back to a single full-answer yield rather than erroring.
                        if response.status_code != 200:
                            yield self.ask(question)
                            return
                        yielded = yield from self._iter_sse(response)
                        if not yielded:
                            yield self.ask(question)
                        return
            except httpx.TimeoutException as exc:
                elapsed = time.monotonic() - start
                raise TimeoutError(
                    f"Streaming request to '{self.endpoint_name}' timed out after "
                    f"{elapsed:.2f}s (timeout={self.timeout}s)"
                ) from exc

    def health_check(self) -> bool:
        """Return True only when the serving endpoint reports READY."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(self._status_url, headers=self._headers())
        except httpx.TimeoutException:
            return False
        if response.status_code != 200:
            return False
        state = response.json().get("state", {})
        return state.get("ready") == "READY"

    # ─── Internals ───────────────────────────────────────────────────────────
    def _iter_sse(self, response) -> Iterator[str]:
        """Parse an SSE stream, yielding delta text. Returns True if anything yielded."""
        yielded = False
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("data:"):
                line = line[len("data:"):].strip()
            if line == "[DONE]":
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                # Non-SSE raw payload (e.g. a LangGraph state list) — let the
                # caller fall back to the full-answer path.
                continue
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content is None:
                    # Some endpoints emit the whole message rather than deltas.
                    message = choice.get("message") or {}
                    content = message.get("content")
                if content:
                    yielded = True
                    yield content
        return yielded

    def _post_with_retries(self, payload: dict):
        start = time.monotonic()
        attempt = 0
        while True:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        self._chat_url, headers=self._headers(), json=payload
                    )
            except httpx.TimeoutException as exc:
                elapsed = time.monotonic() - start
                raise TimeoutError(
                    f"Request to '{self.endpoint_name}' timed out after "
                    f"{elapsed:.2f}s (timeout={self.timeout}s)"
                ) from exc

            if response.status_code in _RETRYABLE and attempt < self.max_retries:
                self._sleep_backoff(attempt)
                attempt += 1
                continue

            self._raise_for_status(response)
            return response

    def _sleep_backoff(self, attempt: int) -> None:
        # Exponential backoff: 1s, 2s, 4s, ... (capped at 30s).
        time.sleep(min(2**attempt, 30))

    def _raise_for_status(self, response) -> None:
        if response.status_code >= 400:
            try:
                body = response.json()
                message = body.get("message") or body.get("error") or str(body)
            except (json.JSONDecodeError, ValueError):
                message = getattr(response, "text", "") or "<no body>"
            raise AnalystClientError(
                f"Endpoint '{self.endpoint_name}' returned {response.status_code}: {message}",
                status_code=response.status_code,
                request_id=_request_id(response.headers),
            )
