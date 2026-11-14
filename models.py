"""Local Ollama access and a deterministic smoke-test backend."""

from __future__ import annotations

import hashlib
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class ModelError(RuntimeError):
    """The requested local inference could not complete."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def local_model_name(model: str) -> str:
    model = model.strip()
    if not model:
        raise ModelError("A local Ollama model tag is required.")
    last_segment = model.rsplit("/", 1)[-1]
    tag = last_segment.partition(":")[2].lower()
    if tag == "cloud" or tag.endswith("-cloud"):
        raise ModelError(f"Cloud model tags are unavailable in this study: {model}")
    return model if ":" in last_segment else f"{model}:latest"


def require_local_model(details: dict, model: str) -> None:
    if details.get("remote_model") or details.get("remote_host"):
        raise ModelError(f"{model} is backed by a remote model. Select downloaded local weights.")


class OllamaClient:
    """Use the Ollama native API over loopback only."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120,
        *,
        keep_alive: str | int = "5m",
    ):
        endpoint = urlsplit(base_url)
        if (
            endpoint.scheme != "http"
            or endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.path not in {"", "/"}
            or endpoint.query
            or endpoint.fragment
        ):
            raise ModelError("Ollama URL must be a local HTTP origin, such as http://127.0.0.1:11434.")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.opener = build_opener(ProxyHandler({}), NoRedirect())
        self.verified_models: set[str] = set()

    def _request(self, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            with error:
                detail = error.read(400).decode("utf-8", errors="replace").strip()
            raise ModelError(f"Ollama {path} returned HTTP {error.code}: {detail or error.reason}") from error
        except (TimeoutError, URLError, OSError) as error:
            raise ModelError(
                f"Could not reach Ollama at {self.base_url}{path}: {error}. Check that Ollama is running."
            ) from error
        except (ValueError, UnicodeError) as error:
            raise ModelError(f"Ollama {path} returned invalid JSON.") from error

        if not isinstance(result, dict):
            raise ModelError(f"Ollama {path} returned an unexpected response.")
        if result.get("error"):
            raise ModelError(f"Ollama {path}: {result['error']}")
        return result

    def metadata(self, models: list[str]) -> dict:
        names = {name: local_model_name(name) for name in models}
        version = self._request("/api/version").get("version")
        if not isinstance(version, str) or not version:
            raise ModelError("Ollama did not report its server version.")
        entries = self._request("/api/tags").get("models")
        if not isinstance(entries, list):
            raise ModelError("Ollama did not report its installed models.")
        installed = {
            entry.get("name") or entry.get("model"): entry
            for entry in entries
            if isinstance(entry, dict)
        }
        selected = {}
        for requested, name in names.items():
            entry = installed.get(name)
            if not entry:
                raise ModelError(f"Missing local model {name}. Run: ollama pull {name}")
            require_local_model(entry, name)
            require_local_model(self._request("/api/show", {"model": name}), name)
            digest = entry.get("digest")
            if not isinstance(digest, str) or not digest:
                raise ModelError(f"Ollama did not report the digest for {name}.")
            selected[requested] = {
                "name": name,
                "digest": digest,
                "details": entry.get("details", {}),
                "size_bytes": entry.get("size"),
            }

        self.verified_models.update(names.values())
        return {
            "backend": "ollama",
            "version": version,
            "models": selected,
            "temperature": 0.0,
            "think": False,
            "keep_alive": self.keep_alive,
        }

    def chat(
        self,
        model: str,
        messages: list[dict],
        schema: dict,
        *,
        seed: int,
        num_ctx: int,
        num_predict: int,
    ) -> dict:
        name = local_model_name(model)
        if name not in self.verified_models:
            self.metadata([model])
        response = self._request(
            "/api/chat",
            {
                "model": name,
                "messages": messages,
                "format": schema,
                "stream": False,
                "think": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "seed": seed,
                    "temperature": 0.0,
                    "num_ctx": num_ctx,
                    "num_predict": num_predict,
                },
            },
        )
        require_local_model(response, name)
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if response.get("done") is not True or not isinstance(content, str):
            raise ModelError("Ollama did not return a completed message.")
        usage_fields = (
            "prompt_eval_count", "prompt_eval_cached_count", "eval_count",
            "total_duration", "load_duration", "prompt_eval_duration", "eval_duration",
        )
        return {
            "content": content,
            "usage": {key: response[key] for key in usage_fields if key in response},
            "done_reason": response.get("done_reason"),
        }


class FakeClient:
    """Scripted responses deliberately exercise catches, misses, and supervisor errors."""

    def __init__(self, supervisor_model: str = "qwen3:8b-q4_K_M"):
        self.supervisor_model = supervisor_model

    def metadata(self, models: list[str]) -> dict:
        return {
            "backend": "scripted-smoke-only",
            "version": "1",
            "models": {
                model: {
                    "name": model,
                    "digest": hashlib.sha256(f"scripted-smoke-v1:{model}".encode()).hexdigest(),
                }
                for model in models
            },
        }

    def chat(
        self,
        model: str,
        messages: list[dict],
        schema: dict,
        *,
        seed: int,
        num_ctx: int,
        num_predict: int,
    ) -> dict:
        request = json.loads(messages[-1]["content"])
        stage = request["stage"]
        field = request["field"]
        source = next(iter(request["evidence"]), None)
        empty = {"value": None, "source": None}
        if stage == "worker":
            output = {"value": f"scripted-invalid-{field}", "source": source}
        elif stage == "review":
            output = {"verdict": "accept", **empty}
            if model == self.supervisor_model and request["candidate"]["value"] is not None:
                if field == "revision":
                    output["verdict"] = "reject"
                elif field == "exception":
                    output = {
                        "verdict": "correct",
                        "value": "scripted-invalid-supervisor-correction",
                        "source": source,
                    }
        elif stage == "synthesis":
            output = {slot: request["context"].get(slot, empty) for slot in request["fields"]}
        else:
            raise ModelError(f"Unknown scripted stage: {stage}")
        return {"content": json.dumps(output), "usage": {}, "done_reason": "stop"}
