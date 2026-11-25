import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from prompt_language_comparison.backends import OllamaBackend, RemoteBackend
from prompt_language_comparison.experiment import make_backends


REPLY = "def answer():\n    return 1\n"


def response(body):
    return io.BytesIO(json.dumps(body).encode("utf-8"))


class ModelBackendsTest(unittest.TestCase):
    @patch("urllib.request.OpenerDirector.open")
    def test_local_model_records_digest_and_submits_independent_prompts(self, open_url):
        open_url.side_effect = [
            response({"models": [{"name": "qwen2.5:3b", "digest": "digest-123",
                                  "details": {"quantization_level": "Q4_K_M"}}]}),
            response({"version": "test-ollama"}),
            response({"message": {"content": REPLY}, "done": True, "done_reason": "stop"}),
            response({"message": {"content": REPLY}, "done": True, "done_reason": "stop"}),
        ]
        model = OllamaBackend("qwen2.5:3b")
        metadata = model.describe()
        self.assertEqual(metadata["model_digest"], "digest-123")
        self.assertEqual(metadata["backend_version"], "test-ollama")
        self.assertEqual(metadata["quantization"], "Q4_K_M")

        for language, seed in (("English", 17), ("中文", 18)):
            generated = model.generate(language, seed)
            request = open_url.call_args.args[0]
            wire = json.loads(request.data)
            self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
            self.assertEqual(wire["messages"], [{"role": "user", "content": language}])
            self.assertEqual(wire["model"], "qwen2.5:3b")
            self.assertIs(wire["stream"], False)
            self.assertEqual(wire["options"]["temperature"], 0.2)
            self.assertEqual(wire["options"]["top_p"], 1.0)
            self.assertEqual(wire["options"]["top_k"], 0)
            self.assertEqual(wire["options"]["repeat_penalty"], 1.0)
            self.assertEqual(wire["options"]["num_predict"], 2048)
            self.assertEqual(wire["options"]["seed"], seed)
            self.assertEqual(generated["options"], wire["options"])
            self.assertEqual(generated["raw_reply"], REPLY)
            self.assertEqual(generated["done_reason"], "stop")
            self.assertIsNone(generated["error"])

    @patch("urllib.request.OpenerDirector.open")
    def test_remote_slot_retries_without_changing_the_request(self, open_url):
        open_url.side_effect = [
            HTTPError("https://provider.invalid/v1/chat/completions", 429, "wait",
                      {"Retry-After": "0"}, io.BytesIO()),
            response({"model": "provider-model-1", "choices": [
                {"message": {"content": REPLY}, "finish_reason": "stop"},
            ]}),
        ]
        with patch.dict(os.environ, {"EXPERIMENT_TEST_TOKEN": "fake-key"}):
            model = RemoteBackend(
                "provider-model", "https://provider.invalid/v1",
                api_key_env="EXPERIMENT_TEST_TOKEN", model_version="provider-model-1",
                retry_delay=0,
            )
            metadata = model.describe()
            self.assertEqual(metadata["model_version"], "provider-model-1")
            self.assertEqual(open_url.call_count, 0)
            result = model.generate("说明", 123)

        requests = [call.args[0] for call in open_url.call_args_list]
        self.assertEqual(len(requests), 2)
        first, second = requests
        self.assertEqual(first.full_url, "https://provider.invalid/v1/chat/completions")
        self.assertEqual(first.get_header("Authorization"), "Bearer fake-key")
        self.assertEqual(first.data, second.data)
        self.assertEqual(json.loads(first.data), {
            "model": "provider-model", "stream": False,
            "messages": [{"role": "user", "content": "说明"}],
            "temperature": 0.2, "top_p": 1.0, "max_tokens": 2048,
        })
        for item in (metadata, result):
            self.assertNotIn("fake-key", json.dumps(item))
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["raw_reply"], REPLY)
        self.assertEqual(result["response_model"], "provider-model-1")
        self.assertEqual(result["done_reason"], "stop")
        self.assertRegex(result["generated_at"], r"^\d{4}-\d{2}-\d{2}T")

    @patch("urllib.request.OpenerDirector.open")
    def test_remote_request_spacing_and_rate_limit_cooldown(self, open_url):
        config = {"models": [{
            "id": "provider", "model": "provider-model", "backend": "remote", "enabled": True,
            "base_url": "https://provider.invalid/v1", "min_request_interval_seconds": 22,
        }]}
        for headers, cooldown in (({"Retry-After": "45"}, 45), ({}, 60)):
            with self.subTest(headers=headers):
                clock = 0
                starts = []

                def wait(seconds):
                    nonlocal clock
                    clock += seconds

                def serve(request, *, timeout):
                    starts.append(clock)
                    if clock < cooldown:
                        raise HTTPError(request.full_url, 429, "wait", headers, io.BytesIO())
                    return response({"choices": [{"message": {"content": REPLY}}]})

                open_url.side_effect = serve
                backend = make_backends(config)["provider"]
                with patch("prompt_language_comparison.backends.time.monotonic", side_effect=lambda: clock), \
                        patch("prompt_language_comparison.backends.time.sleep", side_effect=wait):
                    first = backend.generate("English")
                    second = backend.generate("中文")

                self.assertEqual(starts, [0, cooldown, cooldown + 22])
                self.assertEqual(first["raw_reply"], REPLY)
                self.assertEqual(second["raw_reply"], REPLY)

    @patch("urllib.request.OpenerDirector.open")
    def test_network_failure_retries_locally_without_changing_seed(self, open_url):
        open_url.side_effect = [
            URLError(TimeoutError()),
            response({"message": {"content": REPLY}, "done": True, "done_reason": "length"}),
        ]
        result = OllamaBackend("qwen2.5:3b", retry_delay=0).generate("题目", 312)
        calls = [json.loads(call.args[0].data) for call in open_url.call_args_list]
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["options"]["seed"], 312)
        self.assertEqual(result["done_reason"], "length")
        self.assertEqual(result["raw_reply"], REPLY)

    @patch("urllib.request.OpenerDirector.open")
    def test_both_backends_record_exhausted_requests_as_missing(self, open_url):
        for model in (
            OllamaBackend("qwen2.5:3b", max_attempts=2, retry_delay=0),
            RemoteBackend("provider-model", "https://provider.invalid/v1",
                          max_attempts=2, retry_delay=0),
        ):
            with self.subTest(model=type(model).__name__):
                open_url.side_effect = [
                    URLError(TimeoutError()),
                    URLError(ConnectionRefusedError()),
                ]
                result = model.generate("Question", 42)
                self.assertEqual(result["attempts"], 2)
                self.assertIsNone(result["raw_reply"])
                self.assertEqual(result["error"], "transport error (ConnectionRefusedError)")
                self.assertIsNone(result["done_reason"])

    @patch("urllib.request.OpenerDirector.open")
    def test_remote_blank_reply_is_a_response_and_missing_key_makes_no_request(self, open_url):
        open_url.return_value = response({
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
        })
        model = RemoteBackend("provider-model", "https://provider.invalid/v1")
        self.assertEqual(model.generate("Question")["raw_reply"], "")
        self.assertEqual(open_url.call_count, 1)

        model = RemoteBackend(
            "provider-model", "https://provider.invalid/v1", api_key_env="NOT_SET_IN_TEST",
        )
        with patch.dict(os.environ, {}, clear=True):
            result = model.generate("Question")
        self.assertIsNone(result["raw_reply"])
        self.assertEqual(result["attempts"], 0)
        self.assertEqual(open_url.call_count, 1)
