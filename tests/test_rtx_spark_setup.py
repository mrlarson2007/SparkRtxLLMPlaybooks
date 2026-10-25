import os
import socket
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rtx_spark_setup as rss

FLASH_NEXT = {
    "id": "qwen/qwen3.8-flash-next/rtx-spark/windows/llama-cpp/78gb/iq4-xs/vision-mtp",
    "hardware": "rtx-spark",
    "platform": {"os": "windows", "execution": "native"},
    "backend": "llama-cpp",
    "backend_version": "b11617",
    "backend_min_version": "b11370",
    "min_vidmem_gb": 78,
    "precision": "iq4-xs",
    "variant": "vision-mtp",
    "status": "optimized",
    "checkpoint": {
        "target": "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/UD-IQ4_XS/"
        "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf",
        "draft": "https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF/blob/main/mtp-Qwen3.8-Flash-Next-Q8_0.gguf",
    },
    "serve": {
        "port": 9931,
        "command": textwrap.dedent("""\
            .\\llama-server.exe -hf "unsloth/Qwen3.8-Flash-Next-GGUF" `
              -hff "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf" `
              --mmproj-url "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/38bb39ee/mmproj-BF16.gguf" `
              --host 127.0.0.1 `
              --port 9931 `
              --ctx-size 262144 `
              -fa on `
              --spec-type draft-mtp `
              -hfrd "ggml-org/Qwen3.8-Flash-Next-GGUF" `
              --spec-draft-model "mtp-Qwen3.8-Flash-Next-Q8_0.gguf\""""),
    },
}

BOLT_E4B = {
    "id": "nvidia/nemotron-bolt-e4b/rtx-spark/windows/llama-cpp/q4-k-m",
    "hardware": "rtx-spark",
    "platform": {"os": "windows", "execution": "native"},
    "backend": "llama-cpp",
    "backend_version": "b11803",
    "backend_min_version": None,
    "min_vidmem_gb": 14.55,
    "status": "experimental",
    "checkpoint": {
        "target": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB/blob/main/"
        "Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf",
        "draft": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash/blob/main/"
        "Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf",
    },
    "serve": {
        "port": 8080,
        "command": "llama-server.exe `\n  -m ./Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf `\n  --no-mmproj `\n"
        "  --port 8080 `\n  --spec-draft-model ./Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf `\n"
        "  --chat-template-kwargs '{\"enable_thinking\":true}'",
    },
}

CATALOG = {
    "models": [
        {"id": "qwen/qwen3.8-flash-next", "display_name": "Qwen3.8-Flash-Next", "recipes": [FLASH_NEXT]},
        {
            "id": "nvidia/nemotron-bolt-e4b",
            "display_name": "Nemotron Bolt E4B",
            "recipes": [BOLT_E4B, dict(BOLT_E4B, id="other", min_vidmem_gb=10, status="optimized")],
        },
    ]
}


def _local_files(recipe, root=Path("/models")):
    local = {}
    for f in rss.model_files_for_recipe(recipe):
        local.setdefault(f.role, []).append(f.local_path(root))
    return local


class RecipeSelectionTests(unittest.TestCase):
    def test_memory_budget_capped_at_video_memory_limit(self):
        self.assertEqual(rss.memory_budget_gb("rtx-spark", 128), (96, 96))
        self.assertEqual(rss.memory_budget_gb("rtx-spark", 127.6), (96, 96))
        self.assertEqual(rss.memory_budget_gb("rtx-spark", 128, requested_gb=112), (96, 96))
        self.assertEqual(rss.memory_budget_gb("rtx-spark", 128, requested_gb=80), (80, 96))
        self.assertEqual(rss.memory_budget_gb("rtx-spark", 64), (48, None))
        self.assertEqual(rss.memory_budget_gb("geforce-rtx", 128), (112, None))
        self.assertEqual(rss.memory_budget_gb("rtx-spark", None), (None, None))

    def test_filters_by_memory_budget(self):
        too_small = rss.find_recipes(
            CATALOG, "rtx-spark", "windows", "llama-cpp", "qwen/qwen3.8-flash-next", memory_budget_gb=64
        )
        self.assertEqual(too_small, [])
        found = rss.find_recipes(
            CATALOG, "rtx-spark", "windows", "llama-cpp", "Qwen3.8-Flash-Next", memory_budget_gb=112
        )
        self.assertEqual([r["id"] for _, r in found], [FLASH_NEXT["id"]])

    def test_choose_prefers_optimized_then_largest(self):
        cands = rss.find_recipes(CATALOG, model="nvidia/nemotron-bolt-e4b")
        self.assertEqual(rss.choose_recipe(cands)[1]["id"], "other")
        self.assertEqual(rss.choose_recipe(cands, BOLT_E4B["id"])[1]["id"], BOLT_E4B["id"])
        with self.assertRaises(ValueError):
            rss.choose_recipe(cands, "missing")
        with self.assertRaises(ValueError):
            rss.choose_recipe([])


class ModelFileTests(unittest.TestCase):
    def test_hf_flags_and_shards(self):
        files = rss.model_files_for_recipe(FLASH_NEXT)
        shard = "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-{:05d}-of-00003.gguf"
        self.assertEqual(
            [(f.role, f.repo_id, f.filename, f.revision) for f in files],
            [
                ("target", "unsloth/Qwen3.8-Flash-Next-GGUF", shard.format(1), "main"),
                ("target", "unsloth/Qwen3.8-Flash-Next-GGUF", shard.format(2), "main"),
                ("target", "unsloth/Qwen3.8-Flash-Next-GGUF", shard.format(3), "main"),
                ("draft", "ggml-org/Qwen3.8-Flash-Next-GGUF", "mtp-Qwen3.8-Flash-Next-Q8_0.gguf", "main"),
                ("mmproj", "unsloth/Qwen3.8-Flash-Next-GGUF", "mmproj-BF16.gguf", "38bb39ee"),
            ],
        )

    def test_local_model_flags_use_checkpoint(self):
        files = rss.model_files_for_recipe(BOLT_E4B)
        self.assertEqual(
            [(f.role, f.repo_id) for f in files],
            [
                ("target", "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB"),
                ("draft", "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash"),
            ],
        )

    def test_folder_checkpoint(self):
        recipe = {
            "checkpoint": {"target": "https://huggingface.co/org/Repo-GGUF/tree/main/Q4"},
            "serve": {"command": "llama-server -m Q4/Model-Q4-00001-of-00002.gguf -np 1"},
        }
        names = [f.filename for f in rss.model_files_for_recipe(recipe)]
        self.assertEqual(names, ["Q4/Model-Q4-00001-of-00002.gguf", "Q4/Model-Q4-00002-of-00002.gguf"])


class LaunchCommandTests(unittest.TestCase):
    def test_rewrites_hf_flags_to_local_files(self):
        plan = rss.build_launch_command(FLASH_NEXT, "/bin/llama-server", _local_files(FLASH_NEXT), port=1234)
        argv = plan.argv
        for flag in ("-hf", "-hff", "-hfrd", "--mmproj-url"):
            self.assertNotIn(flag, argv)
        self.assertEqual(argv[0], "/bin/llama-server")
        root = Path("/models")
        self.assertEqual(
            argv[argv.index("-m") + 1],
            str(root / "unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"),
        )
        self.assertEqual(argv[argv.index("--mmproj") + 1], str(root / "unsloth/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf"))
        self.assertEqual(
            argv[argv.index("--spec-draft-model") + 1],
            str(root / "ggml-org/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"),
        )
        self.assertEqual(argv[argv.index("--ctx-size") + 1], "262144")
        self.assertEqual((plan.host, plan.port), ("127.0.0.1", 1234))

    def test_relative_model_paths_and_json_values(self):
        plan = rss.build_launch_command(BOLT_E4B, "llama-server", _local_files(BOLT_E4B), host="0.0.0.0")
        self.assertEqual(plan.argv.count("-m"), 1)
        self.assertFalse(any(a.startswith("./") for a in plan.argv))
        self.assertIn('{"enable_thinking":true}', plan.argv)
        self.assertEqual((plan.host, plan.port), ("0.0.0.0", 8080))
        self.assertIn("'{\"enable_thinking\":true}'", plan.powershell())

    def test_without_local_files_keeps_recipe_flags(self):
        plan = rss.build_launch_command(FLASH_NEXT, "llama-server", None)
        self.assertIn("-hf", plan.argv)
        self.assertNotIn("-m", plan.argv)


class IdempotencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    @unittest.skipIf(os.name == "nt", "uses a POSIX shell script as a fake llama-server")
    def test_install_skipped_when_build_present(self):
        build_dir = self.root / "b11617"
        build_dir.mkdir()
        exe = build_dir / rss.LLAMA_SERVER_EXE
        exe.write_text("#!/bin/sh\necho 'version: 11617 (abc)' >&2\n")
        exe.chmod(0o755)
        (build_dir / rss.INSTALL_MARKER).write_text("b11617\n")
        with mock.patch.object(rss, "_release_asset_urls") as urls:
            self.assertEqual(rss.install_llama_cpp(FLASH_NEXT, self.root), exe)
            urls.assert_not_called()

    @unittest.skipIf(os.name == "nt", "uses a POSIX shell script as a fake llama-server")
    def test_install_redone_when_incomplete(self):
        build_dir = self.root / "b11617"
        build_dir.mkdir()
        exe = build_dir / rss.LLAMA_SERVER_EXE
        exe.write_text("#!/bin/sh\necho 'version: 11617 (abc)' >&2\n")
        exe.chmod(0o755)
        # No completion marker: the install must not be skipped (non-Windows then refuses to install).
        with self.assertRaisesRegex(RuntimeError, "Windows ARM64"):
            rss.install_llama_cpp(FLASH_NEXT, self.root)

    def test_download_skips_complete_files(self):
        files = rss.model_files_for_recipe(BOLT_E4B)
        existing = files[0].local_path(self.root)
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"12345")

        def fake_download(repo_id, filename, revision, local_dir, token):
            path = Path(local_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"12345")
            return str(path)

        fake_hf = types.SimpleNamespace(hf_hub_download=mock.Mock(side_effect=fake_download))
        with mock.patch.object(rss, "_ensure_huggingface_hub", return_value=fake_hf), \
                mock.patch.object(rss, "remote_file_size", return_value=5):
            paths = rss.download_model_files(files, self.root)
        self.assertEqual(fake_hf.hf_hub_download.call_count, 1)
        self.assertEqual(fake_hf.hf_hub_download.call_args.kwargs["filename"], files[1].filename)
        self.assertEqual(paths["target"], [existing])
        self.assertTrue(paths["draft"][0].exists())

    def test_download_rejects_when_disk_too_small(self):
        files = rss.model_files_for_recipe(BOLT_E4B)
        with mock.patch.object(rss, "_ensure_huggingface_hub", return_value=types.SimpleNamespace()), \
                mock.patch.object(rss, "remote_file_size", return_value=10**18), self.assertRaises(RuntimeError):
            rss.download_model_files(files, self.root)

    def test_server_start_and_skip_when_healthy(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        fake = self.root / "fake_server.py"
        fake.write_text(textwrap.dedent("""\
            import http.server, json, sys
            port = int(sys.argv[sys.argv.index("--port") + 1])
            model = sys.argv[sys.argv.index("-m") + 1]
            class H(http.server.BaseHTTPRequestHandler):
                def _send(self, body):
                    data = json.dumps(body).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                def do_GET(self):
                    if self.path == "/health":
                        self._send({"status": "ok"})
                    elif self.path == "/props":
                        self._send({"model_path": model})
                    else:
                        self._send({"data": [{"id": "alias"}]})
                def do_POST(self):
                    self._send({"choices": [{"message": {"content": "hi"}}]})
                def log_message(self, *a):
                    pass
            http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
        """))
        argv = [sys.executable, str(fake), "-m", str(self.root / "a.gguf"), "--port", str(port)]
        plan = rss.LaunchPlan(argv=argv, port=port)
        log = self.root / "server.log"
        proc = rss.start_server(plan, log, timeout_s=30, poll_s=0.2)
        try:
            self.assertIsNotNone(proc)
            self.assertTrue(rss.pid_file_for(log).exists())
            self.assertIsNone(rss.start_server(plan, log))
            self.assertEqual(rss.chat(plan.base_url, "hello"), "hi")
            other = rss.LaunchPlan(argv=[*argv[:3], "b.gguf", *argv[4:]], port=port)
            with self.assertRaisesRegex(RuntimeError, "different model"):
                rss.start_server(other, log)
            # After a kernel restart the Popen handle is gone; the saved PID is used instead.
            rss.stop_server(None, log, plan.base_url)
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        self.assertFalse(rss.pid_file_for(log).exists())


if __name__ == "__main__":
    unittest.main()
