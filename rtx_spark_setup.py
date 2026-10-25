"""Helpers for the RTX Spark local LLM setup notebook.

The notebook (``rtx_spark_setup.ipynb``) drives these helpers step by step.
Recipes come from NVIDIA's Local AI Inference Recipes catalog:
https://github.com/NVIDIA/local-ai-inference-recipes

Every step is idempotent: it first checks whether the work has already been
done and skips it if so, so the notebook can safely be re-run.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

CATALOG_URL = (
    "https://raw.githubusercontent.com/NVIDIA/local-ai-inference-recipes/main/catalog.json"
)
LLAMA_CPP_RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}"
DEFAULT_LLAMA_CPP_DIR = Path.home() / "llama-cuda"
DEFAULT_MODELS_DIR = Path.home() / "llm-models"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rtx-spark-setup"

IS_WINDOWS = os.name == "nt"
LLAMA_SERVER_EXE = "llama-server.exe" if IS_WINDOWS else "llama-server"


def _print_step(status: str, message: str) -> None:
    icons = {"skip": "✅ already done", "ok": "✅ done", "run": "⏳", "warn": "⚠️", "info": "ℹ️"}
    print(f"{icons.get(status, status)}  {message}")


# ---------------------------------------------------------------------------
# System detection
# ---------------------------------------------------------------------------

def detect_total_memory_gb() -> float | None:
    """Return total physical memory in GB, or ``None`` if it can't be detected."""
    try:
        if IS_WINDOWS:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return round(stat.ullTotalPhys / 1024**3, 1)
            return None
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (AttributeError, OSError, ValueError):
        return None


def detect_gpus() -> list[dict]:
    """Return GPUs reported by ``nvidia-smi`` (empty list if unavailable)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "driver": parts[1], "memory": parts[2]})
    return gpus


def detect_system() -> dict:
    """Collect basic facts about the current machine."""
    return {
        "os": {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}.get(
            platform.system(), platform.system().lower()
        ),
        "arch": platform.machine(),
        "python": sys.version.split()[0],
        "total_memory_gb": detect_total_memory_gb(),
        "gpus": detect_gpus(),
    }


# ---------------------------------------------------------------------------
# Catalog + recipe selection
# ---------------------------------------------------------------------------

def _http_get(url: str, headers: dict | None = None, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "rtx-spark-setup", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def load_catalog(
    url: str = CATALOG_URL,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    max_age_hours: float = 24,
    refresh: bool = False,
) -> dict:
    """Load the recipe catalog, re-using a cached copy if it's fresh enough."""
    cache_file = Path(cache_dir) / "catalog.json"
    if cache_file.exists() and not refresh:
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_h < max_age_hours:
            _print_step("skip", f"Catalog cached at {cache_file} ({age_h:.1f}h old)")
            return json.loads(cache_file.read_text(encoding="utf-8"))
    try:
        data = _http_get(url)
        catalog = json.loads(data)
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        if cache_file.exists():
            _print_step("warn", f"Could not refresh catalog ({exc}); using cached copy")
            return json.loads(cache_file.read_text(encoding="utf-8"))
        raise
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(data)
    _print_step("ok", f"Downloaded catalog from {url}")
    return catalog


def iter_recipes(catalog: dict) -> Iterable[tuple[dict, dict]]:
    for model in catalog.get("models", []):
        for recipe in model.get("recipes", []):
            yield model, recipe


def find_recipes(
    catalog: dict,
    hardware: str | None = None,
    os_name: str | None = None,
    backend: str | None = None,
    model: str | None = None,
    precision: str | None = None,
    variant: str | None = None,
    memory_budget_gb: float | None = None,
) -> list[tuple[dict, dict]]:
    """Return ``(model, recipe)`` pairs matching every filter that is not ``None``."""
    results = []
    for m, r in iter_recipes(catalog):
        if hardware and r.get("hardware") != hardware:
            continue
        if os_name and r.get("platform", {}).get("os") != os_name:
            continue
        if backend and r.get("backend") != backend:
            continue
        if model and model.lower() not in (m["id"].lower(), m.get("display_name", "").lower()):
            continue
        if precision and r.get("precision") != precision:
            continue
        if variant and r.get("variant") != variant:
            continue
        if memory_budget_gb is not None and (r.get("min_vidmem_gb") or 0) > memory_budget_gb:
            continue
        results.append((m, r))
    return results


def choose_recipe(candidates: list[tuple[dict, dict]], recipe_id: str | None = None) -> tuple[dict, dict]:
    """Pick a recipe: an explicit ``recipe_id`` wins, otherwise the best fit.

    "Best fit" prefers ``optimized`` recipes and then the one with the largest
    memory requirement that still fits (larger budgets get more context or a
    higher-quality quantization).
    """
    if recipe_id:
        for m, r in candidates:
            if r["id"] == recipe_id:
                return m, r
        raise ValueError(f"Recipe {recipe_id!r} is not among the matching recipes.")
    if not candidates:
        raise ValueError("No recipe matches your hardware / model / backend / memory selection.")
    return max(
        candidates,
        key=lambda mr: (mr[1].get("status") == "optimized", mr[1].get("min_vidmem_gb") or 0),
    )


def recipe_table(candidates: list[tuple[dict, dict]]) -> str:
    """Render matching recipes as a plain-text table."""
    rows = [("min GB", "status", "precision", "variant", "recipe id")]
    for _, r in sorted(candidates, key=lambda mr: (mr[1]["id"].split("/rtx")[0], mr[1].get("min_vidmem_gb") or 0)):
        rows.append((
            str(r.get("min_vidmem_gb")), r.get("status", ""), r.get("precision") or "",
            r.get("variant") or "", r["id"],
        ))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in rows)


# ---------------------------------------------------------------------------
# llama.cpp installation
# ---------------------------------------------------------------------------

def _build_number(version: str | None) -> int | None:
    if not version:
        return None
    match = re.search(r"(\d+)", str(version))
    return int(match.group(1)) if match else None


def llama_server_version(exe: Path | str) -> int | None:
    """Return the build number reported by ``llama-server --version``."""
    try:
        proc = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"version:\s*(\d+)", (proc.stdout or "") + (proc.stderr or ""))
    return int(match.group(1)) if match else None


def find_llama_server(install_dir: Path) -> Path | None:
    if not install_dir.exists():
        return None
    direct = install_dir / LLAMA_SERVER_EXE
    if direct.exists():
        return direct
    return next(iter(sorted(install_dir.rglob(LLAMA_SERVER_EXE))), None)


def _release_asset_urls(recipe: dict, tag: str, github_token: str | None) -> list[tuple[str, str]]:
    """Return ``(name, url)`` for the llama.cpp + CUDA runtime ZIPs of a release."""
    headers = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = "Bearer " + github_token
    try:
        release = json.loads(_http_get(LLAMA_CPP_RELEASE_API.format(tag=tag), headers=headers))
        assets = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        # Some recipes link the exact ZIPs in their notes; fall back to those.
        urls = re.findall(
            r"https://github\.com/ggml-org/llama\.cpp/releases/download/[^\s)\]]+\.zip",
            recipe.get("notes") or "",
        )
        if urls:
            return [(u.rsplit("/", 1)[-1], u) for u in urls]
        raise RuntimeError(f"Failed to query the GitHub Releases API: {exc}") from exc

    import fnmatch

    patterns = ["cudart-llama-bin-win-cuda-*-arm64.zip", "llama-*-bin-win-cuda-*-arm64.zip"]
    picked = []
    for pattern in patterns:
        name = next((n for n in sorted(assets) if fnmatch.fnmatch(n, pattern)), None)
        if not name:
            raise RuntimeError(f"No release asset matching {pattern!r} found in {tag}.")
        picked.append((name, assets[name]))
    return picked


def _download(url: str, dest: Path, headers: dict | None = None) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "rtx-spark-setup", **(headers or {})})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1024 * 1024)
    tmp.replace(dest)


def install_llama_cpp(
    recipe: dict,
    base_dir: Path | str = DEFAULT_LLAMA_CPP_DIR,
    github_token: str | None = None,
    force: bool = False,
) -> Path:
    """Install the pinned llama.cpp build for ``recipe`` and return ``llama-server`` path.

    Builds are installed side by side under ``base_dir/<tag>`` (default
    ``~/llama-cuda/b11617``). If a build at least as new as the recipe's
    minimum version is already installed there, the step is skipped.
    """
    tag = recipe.get("backend_version")
    if not tag or not str(tag).startswith("b"):
        raise ValueError(f"Recipe does not pin a llama.cpp build tag (backend_version={tag!r}).")
    min_build = _build_number(recipe.get("backend_min_version")) or _build_number(tag)
    install_dir = Path(base_dir).expanduser() / tag

    exe = find_llama_server(install_dir)
    if exe and not force:
        build = llama_server_version(exe)
        if build is None or build >= min_build:
            _print_step("skip", f"llama.cpp {tag} installed at {exe}")
            return exe
        _print_step("warn", f"{exe} reports build {build} < required {min_build}; reinstalling")

    if not IS_WINDOWS:
        raise RuntimeError(
            "Automatic llama.cpp install uses the Windows ARM64 CUDA release ZIPs. "
            "On this OS, build llama.cpp yourself and set LLAMA_SERVER_PATH in the notebook."
        )

    install_dir.mkdir(parents=True, exist_ok=True)
    for name, url in _release_asset_urls(recipe, tag, github_token):
        zip_path = install_dir / name
        if not zip_path.exists():
            _print_step("run", f"Downloading {name}")
            _download(url, zip_path)
        _print_step("run", f"Extracting {name}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(install_dir)

    exe = find_llama_server(install_dir)
    if not exe:
        raise RuntimeError(f"{LLAMA_SERVER_EXE} not found after extracting into {install_dir}.")
    _print_step("ok", f"Installed llama.cpp {tag} at {exe}")
    return exe


# ---------------------------------------------------------------------------
# Hugging Face model files
# ---------------------------------------------------------------------------

_HF_URL_RE = re.compile(
    r"^https://huggingface\.co/(?P<repo>[^/]+/[^/]+)/(?:blob|resolve|tree)/(?P<rev>[^/]+)/(?P<path>.+)$"
)
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


@dataclass(frozen=True)
class HFFile:
    repo_id: str
    filename: str
    revision: str = "main"
    role: str = "target"  # target | draft | mmproj

    def local_path(self, models_dir: Path) -> Path:
        return Path(models_dir) / self.repo_id / self.filename


def parse_hf_url(url: str, role: str = "target") -> HFFile:
    match = _HF_URL_RE.match(urllib.parse.unquote(url))
    if not match:
        raise ValueError(f"Not a Hugging Face file URL: {url}")
    return HFFile(match["repo"], match["path"], match["rev"], role)


def expand_shards(f: HFFile) -> list[HFFile]:
    """Expand ``...-00001-of-00003.gguf`` into every shard of a split checkpoint."""
    match = _SHARD_RE.search(f.filename)
    if not match:
        return [f]
    total = int(match.group(2))
    stem = f.filename[: match.start()]
    return [
        HFFile(f.repo_id, f"{stem}-{i:05d}-of-{total:05d}.gguf", f.revision, f.role)
        for i in range(1, total + 1)
    ]


def _tokenize_command(command: str) -> list[str]:
    """Split a recipe's PowerShell/bash launch command into argv tokens."""
    joined = re.sub(r"[`\\]\r?\n", " ", command)
    return shlex.split(joined, posix=True)


_TARGET_REPO = ("-hf", "-hfr", "--hf-repo")
_TARGET_FILE = ("-hff", "--hf-file")
_DRAFT_REPO = ("-hfrd", "-hfd", "--hf-repo-draft")
_DRAFT_MODEL = ("--spec-draft-model", "-md", "--model-draft")
_MMPROJ_URL = ("--mmproj-url", "-mmu")
_MODEL = ("-m", "--model")


def _option_values(tokens: list[str]) -> dict[str, str]:
    values = {}
    for i, tok in enumerate(tokens[:-1]):
        if tok.startswith("-"):
            values.setdefault(tok, tokens[i + 1])
    return values


def _first(values: dict[str, str], names: tuple[str, ...]) -> str | None:
    return next((values[n] for n in names if n in values), None)


def model_files_for_recipe(recipe: dict) -> list[HFFile]:
    """Work out every Hugging Face file the recipe's launch command needs."""
    tokens = _tokenize_command(recipe["serve"]["command"])
    values = _option_values(tokens)
    checkpoint = recipe.get("checkpoint") or {}
    files: list[HFFile] = []

    repo = _first(values, _TARGET_REPO)
    hf_file = _first(values, _TARGET_FILE)
    if repo:
        if not hf_file:
            raise ValueError(f"Recipe uses '-hf {repo}' without '-hff'; cannot resolve the file name.")
        files.append(HFFile(repo.split(":")[0], hf_file, "main", "target"))
    elif checkpoint.get("target"):
        target = parse_hf_url(checkpoint["target"], "target")
        local_model = _first(values, _MODEL)
        if not target.filename.endswith(".gguf") and local_model:
            # Checkpoint points at a folder; the -m flag names the file inside it.
            name = local_model.replace("\\", "/").removeprefix("./")
            if not name.startswith(target.filename.rstrip("/") + "/"):
                name = f"{target.filename.rstrip('/')}/{name.rsplit('/', 1)[-1]}"
            target = HFFile(target.repo_id, name, target.revision, "target")
        files.append(target)
    else:
        raise ValueError("Recipe has no Hugging Face target checkpoint.")

    draft_repo = _first(values, _DRAFT_REPO)
    draft_model = _first(values, _DRAFT_MODEL)
    if draft_repo and draft_model:
        files.append(HFFile(draft_repo, draft_model, "main", "draft"))
    elif draft_model and checkpoint.get("draft"):
        files.append(parse_hf_url(checkpoint["draft"], "draft"))

    mmproj_url = _first(values, _MMPROJ_URL)
    if mmproj_url:
        files.append(parse_hf_url(mmproj_url, "mmproj"))

    expanded: list[HFFile] = []
    for f in files:
        expanded.extend(expand_shards(f))
    return expanded


def _ensure_huggingface_hub():
    try:
        import huggingface_hub
    except ImportError:
        _print_step("run", "Installing huggingface_hub")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub"])
    import huggingface_hub

    return huggingface_hub


def remote_file_size(f: HFFile, token: str | None = None) -> int | None:
    hf = _ensure_huggingface_hub()
    try:
        meta = hf.get_hf_file_metadata(hf.hf_hub_url(f.repo_id, f.filename, revision=f.revision), token=token)
        return meta.size
    except Exception:  # noqa: BLE001 - offline, gated repo, etc.
        return None


def download_model_files(
    files: list[HFFile],
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    token: str | None = None,
) -> dict[str, list[Path]]:
    """Download ``files`` into ``models_dir/<repo_id>/<filename>``.

    Files that already exist locally with the expected size are skipped.
    Returns local paths grouped by role (``target``, ``draft``, ``mmproj``).
    """
    hf = _ensure_huggingface_hub()
    models_dir = Path(models_dir).expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)

    sizes = {f: remote_file_size(f, token) for f in files}
    missing = [f for f in files if not _is_complete(f.local_path(models_dir), sizes[f])]
    needed = sum(sizes[f] or 0 for f in missing)
    free = shutil.disk_usage(models_dir).free
    if needed and needed > free:
        raise RuntimeError(
            f"Not enough disk space in {models_dir}: need {needed / 1024**3:.1f} GB, "
            f"have {free / 1024**3:.1f} GB free."
        )

    paths: dict[str, list[Path]] = {}
    for f in files:
        local = f.local_path(models_dir)
        size = sizes[f]
        if f in missing:
            label = f" ({size / 1024**3:.2f} GB)" if size else ""
            _print_step("run", f"Downloading {f.repo_id}/{f.filename}{label}")
            hf.hf_hub_download(
                repo_id=f.repo_id, filename=f.filename, revision=f.revision,
                local_dir=models_dir / f.repo_id, token=token,
            )
            _print_step("ok", f"{local}")
        else:
            _print_step("skip", f"{local}")
        paths.setdefault(f.role, []).append(local)
    return paths


def _is_complete(path: Path, expected_size: int | None) -> bool:
    if not path.exists():
        return False
    return expected_size is None or path.stat().st_size == expected_size


# ---------------------------------------------------------------------------
# Launch command + server lifecycle
# ---------------------------------------------------------------------------

@dataclass
class LaunchPlan:
    argv: list[str]
    host: str = "127.0.0.1"
    port: int = 8080
    extra: dict = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def powershell(self) -> str:
        """Return the command formatted for copy/paste into PowerShell."""

        def q(arg: str) -> str:
            if re.fullmatch(r"[\w\-./:\\=]+", arg):
                return arg
            return "'" + arg.replace("'", "''") + "'"

        head, *rest = self.argv
        lines = [f"& {q(head)}"]
        i = 0
        while i < len(rest):
            if rest[i].startswith("-") and i + 1 < len(rest) and not rest[i + 1].startswith("-"):
                lines.append(f"{rest[i]} {q(rest[i + 1])}")
                i += 2
            else:
                lines.append(q(rest[i]))
                i += 1
        return " `\n  ".join(lines)


def build_launch_command(
    recipe: dict,
    llama_server: Path | str,
    local_files: dict[str, list[Path]] | None = None,
    host: str | None = None,
    port: int | None = None,
) -> LaunchPlan:
    """Turn the recipe's launch command into argv using local model files.

    Hugging Face download flags (``-hf``/``-hff``/``-hfrd``/``--mmproj-url``)
    are replaced with ``-m``/``--spec-draft-model``/``--mmproj`` pointing at
    the files fetched by :func:`download_model_files`. Every tuning flag from
    the recipe is kept as-is. If ``local_files`` is ``None`` the original
    flags are kept and llama-server downloads into its own cache.
    """
    tokens = _tokenize_command(recipe["serve"]["command"])
    args = tokens[1:]
    out: list[str] = [str(llama_server)]

    drop_with_value = set(_TARGET_REPO + _TARGET_FILE + _DRAFT_REPO) if local_files else set()
    replace = {}
    if local_files:
        if local_files.get("target"):
            out += ["-m", str(local_files["target"][0])]
        if local_files.get("draft"):
            replace.update({n: str(local_files["draft"][0]) for n in _DRAFT_MODEL})
        if local_files.get("mmproj"):
            drop_with_value |= set(_MMPROJ_URL)
            out += ["--mmproj", str(local_files["mmproj"][0])]

    i = 0
    while i < len(args):
        tok = args[i]
        has_value = i + 1 < len(args)
        if tok in drop_with_value and has_value:
            i += 2
            continue
        if tok in _MODEL and has_value and local_files:
            i += 2  # already emitted as the first -m
            continue
        if tok in replace and has_value:
            out += [tok, replace[tok]]
            i += 2
            continue
        if tok == "--host" and host and has_value:
            out += [tok, host]
            i += 2
            continue
        if tok == "--port" and port and has_value:
            out += [tok, str(port)]
            i += 2
            continue
        out.append(tok)
        i += 1

    values = _option_values(out)
    if host and "--host" not in values:
        out += ["--host", host]
    if port and "--port" not in values:
        out += ["--port", str(port)]
    values = _option_values(out)
    return LaunchPlan(
        argv=out,
        host=values.get("--host", "127.0.0.1"),
        port=int(values.get("--port", recipe["serve"].get("port") or 8080)),
    )


def _http_json(url: str, payload: dict | None = None, timeout: float = 5) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read() or b"{}")


def server_is_healthy(base_url: str) -> bool:
    try:
        return _http_json(f"{base_url}/health").get("status") == "ok"
    except (urllib.error.URLError, OSError, ValueError):
        return False


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def start_server(
    plan: LaunchPlan, log_file: Path | str, timeout_s: int = 1800, poll_s: float = 5
) -> subprocess.Popen | None:
    """Start llama-server in the background and wait until ``/health`` is ok.

    Returns ``None`` (and does nothing) if a healthy server is already
    listening on the configured host/port.
    """
    if server_is_healthy(plan.base_url):
        _print_step("skip", f"llama-server already running at {plan.base_url}")
        return None
    if port_in_use(plan.host, plan.port):
        raise RuntimeError(
            f"Port {plan.port} is in use but /health is not ok. A server may still be loading, "
            "or another program owns the port. Wait and re-run, or choose a different PORT."
        )

    log_file = Path(log_file).expanduser()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0
    with open(log_file, "ab") as log:
        proc = subprocess.Popen(
            plan.argv, stdout=log, stderr=subprocess.STDOUT,
            cwd=str(Path(plan.argv[0]).parent), creationflags=flags,
        )
    _print_step("run", f"Started llama-server (pid {proc.pid}); logging to {log_file}")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {proc.returncode}; see {log_file}")
        if server_is_healthy(plan.base_url):
            _print_step("ok", f"llama-server is ready at {plan.base_url}")
            return proc
        time.sleep(poll_s)
    raise TimeoutError(f"llama-server did not become healthy within {timeout_s}s; see {log_file}")


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        _print_step("skip", "No server process started from this notebook is running")
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    _print_step("ok", f"Stopped llama-server (pid {proc.pid})")


def chat(base_url: str, prompt: str, model: str | None = None, timeout: float = 600) -> str:
    """Send one chat completion request to the OpenAI-compatible endpoint."""
    if model is None:
        models = _http_json(f"{base_url}/v1/models").get("data") or [{}]
        model = models[0].get("id", "local")
    resp = _http_json(
        f"{base_url}/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=timeout,
    )
    return resp["choices"][0]["message"]["content"]
