from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from .models import Backend, CodeBundle


ALLOWED_FILES = {
    Backend.AMD_XDNA2: {"design.py", "kernel.cc"},
    Backend.INTEL_OPENVINO: {"model.py"},
}


def bundle_hash(bundle: CodeBundle) -> str:
    payload = bundle.model_dump(mode="json")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def store_artifact(path: Path, store_root: Path) -> tuple[Path, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    destination = store_root / digest[:2] / digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(path, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination, digest


def materialize_bundle(bundle: CodeBundle, destination: Path, max_bytes: int = 2_000_000) -> dict[str, str]:
    allowed = ALLOWED_FILES[bundle.backend]
    names = [item.relative_path for item in bundle.files]
    if len(names) != len(set(names)):
        raise ValueError("generated bundle contains duplicate paths")
    unexpected = set(names) - allowed
    if unexpected:
        raise ValueError(f"unexpected generated files for {bundle.backend.value}: {sorted(unexpected)}")
    required = {"design.py", "kernel.cc"} if bundle.backend == Backend.AMD_XDNA2 else {"model.py"}
    missing = required - set(names)
    if missing and not bundle.unsupported_operations:
        raise ValueError(f"generated bundle is missing required files: {sorted(missing)}")
    total = sum(len(item.content.encode("utf-8")) for item in bundle.files)
    if total > max_bytes:
        raise ValueError("generated bundle exceeds total size limit")
    destination.mkdir(parents=True, exist_ok=True)
    resolved_root = destination.resolve()
    result: dict[str, str] = {}
    for item in bundle.files:
        output = destination / item.relative_path
        if output.exists() and output.is_symlink():
            raise ValueError(f"refusing to replace symlink: {item.relative_path}")
        if output.resolve().parent != resolved_root:
            raise ValueError(f"generated path escapes destination: {item.relative_path}")
        output.write_text(item.content, encoding="utf-8")
        result[item.relative_path] = hashlib.sha256(item.content.encode()).hexdigest()
    return result
