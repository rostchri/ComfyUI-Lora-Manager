"""aimodelinventory integration (fork branch ``aimodelinventory``).

Models live once in a shared store (``aimodelinventory/blobs/...``) and an app
only sees a view of relative links::

    <view>/loras/x.safetensors  ->  ../../../aimodelinventory/models/lora/civitai/<mid>/<vid>/x.safetensors
                                ->  ../../../../../../blobs/sha256/<ab>/<sha256>

The store sits behind a FUSE mount. Reading a model there pulls the whole blob
over the network and through the shared VFS cache, so this module keeps the
LoRA Manager away from model bytes:

* the SHA256 comes from the link chain (blob name, or the store manifest for
  legacy ``blobs/sha1`` blobs), never from reading the file;
* header reads (AutoV3, quantization, safetensors metadata) are skipped unless
  ``LORA_MANAGER_AIMI_HEADER_READS=1``;
* ``ro`` refuses every mutation of the governed roots (ComfyUI custom node);
* ``rw`` (standalone only) mutates links instead of bytes: downloads are staged
  outside the mount, verified and booked into the store, moves/renames re-link,
  deletes remove the app link only. Blobs are never deleted (gc runs on the
  store authority).

Configuration (environment only; unset mode = upstream behaviour):

``LORA_MANAGER_AIMI_MODE``          off | ro | rw
``LORA_MANAGER_AIMI_STORE``         container path of the bucket's ``aimodelinventory/`` prefix
``LORA_MANAGER_AIMI_VIEWS``         ``<container path>=<bucket key>[;...]``, e.g.
                                    ``/app/ComfyUI/models=app/comfyui``
``LORA_MANAGER_AIMI_RW_DIRS``       view subfolders writable in rw mode (default ``loras,checkpoints``)
``LORA_MANAGER_AIMI_TYPE_MAP``      view subfolder -> store type (default ``loras=lora,checkpoints=diffusion``)
``LORA_MANAGER_AIMI_HEADER_READS``  1 = allow header reads on governed files (default 0)
``LORA_MANAGER_AIMI_BUCKET_MOUNT``  (rw) container path of the whole bucket mount
``LORA_MANAGER_AIMI_CLI``           (rw) path of the ``aimodelinventory`` CLI script
``LORA_MANAGER_AIMI_STAGING``       (rw) download staging directory, outside the mount
``LORA_MANAGER_AIMI_BLOB_UPLOAD``   (rw) auto (default: mount when the staging disk has room
                                    for a second copy, else rclone) | mount (write through the
                                    mount, keeps the VFS cache warm) | rclone (``rclone copyto``)
``LORA_MANAGER_AIMI_UPLOAD_WAIT``   (rw) seconds to wait for a mount upload (default 7200)
``LORA_MANAGER_AIMI_FREE_RESERVE_GB`` (rw) free space kept on the staging disk (default 10)
``LORA_MANAGER_AIMI_SPACE_WAIT``    (rw) seconds a download waits for that space before it
                                    starts anyway (default 21600); downloads are never aborted
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.machinery
import json
import logging
import os
import posixpath
import shutil
import socket
import sys
import threading
import time
import types
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

MODE_OFF = "off"
MODE_RO = "ro"
MODE_RW = "rw"

MANIFEST_NAME = ".aimodelinventory.json"
_MAX_HOPS = 8
_HEX = frozenset("0123456789abcdef")


class AimiError(Exception):
    """An operation that aimodelinventory conventions do not allow."""


class AimiReadOnlyError(AimiError, PermissionError):
    """A mutation of a governed root in read-only mode."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def mode() -> str:
    value = _env("LORA_MANAGER_AIMI_MODE", MODE_OFF).lower()
    return value if value in (MODE_RO, MODE_RW) else MODE_OFF


def active() -> bool:
    return mode() != MODE_OFF and bool(store_root())


def store_root() -> str:
    value = _env("LORA_MANAGER_AIMI_STORE")
    return os.path.abspath(value) if value else ""


def _norm(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def views() -> List[Tuple[str, str]]:
    """(container root, bucket key) pairs, longest container root first."""
    pairs: List[Tuple[str, str]] = []
    for item in _env("LORA_MANAGER_AIMI_VIEWS").split(";"):
        if "=" not in item:
            continue
        root, key = item.split("=", 1)
        root, key = root.strip(), key.strip().strip("/")
        if root and key:
            pairs.append((_norm(root), key))
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


def _list_env(name: str, default: str) -> List[str]:
    return [v.strip() for v in _env(name, default).split(",") if v.strip()]


def rw_dirs() -> List[str]:
    return _list_env("LORA_MANAGER_AIMI_RW_DIRS", "loras,checkpoints")


def type_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in _list_env("LORA_MANAGER_AIMI_TYPE_MAP", "loras=lora,checkpoints=diffusion"):
        if "=" in item:
            sub, typ = item.split("=", 1)
            mapping[sub.strip()] = typ.strip()
    return mapping


def header_reads_allowed() -> bool:
    return _env("LORA_MANAGER_AIMI_HEADER_READS") == "1"


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def _roots() -> List[str]:
    roots = [root for root, _key in views()]
    store = store_root()
    if store:
        roots.append(_norm(store))
    return roots


def governs(path: Optional[str]) -> bool:
    """True when *path* lies in an aimodelinventory view or in the store."""
    if not path or not active():
        return False
    p = _norm(path)
    return any(_under(p, root) for root in _roots())


def skip_header_read(path: Optional[str]) -> bool:
    """True when reading the file header would pull bytes through the mount."""
    return governs(path) and not header_reads_allowed()


def view_subdir(path: str) -> Optional[str]:
    """First folder below the view root (``loras``, ``checkpoints``, ...)."""
    p = _norm(path)
    for root, _key in views():
        if _under(p, root) and p != root:
            return os.path.relpath(p, root).split(os.sep)[0]
    return None


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def guard_write(path: Optional[str], action: str) -> None:
    """Refuse a mutation of a governed path that the current mode forbids.

    ``ro``: every mutation inside a view or the store is refused.
    ``rw``: only view subfolders listed in ``LORA_MANAGER_AIMI_RW_DIRS``; the
    store itself is written only by this module.
    """
    if not governs(path):
        return
    if mode() == MODE_RO:
        raise AimiReadOnlyError(
            f"aimodelinventory read-only: {action} not allowed here ({path}); "
            "use the standalone LoRA Manager"
        )
    p = _norm(path)
    if _under(p, _norm(store_root())):
        raise AimiReadOnlyError(f"aimodelinventory: {action} inside the store is not allowed ({path})")
    sub = view_subdir(p)
    if sub not in rw_dirs():
        raise AimiReadOnlyError(
            f"aimodelinventory: {action} not allowed for '{sub or path}' "
            f"(writable: {', '.join(rw_dirs()) or 'none'})"
        )


def guard_move(source: Optional[str], target: Optional[str], action: str = "move") -> None:
    """Guard both ends; a model never crosses the view boundary (a view link
    outside the view would dangle, a real file inside it would bypass the store)."""
    guard_write(source, action)
    guard_write(target, action)
    if governs(source) != governs(target):
        raise AimiError(f"aimodelinventory: {action} across the view boundary ({source} -> {target})")


def manages_links(path: Optional[str]) -> bool:
    """True when *path* is governed and this process may re-link it (rw)."""
    return mode() == MODE_RW and governs(path)


def skip_root_maintenance(root: Optional[str]) -> bool:
    """True for roots that startup sweeps (.bak cleanup, pending-delete
    reconcile) must not walk: they live in the mount."""
    return governs(root)


# ---------------------------------------------------------------------------
# Link chain and identity
# ---------------------------------------------------------------------------


def chain(path: str) -> List[Tuple[str, str]]:
    """(link, resolved target) hops from *path* to the final object.

    Only ``readlink`` is used: nothing is opened or read.
    """
    hops: List[Tuple[str, str]] = []
    cur = os.path.abspath(path)
    for _ in range(_MAX_HOPS):
        try:
            target = os.readlink(cur)
        except OSError:
            break
        nxt = os.path.normpath(os.path.join(os.path.dirname(cur), target))
        hops.append((cur, nxt))
        cur = nxt
    return hops


def _blob_namespace(path: str) -> Optional[Tuple[str, str]]:
    """('sha256'|'sha1', name) for a store blob path, else None."""
    store = store_root()
    if not store:
        return None
    p = _norm(path)
    blobs = os.path.join(_norm(store), "blobs")
    if not _under(p, blobs):
        return None
    parts = os.path.relpath(p, blobs).split(os.sep)
    if len(parts) != 3:
        return None
    namespace, bucket, name = parts
    name = name.lower()
    if namespace not in ("sha256", "sha1") or bucket != name[:2] or not set(name) <= _HEX:
        return None
    if len(name) != (64 if namespace == "sha256" else 40):
        return None
    return namespace, name


_manifest_lock = threading.Lock()
_manifest_cache: Dict[str, Tuple[float, Optional[dict]]] = {}
_repo_dir_cache: Dict[str, Optional[str]] = {}


def _repo_dir_of(model_link: str) -> Optional[str]:
    """Repo folder (holding the manifest) of a ``models/...`` link."""
    models = os.path.join(_norm(store_root()), "models")
    d = os.path.dirname(_norm(model_link))
    probed: List[str] = []
    found: Optional[str] = None
    while _under(d, models) and d != models:
        if d in _repo_dir_cache:
            found = _repo_dir_cache[d]
            break
        probed.append(d)
        if os.path.isfile(os.path.join(d, MANIFEST_NAME)):
            found = d
            break
        d = os.path.dirname(d)
    for item in probed:
        _repo_dir_cache[item] = found
    return found


def _load_manifest(repo_dir: str) -> Optional[dict]:
    path = os.path.join(repo_dir, MANIFEST_NAME)
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        return None
    with _manifest_lock:
        cached = _manifest_cache.get(repo_dir)
        if cached and cached[0] == mtime:
            return cached[1]
    try:
        # A manifest is a small real file, never a link into the blob store.
        if os.path.islink(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        data = None
    with _manifest_lock:
        _manifest_cache[repo_dir] = (mtime, data)
    return data


def manifest_entry(model_link: str) -> Tuple[Optional[str], Optional[dict], Optional[dict]]:
    """(repo dir, manifest, file entry) for a ``models/...`` link."""
    repo_dir = _repo_dir_of(model_link)
    if not repo_dir:
        return None, None, None
    manifest = _load_manifest(repo_dir)
    if not manifest:
        return repo_dir, None, None
    rel = os.path.relpath(_norm(model_link), repo_dir).replace(os.sep, "/")
    entry = (manifest.get("files") or {}).get(rel)
    return repo_dir, manifest, entry if isinstance(entry, dict) else None


def _valid_sha256(value: Any) -> Optional[str]:
    if isinstance(value, str):
        value = value.strip().lower()
        if len(value) == 64 and set(value) <= _HEX:
            return value
    return None


def sha256_without_reading(path: str) -> str:
    """SHA256 of a governed model, derived from its link chain.

    ``blobs/sha256/<ab>/<sha>`` names the hash. A legacy ``blobs/sha1`` blob
    takes ``sha256`` (or ``sha256_claimed``) from the manifest of the
    ``models/`` link that points at it. Returns ``""`` when no identity can be
    derived without reading bytes (real files in the view, unbooked blobs):
    the caller then treats the file as having no hash.
    """
    hops = chain(path)
    final = hops[-1][1] if hops else _norm(path)
    blob = _blob_namespace(final)
    if blob and blob[0] == "sha256":
        return blob[1]
    if blob and blob[0] == "sha1":
        models = os.path.join(_norm(store_root()), "models")
        for link, _target in reversed(hops):
            if _under(_norm(link), models):
                _dir, _man, entry = manifest_entry(link)
                if entry:
                    found = _valid_sha256(entry.get("sha256")) or _valid_sha256(entry.get("sha256_claimed"))
                    if found:
                        return found
                break
    logger.debug("aimodelinventory: no identity without reading for %s", path)
    return ""


# ---------------------------------------------------------------------------
# Bucket keys
# ---------------------------------------------------------------------------


def to_key(path: str) -> Optional[str]:
    """Bucket key of a container path (views and store), else None."""
    p = _norm(path)
    pairs = list(views())
    store = store_root()
    if store:
        pairs.append((_norm(store), "aimodelinventory"))
    for root, key in sorted(pairs, key=lambda x: len(x[0]), reverse=True):
        if _under(p, root):
            rel = os.path.relpath(p, root)
            return key if rel == os.curdir else posixpath.join(key, rel.replace(os.sep, "/"))
    return None


def relative_link(link_key: str, target_key: str) -> str:
    return posixpath.relpath(target_key, posixpath.dirname(link_key))


def _link_text(link_path: str, target_path: str) -> str:
    """Relative link text, computed in bucket-key space when both sides map."""
    link_key, target_key = to_key(link_path), to_key(target_path)
    if link_key and target_key:
        return relative_link(link_key, target_key)
    return os.path.relpath(_norm(target_path), os.path.dirname(_norm(link_path)))


# ---------------------------------------------------------------------------
# rw: manifest bookkeeping via the aimodelinventory CLI
# ---------------------------------------------------------------------------

_cli_lock = threading.Lock()
_cli_state: Dict[str, Any] = {}
_book_lock = threading.Lock()


def _cli() -> Tuple[Any, Any]:
    """(module, Config) of the aimodelinventory CLI, loaded once."""
    with _cli_lock:
        if "module" not in _cli_state:
            path = _env("LORA_MANAGER_AIMI_CLI")
            mount = _env("LORA_MANAGER_AIMI_BUCKET_MOUNT")
            if not path or not mount:
                raise AimiError(
                    "aimodelinventory rw needs LORA_MANAGER_AIMI_CLI and LORA_MANAGER_AIMI_BUCKET_MOUNT"
                )
            os.environ.setdefault("AIMI_HASH", "none")
            os.environ["AIMI_MOUNT"] = mount
            loader = importlib.machinery.SourceFileLoader("aimodelinventory_cli", path)
            module = types.ModuleType(loader.name)
            module.__file__ = path
            # The CLI defines dataclasses; @dataclass resolves annotations
            # through sys.modules[cls.__module__], so the module must be
            # registered before it executes.
            sys.modules[loader.name] = module
            try:
                loader.exec_module(module)
            except BaseException:
                sys.modules.pop(loader.name, None)
                raise
            _cli_state["module"] = module
        module = _cli_state["module"]
        cfg = module.Config()
        cfg.mount = module.Path(_env("LORA_MANAGER_AIMI_BUCKET_MOUNT"))
        cfg.hash_mode = "none"
        return module, cfg


def _bucket_path(key: str) -> str:
    return os.path.join(_env("LORA_MANAGER_AIMI_BUCKET_MOUNT"), key)


def _update_app_paths(model_link: str, remove: Iterable[str] = (), add: Iterable[str] = ()) -> None:
    """Keep ``app_paths`` (and ``app_links``) of the store manifest in sync."""
    remove, add = [k for k in remove if k], [k for k in add if k]
    if not remove and not add:
        return
    repo_dir = _repo_dir_of(model_link)
    if not repo_dir:
        logger.warning("aimodelinventory: no manifest for %s; app_paths not updated", model_link)
        return
    module, cfg = _cli()
    dir_key = to_key(repo_dir)
    rel = os.path.relpath(_norm(model_link), repo_dir).replace(os.sep, "/")
    with _book_lock:
        text = module.read_remote_text(cfg, f"{dir_key}/{MANIFEST_NAME}")
        if not text:
            logger.warning("aimodelinventory: manifest %s unreadable; app_paths not updated", dir_key)
            return
        manifest = json.loads(text)
        entry = (manifest.get("files") or {}).get(rel)
        if not isinstance(entry, dict):
            logger.warning("aimodelinventory: %s not in manifest %s", rel, dir_key)
            return
        paths = [p for p in entry.get("app_paths") or [] if p not in remove]
        paths.extend(k for k in add if k not in paths)
        entry["app_paths"] = sorted(paths)
        _sync_app_links(manifest, add)
        manifest["updated_at"] = module._now()
        module.save_manifest(cfg, manifest, dir_key=dir_key)
    with _manifest_lock:
        _manifest_cache.pop(repo_dir, None)


def _sync_app_links(manifest: dict, app_keys: Iterable[str]) -> None:
    links = list(manifest.get("app_links") or [])
    for key in app_keys:
        for _root, view_key in views():
            if (key == view_key or key.startswith(view_key + "/")) and view_key not in links:
                links.append(view_key)
    manifest["app_links"] = sorted(links)


def _first_model_target(link_path: str) -> Optional[str]:
    """The ``models/...`` link an app link (or alias chain) leads to."""
    models = os.path.join(_norm(store_root()), "models")
    for _link, target in chain(link_path):
        if _under(_norm(target), models):
            return target
    return None


def relink(old_path: str, new_path: str) -> None:
    """Move/rename an app link: new relative link, old one removed, manifest
    ``app_paths`` updated. Bytes are never touched."""
    if not os.path.islink(old_path):
        raise AimiError(
            f"aimodelinventory: {old_path} is not a link; real model files in a view "
            "are not moved (migrate them into the store first)"
        )
    target = os.path.normpath(os.path.join(os.path.dirname(old_path), os.readlink(old_path)))
    text = _link_text(new_path, target)
    if os.path.lexists(new_path):
        raise AimiError(f"aimodelinventory: target exists: {new_path}")
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    os.symlink(text, new_path)
    os.unlink(old_path)
    model = _first_model_target(new_path)
    if model:
        _update_app_paths(model, remove=[to_key(old_path) or ""], add=[to_key(new_path) or ""])


def unlink_view_entry(path: str) -> None:
    """Delete a model from the view: the app link only; the store keeps the blob."""
    if not os.path.islink(path):
        raise AimiError(
            f"aimodelinventory: {path} is not a link; real files in a view are not deleted here"
        )
    model = _first_model_target(path)
    os.unlink(path)
    if model:
        _update_app_paths(model, remove=[to_key(path) or ""])


def after_folder_rename(old_dir: str, new_dir: str) -> None:
    """Re-key ``app_paths`` after a folder rename (links keep their depth, so
    their relative targets stay valid)."""
    old_key, new_key = to_key(old_dir), to_key(new_dir)
    if not old_key or not new_key:
        return
    for dirpath, _dirnames, filenames in os.walk(new_dir):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if not os.path.islink(path):
                continue
            model = _first_model_target(path)
            if not model:
                continue
            new_link_key = to_key(path) or ""
            old_link_key = old_key + new_link_key[len(new_key):]
            try:
                _update_app_paths(model, remove=[old_link_key], add=[new_link_key])
            except Exception as exc:  # keep re-keying the rest
                logger.warning("aimodelinventory: app_paths for %s: %s", path, exc)


# ---------------------------------------------------------------------------
# rw: downloads
# ---------------------------------------------------------------------------


@dataclass
class DownloadPlan:
    """What a governed download books into the store."""

    save_path: str
    store_type: str
    model_id: int
    version_id: int
    file_name: str
    expected_sha256: Optional[str]
    info: Dict[str, Any] = field(default_factory=dict)
    expected_size: int = 0

    @property
    def repo(self) -> str:
        return f"{self.model_id}/{self.version_id}"

    @property
    def dir_key(self) -> str:
        return f"aimodelinventory/models/{self.store_type}/civitai/{self.model_id}/{self.version_id}"

    @property
    def staging_path(self) -> str:
        digest = hashlib.sha1(self.save_path.encode()).hexdigest()[:16]
        return os.path.join(staging_dir(), digest, self.file_name)


def staging_dir() -> str:
    value = _env("LORA_MANAGER_AIMI_STAGING")
    if not value:
        raise AimiError("aimodelinventory rw needs LORA_MANAGER_AIMI_STAGING")
    staging = _norm(value)
    if any(_under(staging, root) for root in _roots()) or _under(
        staging, _norm(_env("LORA_MANAGER_AIMI_BUCKET_MOUNT") or "/nonexistent")
    ):
        raise AimiError("aimodelinventory: staging must not lie inside the mount")
    return staging


def _primary_file(version_info: Dict[str, Any], save_path: str) -> Dict[str, Any]:
    files = [f for f in version_info.get("files") or [] if isinstance(f, dict)]
    for f in files:
        if f.get("primary"):
            return f
    base = os.path.basename(save_path)
    for f in files:
        if f.get("name") == base:
            return f
    return files[0] if files else {}


def plan_download(save_path: str, version_info: Dict[str, Any], sha256: Optional[str]) -> DownloadPlan:
    """Decide how a download into a governed view is booked."""
    guard_write(save_path, "download")
    sub = view_subdir(save_path)
    store_type = type_map().get(sub or "")
    if not store_type:
        raise AimiError(f"aimodelinventory: no store type for view folder '{sub}'")
    model_id = version_info.get("modelId") or (version_info.get("model") or {}).get("id")
    version_id = version_info.get("id")
    if not model_id or not version_id:
        raise AimiError("aimodelinventory: only CivitAI downloads with model and version id are supported")
    primary = _primary_file(version_info, save_path)
    file_name = os.path.basename(primary.get("name") or save_path)
    expected = _valid_sha256((primary.get("hashes") or {}).get("SHA256")) or _valid_sha256(sha256)
    model = version_info.get("model") or {}
    info = {
        "civitai_url": f"https://civitai.com/models/{model_id}?modelVersionId={version_id}",
        "civitai_model": model.get("name"),
        "civitai_version": version_info.get("name"),
        "civitai_type": model.get("type"),
        "base_model": version_info.get("baseModel"),
    }
    return DownloadPlan(
        save_path=_norm(save_path),
        store_type=store_type,
        model_id=int(model_id),
        version_id=int(version_id),
        file_name=file_name,
        expected_sha256=expected,
        info={k: v for k, v in info.items() if v},
        expected_size=int(float(primary.get("sizeKB") or 0) * 1024),
    )


def _reserve_bytes() -> int:
    return int(float(_env("LORA_MANAGER_AIMI_FREE_RESERVE_GB", "10")) * (1 << 30))


def _free_bytes(path: str) -> int:
    return shutil.disk_usage(path).free


def has_room(size: int, copies: int = 1) -> bool:
    """True when the staging disk holds *copies* more files of *size* bytes
    and still keeps the reserve."""
    return _free_bytes(staging_dir()) >= copies * size + _reserve_bytes()


async def wait_for_room(plan: DownloadPlan, poll: float = 30.0) -> None:
    """Pause a download until the staging disk has room for it.

    The shared VFS cache frees space by evicting old blobs, so waiting is
    usually enough. Never aborts: after LORA_MANAGER_AIMI_SPACE_WAIT seconds
    the download starts anyway.
    """
    if not plan.expected_size or has_room(plan.expected_size):
        return
    deadline = time.monotonic() + float(_env("LORA_MANAGER_AIMI_SPACE_WAIT", "21600"))
    logger.warning(
        "aimodelinventory: waiting for %.1f GiB free on %s before downloading %s",
        (plan.expected_size + _reserve_bytes()) / (1 << 30),
        staging_dir(),
        plan.file_name,
    )
    while not has_room(plan.expected_size):
        if time.monotonic() > deadline:
            logger.warning("aimodelinventory: still short of space, downloading %s anyway", plan.file_name)
            return
        await asyncio.sleep(poll)


def _sha256_local(path: str) -> str:
    """Hash of a staging file (local disk, never the mount)."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _place_blob(module: Any, cfg: Any, staging: str, sha256: str) -> None:
    key = cfg.blob_key(sha256)
    size = os.path.getsize(staging)
    existing = module.remote_stat(cfg, key)
    if existing is not None:
        if existing["size"] != size:
            raise AimiError(f"aimodelinventory: blob collision {key}: {existing['size']} != {size}")
        return
    method = _env("LORA_MANAGER_AIMI_BLOB_UPLOAD", "auto")
    # The mount keeps a second copy in the VFS cache until the upload is done.
    if method == "rclone" or (method == "auto" and not has_room(size)):
        module.ensure_blob(cfg, module.Path(staging), sha256)
        return
    # Through the mount: rclone uploads it and keeps the bytes in the VFS
    # cache, so ComfyUI loads the new model without fetching it back.
    final = _bucket_path(key)
    partial = final + ".partial"
    os.makedirs(os.path.dirname(final), exist_ok=True)
    shutil.copyfile(staging, partial)
    os.replace(partial, final)
    deadline = time.monotonic() + float(_env("LORA_MANAGER_AIMI_UPLOAD_WAIT", "7200"))
    while True:
        st = module.remote_stat(cfg, key)
        if st is not None and st["size"] == size:
            return
        if time.monotonic() > deadline:
            raise AimiError(f"aimodelinventory: upload of {key} not confirmed in time; staging kept")
        time.sleep(10)


def publish_download(plan: DownloadPlan, staging: str) -> Dict[str, Any]:
    """Book a finished staging download: blob, repo link, manifest, app link.

    Runs blocking I/O; call it in a worker thread.
    """
    module, cfg = _cli()
    actual = _sha256_local(staging)
    if plan.expected_sha256 and actual != plan.expected_sha256:
        os.remove(staging)
        raise AimiError(
            f"aimodelinventory: SHA256 mismatch for {plan.file_name}: "
            f"download {actual[:12]}… != CivitAI {plan.expected_sha256[:12]}…"
        )
    verified = bool(plan.expected_sha256)
    size = os.path.getsize(staging)
    _place_blob(module, cfg, staging, actual)

    blob_key = cfg.blob_key(actual)
    model_key = f"{plan.dir_key}/{plan.file_name}"
    app_key = to_key(plan.save_path)
    if not app_key:
        raise AimiError(f"aimodelinventory: {plan.save_path} is not in a view")

    with _book_lock:
        model_path = _bucket_path(model_key)
        if os.path.lexists(model_path):
            current_key = posixpath.normpath(
                posixpath.join(posixpath.dirname(model_key), os.readlink(model_path))
            )
            if current_key != blob_key:
                # Legacy sha1 blob of the same file: accept when the manifest
                # vouches for this sha256; anything else is a conflict.
                known = module.load_manifest(cfg, plan.repo, dir_key=plan.dir_key) or {}
                old = (known.get("files") or {}).get(plan.file_name) or {}
                claimed = _valid_sha256(old.get("sha256")) or _valid_sha256(old.get("sha256_claimed"))
                if claimed != actual:
                    raise AimiError(f"aimodelinventory: {model_key} points elsewhere ({current_key})")
                blob_key = current_key
        else:
            module.make_symlink(cfg, model_key, relative_link(model_key, blob_key))

        manifest = module.load_manifest(cfg, plan.repo, dir_key=plan.dir_key)
        if not manifest:
            manifest = {
                "repo_id": plan.repo,
                "revision": str(plan.version_id),
                "source": "loramanager-download",
                "app_links": [],
                "files": {},
            }
        manifest.update(plan.info)
        entry = (manifest.setdefault("files", {})).get(plan.file_name) or {}
        paths = sorted(set(entry.get("app_paths") or []) | {app_key})
        entry.update(
            {
                "size": size,
                "sha256": actual,
                "verified": verified,
                "identity": (
                    "Download-sha256 == CivitAI-SHA256 (loramanager, "
                    f"{socket.gethostname()})"
                    if verified
                    else f"Download-sha256, keine CivitAI-Angabe (loramanager, {socket.gethostname()})"
                ),
                "blob": blob_key,
                "model_link": model_key,
                "app_paths": paths,
                "sha1_b2": entry.get("sha1_b2"),
            }
        )
        if not verified:
            entry["sha256_claimed"] = actual
        manifest["files"][plan.file_name] = entry
        _sync_app_links(manifest, [app_key])
        manifest["updated_at"] = module._now()
        module.save_manifest(cfg, manifest, dir_key=plan.dir_key)

        if os.path.lexists(plan.save_path):
            raise AimiError(f"aimodelinventory: {plan.save_path} appeared during the download")
        module.make_symlink(cfg, app_key, relative_link(app_key, model_key))

    os.remove(staging)
    try:
        os.rmdir(os.path.dirname(staging))
    except OSError:
        pass
    with _manifest_lock:
        _manifest_cache.clear()
        _repo_dir_cache.clear()
    logger.info("aimodelinventory: booked %s -> %s (%s)", app_key, model_key, blob_key)
    return {"sha256": actual, "blob": blob_key, "model_link": model_key, "app_path": app_key}
