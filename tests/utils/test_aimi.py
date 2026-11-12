"""aimodelinventory integration (py/utils/aimi.py).

The bucket is a tmp directory tree with real symlinks, the way ``rclone mount
--vfs-links`` presents it; the aimodelinventory CLI is replaced by a small fake
module that works on the same tree. Nothing here reaches a remote.
"""

from __future__ import annotations

import hashlib
import json
import os
import textwrap
from pathlib import Path

import pytest

from py.utils import aimi
from py.utils import file_utils

FAKE_CLI = textwrap.dedent(
    '''
    from __future__ import annotations

    import json, os, shutil
    from dataclasses import dataclass
    from pathlib import Path

    CALLS = []

    # A dataclass with string annotations like the real CLI: the decorator
    # looks the defining module up in sys.modules, so loading the CLI
    # without registering it fails.
    @dataclass
    class Config:
        mount: object = None
        hash_mode: str = "none"
        dry_run: bool = False

        def blob_key(self, sha256):
            return f"aimodelinventory/blobs/sha256/{sha256[:2]}/{sha256}"

    def _p(cfg, key):
        return Path(cfg.mount) / key

    def remote_stat(cfg, key):
        p = _p(cfg, key)
        if not p.is_file():
            return None
        return {"size": p.stat().st_size, "sha1": None}

    def make_symlink(cfg, key, rel):
        CALLS.append(("make_symlink", key, rel))
        p = _p(cfg, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_symlink() or p.exists():
            p.unlink()
        os.symlink(rel, p)
        return "mount"

    def read_remote_text(cfg, key):
        p = _p(cfg, key)
        return p.read_text() if p.is_file() and not p.is_symlink() else None

    def load_manifest(cfg, repo, dir_key=None):
        text = read_remote_text(cfg, f"{dir_key}/.aimodelinventory.json")
        if not text:
            return None
        man = json.loads(text)
        man["_dir_key"] = dir_key
        return man

    def save_manifest(cfg, manifest, dir_key=None):
        CALLS.append(("save_manifest", dir_key))
        clean = {k: v for k, v in manifest.items() if not k.startswith("_")}
        clean["model_dir"] = dir_key
        p = _p(cfg, f"{dir_key}/.aimodelinventory.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\\n")

    def ensure_blob(cfg, local_path, sha256):
        CALLS.append(("ensure_blob", sha256))
        dst = _p(cfg, cfg.blob_key(sha256))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, dst)

    def _now():
        return "2026-11-12T00:00:00Z"
    '''
)

LORA_DIR = "aimodelinventory/models/lora/civitai/100/200"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _link(bucket: Path, key: str, target_key: str) -> Path:
    path = bucket / key
    path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(aimi.relative_link(key, target_key), path)
    return path


def _blob(bucket: Path, namespace: str, name: str, data: bytes) -> str:
    key = f"aimodelinventory/blobs/{namespace}/{name[:2]}/{name}"
    path = bucket / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return key


def _manifest(bucket: Path, dir_key: str, files: dict) -> None:
    path = bucket / dir_key / aimi.MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"repo_id": "100/200", "files": files}))


def _read_manifest(bucket: Path, dir_key: str) -> dict:
    return json.loads((bucket / dir_key / aimi.MANIFEST_NAME).read_text())


@pytest.fixture
def bucket(tmp_path, monkeypatch):
    """Bucket tree: the view and the store are both mapped at their bucket
    keys, so relative link texts resolve exactly as in the bucket."""
    root = tmp_path / "bucket"
    (root / "app/comfyui/loras").mkdir(parents=True)
    (root / "app/comfyui/checkpoints").mkdir(parents=True)
    (root / "app/comfyui/vae").mkdir(parents=True)
    (root / "aimodelinventory/models").mkdir(parents=True)
    cli = tmp_path / "cli" / "aimodelinventory"
    cli.parent.mkdir()
    cli.write_text(FAKE_CLI)
    staging = tmp_path / "staging"
    staging.mkdir()

    monkeypatch.setenv("LORA_MANAGER_AIMI_MODE", "rw")
    monkeypatch.setenv("LORA_MANAGER_AIMI_STORE", str(root / "aimodelinventory"))
    monkeypatch.setenv("LORA_MANAGER_AIMI_VIEWS", f"{root / 'app/comfyui'}=app/comfyui")
    monkeypatch.setenv("LORA_MANAGER_AIMI_BUCKET_MOUNT", str(root))
    monkeypatch.setenv("LORA_MANAGER_AIMI_CLI", str(cli))
    monkeypatch.setenv("LORA_MANAGER_AIMI_STAGING", str(staging))
    monkeypatch.setenv("LORA_MANAGER_AIMI_UPLOAD_WAIT", "5")
    monkeypatch.setenv("LORA_MANAGER_AIMI_FREE_RESERVE_GB", "0")
    monkeypatch.delenv("LORA_MANAGER_AIMI_SPACE_WAIT", raising=False)
    for name in ("RW_DIRS", "TYPE_MAP", "HEADER_READS", "BLOB_UPLOAD"):
        monkeypatch.delenv(f"LORA_MANAGER_AIMI_{name}", raising=False)
    monkeypatch.setenv("AIMI_NO_REMOTE", "1")
    # _cli() exports these for the real CLI; keep them out of other tests.
    monkeypatch.delenv("AIMI_MOUNT", raising=False)
    monkeypatch.delenv("AIMI_HASH", raising=False)

    aimi._cli_state.clear()
    aimi._manifest_cache.clear()
    aimi._repo_dir_cache.clear()
    yield root
    aimi._cli_state.clear()
    aimi._manifest_cache.clear()
    aimi._repo_dir_cache.clear()


@pytest.fixture
def booked_lora(bucket):
    """A LoRA booked the regular way: app link -> repo link -> sha256 blob."""
    data = b"lora-bytes"
    sha = _sha(data)
    blob = _blob(bucket, "sha256", sha, data)
    model_key = f"{LORA_DIR}/x.safetensors"
    _link(bucket, model_key, blob)
    app_key = "app/comfyui/loras/x.safetensors"
    app = _link(bucket, app_key, model_key)
    _manifest(
        bucket,
        LORA_DIR,
        {"x.safetensors": {"sha256": sha, "blob": blob, "app_paths": [app_key]}},
    )
    return app, sha


# --- configuration and guards -------------------------------------------------


def test_off_mode_governs_nothing(bucket, monkeypatch):
    monkeypatch.setenv("LORA_MANAGER_AIMI_MODE", "off")
    assert not aimi.active()
    assert not aimi.governs(str(bucket / "app/comfyui/loras/x.safetensors"))
    aimi.guard_write(str(bucket / "app/comfyui/loras/x.safetensors"), "delete")


def test_unknown_mode_is_off(bucket, monkeypatch):
    monkeypatch.setenv("LORA_MANAGER_AIMI_MODE", "yes")
    assert aimi.mode() == aimi.MODE_OFF


def test_governs_views_and_store_only(bucket, tmp_path):
    assert aimi.governs(str(bucket / "app/comfyui/loras/a/b.safetensors"))
    assert aimi.governs(str(bucket / "aimodelinventory/blobs"))
    assert not aimi.governs(str(tmp_path / "elsewhere/x.safetensors"))
    assert not aimi.governs(str(bucket / "app/comfyui-other/x"))


def test_ro_refuses_every_mutation(bucket, monkeypatch):
    monkeypatch.setenv("LORA_MANAGER_AIMI_MODE", "ro")
    with pytest.raises(aimi.AimiReadOnlyError):
        aimi.guard_write(str(bucket / "app/comfyui/loras/x.safetensors"), "delete")
    # A PermissionError, so upstream error paths report it as such.
    with pytest.raises(PermissionError):
        aimi.guard_write(str(bucket / "app/comfyui/checkpoints/y"), "rename")


def test_rw_allows_configured_folders_only(bucket, monkeypatch):
    aimi.guard_write(str(bucket / "app/comfyui/loras/sub/x.safetensors"), "move")
    aimi.guard_write(str(bucket / "app/comfyui/checkpoints/y.safetensors"), "move")
    with pytest.raises(aimi.AimiReadOnlyError):
        aimi.guard_write(str(bucket / "app/comfyui/vae/z.safetensors"), "move")
    with pytest.raises(aimi.AimiReadOnlyError):
        aimi.guard_write(str(bucket / "app/comfyui"), "create folder")
    with pytest.raises(aimi.AimiReadOnlyError):
        aimi.guard_write(str(bucket / "aimodelinventory/models/lora/x"), "delete")
    monkeypatch.setenv("LORA_MANAGER_AIMI_RW_DIRS", "loras,checkpoints,vae")
    aimi.guard_write(str(bucket / "app/comfyui/vae/z.safetensors"), "move")


def test_guard_move_refuses_crossing_the_view(bucket, tmp_path):
    inside = str(bucket / "app/comfyui/loras/x.safetensors")
    outside = str(tmp_path / "local/x.safetensors")
    with pytest.raises(aimi.AimiError):
        aimi.guard_move(inside, outside)
    with pytest.raises(aimi.AimiError):
        aimi.guard_move(outside, inside)
    aimi.guard_move(inside, str(bucket / "app/comfyui/loras/sub/x.safetensors"))


def test_header_reads_gated(bucket, monkeypatch):
    path = str(bucket / "app/comfyui/loras/x.safetensors")
    assert aimi.skip_header_read(path)
    monkeypatch.setenv("LORA_MANAGER_AIMI_HEADER_READS", "1")
    assert not aimi.skip_header_read(path)


def test_read_safetensors_metadata_does_not_open_governed_files(booked_lora, monkeypatch):
    app, _sha256 = booked_lora

    def boom(*_args, **_kwargs):
        raise AssertionError("governed file opened")

    monkeypatch.setattr("builtins.open", boom)
    assert file_utils.read_safetensors_metadata(str(app)) == {}


# --- identity without reading ---------------------------------------------------


def test_sha256_from_sha256_blob_name(booked_lora, monkeypatch):
    app, sha = booked_lora

    def boom(*_args, **_kwargs):
        raise AssertionError("model bytes read")

    monkeypatch.setattr("builtins.open", boom)
    assert aimi.sha256_without_reading(str(app)) == sha


def test_sha256_for_sha1_blob_from_manifest(bucket):
    data = b"legacy"
    sha1 = hashlib.sha1(data).hexdigest()
    blob = _blob(bucket, "sha1", sha1, data)
    _link(bucket, f"{LORA_DIR}/old.safetensors", blob)
    app = _link(bucket, "app/comfyui/loras/old.safetensors", f"{LORA_DIR}/old.safetensors")
    claimed = _sha(data)
    _manifest(bucket, LORA_DIR, {"old.safetensors": {"sha256_claimed": claimed, "verified": False}})
    assert aimi.sha256_without_reading(str(app)) == claimed


def test_sha1_blob_without_claim_has_no_identity(bucket):
    data = b"legacy"
    sha1 = hashlib.sha1(data).hexdigest()
    blob = _blob(bucket, "sha1", sha1, data)
    _link(bucket, f"{LORA_DIR}/old.safetensors", blob)
    app = _link(bucket, "app/comfyui/loras/old.safetensors", f"{LORA_DIR}/old.safetensors")
    _manifest(bucket, LORA_DIR, {})
    assert aimi.sha256_without_reading(str(app)) == ""


def test_real_file_in_view_has_no_identity(bucket):
    path = bucket / "app/comfyui/loras/real.safetensors"
    path.write_bytes(b"not booked")
    assert aimi.sha256_without_reading(str(path)) == ""


def test_misplaced_blob_name_is_not_trusted(bucket):
    data = b"x"
    sha = _sha(data)
    wrong = f"aimodelinventory/blobs/sha256/zz/{sha}"
    (bucket / wrong).parent.mkdir(parents=True)
    (bucket / wrong).write_bytes(data)
    app = _link(bucket, "app/comfyui/loras/w.safetensors", wrong)
    assert aimi.sha256_without_reading(str(app)) == ""


def test_file_utils_hashes_use_the_chain(booked_lora):
    app, sha = booked_lora
    assert file_utils._calculate_sha256_sync(str(app)) == sha
    assert file_utils.calculate_autov2(str(app)) == sha[:10]


def test_file_utils_unchanged_outside_views(bucket, tmp_path):
    path = tmp_path / "plain.safetensors"
    path.write_bytes(b"plain")
    assert file_utils._calculate_sha256_sync(str(path)) == _sha(b"plain")


# --- rw: re-linking ---------------------------------------------------------------


def test_relink_moves_link_and_books_app_paths(booked_lora, bucket):
    app, sha = booked_lora
    new = bucket / "app/comfyui/loras/style/x.safetensors"
    aimi.relink(str(app), str(new))

    assert not os.path.lexists(app)
    assert os.readlink(new) == f"../../../../{LORA_DIR}/x.safetensors"
    assert aimi.sha256_without_reading(str(new)) == sha
    entry = _read_manifest(bucket, LORA_DIR)["files"]["x.safetensors"]
    assert entry["app_paths"] == ["app/comfyui/loras/style/x.safetensors"]
    assert _read_manifest(bucket, LORA_DIR)["app_links"] == ["app/comfyui"]


def test_relink_refuses_real_files_and_existing_targets(booked_lora, bucket):
    app, _sha256 = booked_lora
    real = bucket / "app/comfyui/loras/real.safetensors"
    real.write_bytes(b"real")
    with pytest.raises(aimi.AimiError):
        aimi.relink(str(real), str(bucket / "app/comfyui/loras/moved.safetensors"))
    with pytest.raises(aimi.AimiError):
        aimi.relink(str(app), str(real))
    assert os.path.islink(app)


def test_unlink_view_entry_keeps_blob(booked_lora, bucket):
    app, sha = booked_lora
    aimi.unlink_view_entry(str(app))
    assert not os.path.lexists(app)
    assert (bucket / f"aimodelinventory/blobs/sha256/{sha[:2]}/{sha}").is_file()
    assert os.path.islink(bucket / f"{LORA_DIR}/x.safetensors")
    assert _read_manifest(bucket, LORA_DIR)["files"]["x.safetensors"]["app_paths"] == []


def test_after_folder_rename_rekeys_app_paths(booked_lora, bucket):
    app, _sha256 = booked_lora
    aimi.relink(str(app), str(bucket / "app/comfyui/loras/old/x.safetensors"))
    os.rename(bucket / "app/comfyui/loras/old", bucket / "app/comfyui/loras/new")
    aimi.after_folder_rename(str(bucket / "app/comfyui/loras/old"), str(bucket / "app/comfyui/loras/new"))
    entry = _read_manifest(bucket, LORA_DIR)["files"]["x.safetensors"]
    assert entry["app_paths"] == ["app/comfyui/loras/new/x.safetensors"]


# --- rw: downloads -----------------------------------------------------------------


def _version(sha: str | None, name: str = "dl.safetensors") -> dict:
    hashes = {"SHA256": sha.upper()} if sha else {}
    return {
        "id": 200,
        "modelId": 100,
        "name": "v1",
        "baseModel": "SDXL 1.0",
        "model": {"name": "Some LoRA", "type": "LORA"},
        "files": [{"name": name, "primary": True, "hashes": hashes}],
    }


def _stage(plan: aimi.DownloadPlan, data: bytes) -> str:
    os.makedirs(os.path.dirname(plan.staging_path), exist_ok=True)
    with open(plan.staging_path, "wb") as handle:
        handle.write(data)
    return plan.staging_path


def test_plan_download_maps_type_and_ids(bucket):
    data = b"new lora"
    save = str(bucket / "app/comfyui/loras/new/dl.safetensors")
    plan = aimi.plan_download(save, _version(_sha(data)), None)
    assert plan.store_type == "lora"
    assert plan.dir_key == LORA_DIR
    assert plan.expected_sha256 == _sha(data)
    assert not aimi.governs(plan.staging_path)


def test_plan_download_refuses_without_civitai_ids_or_type(bucket, monkeypatch):
    save = str(bucket / "app/comfyui/loras/dl.safetensors")
    with pytest.raises(aimi.AimiError):
        aimi.plan_download(save, {"files": []}, None)
    monkeypatch.setenv("LORA_MANAGER_AIMI_RW_DIRS", "loras,checkpoints,vae")
    with pytest.raises(aimi.AimiError):
        aimi.plan_download(str(bucket / "app/comfyui/vae/v.safetensors"), _version(None), None)


def test_plan_download_refused_in_ro(bucket, monkeypatch):
    monkeypatch.setenv("LORA_MANAGER_AIMI_MODE", "ro")
    with pytest.raises(aimi.AimiReadOnlyError):
        aimi.plan_download(str(bucket / "app/comfyui/loras/dl.safetensors"), _version(None), None)


def test_staging_inside_the_mount_is_refused(bucket, monkeypatch):
    monkeypatch.setenv("LORA_MANAGER_AIMI_STAGING", str(bucket / "tmp"))
    with pytest.raises(aimi.AimiError):
        aimi.staging_dir()


def test_publish_download_books_the_whole_chain(bucket):
    data = b"new lora bytes"
    sha = _sha(data)
    save = bucket / "app/comfyui/loras/new/dl.safetensors"
    plan = aimi.plan_download(str(save), _version(sha), None)
    staging = _stage(plan, data)

    result = aimi.publish_download(plan, staging)

    blob = f"aimodelinventory/blobs/sha256/{sha[:2]}/{sha}"
    assert result == {
        "sha256": sha,
        "blob": blob,
        "model_link": f"{LORA_DIR}/dl.safetensors",
        "app_path": "app/comfyui/loras/new/dl.safetensors",
    }
    assert (bucket / blob).read_bytes() == data
    assert os.readlink(save) == f"../../../../{LORA_DIR}/dl.safetensors"
    assert save.read_bytes() == data
    assert aimi.sha256_without_reading(str(save)) == sha
    assert not os.path.exists(staging)

    manifest = _read_manifest(bucket, LORA_DIR)
    entry = manifest["files"]["dl.safetensors"]
    assert entry["verified"] is True and entry["sha256"] == sha
    assert entry["size"] == len(data)
    assert entry["app_paths"] == ["app/comfyui/loras/new/dl.safetensors"]
    assert manifest["source"] == "loramanager-download"
    assert manifest["app_links"] == ["app/comfyui"]
    assert manifest["civitai_url"] == "https://civitai.com/models/100?modelVersionId=200"


def test_publish_download_rclone_upload(bucket, monkeypatch):
    monkeypatch.setenv("LORA_MANAGER_AIMI_BLOB_UPLOAD", "rclone")
    data = b"via rclone"
    plan = aimi.plan_download(str(bucket / "app/comfyui/loras/r.safetensors"), _version(_sha(data), "r.safetensors"), None)
    aimi.publish_download(plan, _stage(plan, data))
    module = aimi._cli_state["module"]
    assert ("ensure_blob", _sha(data)) in module.CALLS


def test_publish_download_auto_falls_back_to_rclone_when_short(bucket, monkeypatch):
    monkeypatch.setattr(aimi, "_free_bytes", lambda _path: 0)
    data = b"no room for a cache copy"
    plan = aimi.plan_download(str(bucket / "app/comfyui/loras/s.safetensors"), _version(_sha(data), "s.safetensors"), None)
    aimi.publish_download(plan, _stage(plan, data))
    assert ("ensure_blob", _sha(data)) in aimi._cli_state["module"].CALLS


def test_publish_download_auto_uses_mount_with_room(bucket):
    data = b"room enough"
    plan = aimi.plan_download(str(bucket / "app/comfyui/loras/m.safetensors"), _version(_sha(data), "m.safetensors"), None)
    aimi.publish_download(plan, _stage(plan, data))
    calls = aimi._cli_state["module"].CALLS
    assert not any(call[0] == "ensure_blob" for call in calls)
    assert not os.path.exists(str(bucket / aimi._cli()[1].blob_key(_sha(data))) + ".partial")


def test_plan_download_reads_expected_size(bucket):
    version = _version(None)
    version["files"][0]["sizeKB"] = 2.5
    plan = aimi.plan_download(str(bucket / "app/comfyui/loras/dl.safetensors"), version, None)
    assert plan.expected_size == 2560


async def test_wait_for_room_pauses_until_space(bucket, monkeypatch):
    free = iter([0, 0, 1 << 40])
    monkeypatch.setattr(aimi, "_free_bytes", lambda _path: next(free))
    version = _version(None)
    version["files"][0]["sizeKB"] = 1024
    plan = aimi.plan_download(str(bucket / "app/comfyui/loras/dl.safetensors"), version, None)
    await aimi.wait_for_room(plan, poll=0)
    assert next(free, None) is None


async def test_wait_for_room_never_aborts(bucket, monkeypatch):
    monkeypatch.setattr(aimi, "_free_bytes", lambda _path: 0)
    monkeypatch.setenv("LORA_MANAGER_AIMI_SPACE_WAIT", "0")
    version = _version(None)
    version["files"][0]["sizeKB"] = 1024
    plan = aimi.plan_download(str(bucket / "app/comfyui/loras/dl.safetensors"), version, None)
    await aimi.wait_for_room(plan, poll=0)


def test_publish_download_mismatch_books_nothing(bucket):
    plan = aimi.plan_download(str(bucket / "app/comfyui/loras/bad.safetensors"), _version("a" * 64, "bad.safetensors"), None)
    staging = _stage(plan, b"tampered")
    with pytest.raises(aimi.AimiError, match="mismatch"):
        aimi.publish_download(plan, staging)
    assert not os.path.exists(staging)
    assert not os.path.lexists(bucket / "app/comfyui/loras/bad.safetensors")
    assert not (bucket / "aimodelinventory/blobs").exists()
    assert not (bucket / LORA_DIR).exists()


def test_publish_download_without_civitai_hash_is_unverified(bucket):
    data = b"no hash given"
    plan = aimi.plan_download(str(bucket / "app/comfyui/loras/u.safetensors"), _version(None, "u.safetensors"), None)
    aimi.publish_download(plan, _stage(plan, data))
    entry = _read_manifest(bucket, LORA_DIR)["files"]["u.safetensors"]
    assert entry["verified"] is False
    assert entry["sha256_claimed"] == _sha(data)


def test_publish_download_reuses_existing_blob_and_adds_app_path(booked_lora, bucket):
    app, sha = booked_lora
    plan = aimi.plan_download(str(bucket / "app/comfyui/loras/again/x.safetensors"), _version(sha, "x.safetensors"), None)
    aimi.publish_download(plan, _stage(plan, b"lora-bytes"))
    entry = _read_manifest(bucket, LORA_DIR)["files"]["x.safetensors"]
    assert entry["app_paths"] == [
        "app/comfyui/loras/again/x.safetensors",
        "app/comfyui/loras/x.safetensors",
    ]
    assert os.path.islink(app)


def test_publish_download_accepts_vouched_legacy_link(bucket):
    data = b"legacy model"
    sha1 = hashlib.sha1(data).hexdigest()
    legacy = _blob(bucket, "sha1", sha1, data)
    _link(bucket, f"{LORA_DIR}/l.safetensors", legacy)
    _manifest(bucket, LORA_DIR, {"l.safetensors": {"sha256_claimed": _sha(data), "verified": False}})
    plan = aimi.plan_download(str(bucket / "app/comfyui/loras/l.safetensors"), _version(_sha(data), "l.safetensors"), None)
    result = aimi.publish_download(plan, _stage(plan, data))
    assert result["blob"] == legacy
    assert _read_manifest(bucket, LORA_DIR)["files"]["l.safetensors"]["blob"] == legacy


def test_publish_download_refuses_foreign_repo_link(bucket):
    other = _blob(bucket, "sha256", _sha(b"other"), b"other")
    _link(bucket, f"{LORA_DIR}/c.safetensors", other)
    data = b"mine"
    plan = aimi.plan_download(str(bucket / "app/comfyui/loras/c.safetensors"), _version(_sha(data), "c.safetensors"), None)
    with pytest.raises(aimi.AimiError, match="points elsewhere"):
        aimi.publish_download(plan, _stage(plan, data))
    assert not os.path.lexists(bucket / "app/comfyui/loras/c.safetensors")


# --- sidecars and recipes -------------------------------------------------------------


def test_sidecar_storage_is_forced_centralized(bucket, monkeypatch):
    from py.utils import sidecar_paths

    monkeypatch.setattr(sidecar_paths, "_get_settings_value", lambda key, default=None: "alongside")
    assert sidecar_paths.get_storage_mode() == sidecar_paths.STORAGE_MODE_CENTRALIZED
    monkeypatch.setenv("LORA_MANAGER_AIMI_MODE", "off")
    assert sidecar_paths.get_storage_mode() == sidecar_paths.STORAGE_MODE_ALONGSIDE


def test_recipes_leave_the_view(bucket, tmp_path, monkeypatch):
    from py.services import settings_manager
    from py.utils import recipes_paths, sidecar_paths

    class _Settings:
        def __init__(self, recipes_path):
            self.recipes_path = recipes_path

        def get(self, key, default=None):
            return self.recipes_path if key == "recipes_path" else default

    sidecars = tmp_path / "loramanager-metadata"
    monkeypatch.setattr(sidecar_paths, "get_sidecar_root", lambda: str(sidecars))
    monkeypatch.setattr(
        settings_manager, "get_settings_manager", lambda: _Settings(str(bucket / "app/comfyui/loras/recipes"))
    )
    assert recipes_paths.get_effective_recipes_dir() == str(sidecars / "recipes")

    custom = tmp_path / "my-recipes"
    monkeypatch.setattr(settings_manager, "get_settings_manager", lambda: _Settings(str(custom)))
    assert recipes_paths.get_effective_recipes_dir() == str(custom)


def test_stats_migration_keeps_the_shared_file(bucket, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from py.utils.usage_stats import UsageStats

    legacy = bucket / "app/comfyui/loras" / UsageStats.STATS_FILENAME
    legacy.write_text("{}")
    target = tmp_path / "settings/stats" / UsageStats.STATS_FILENAME
    stats = SimpleNamespace(_stats_file_path=str(target), _get_old_stats_file_path=lambda: str(legacy))

    UsageStats._migrate_from_old_location(stats)
    assert target.read_text() == "{}"
    assert legacy.exists()
