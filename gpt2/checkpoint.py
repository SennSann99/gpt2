from pathlib import Path


def resolve_checkpoint_path(checkpoint_path: str | None = None) -> Path:
    """Return an explicit checkpoint or the newest usable saved checkpoint."""
    path = Path(checkpoint_path or "checkpoints")

    if path.is_file():
        return path

    if path.is_dir():
        version_dirs: list[tuple[int, Path]] = []
        for child in path.iterdir():
            if not child.is_dir() or not child.name.startswith("version_"):
                continue
            try:
                version = int(child.name[len("version_") :])
            except ValueError:
                continue
            version_dirs.append((version, child))

        for _, version_dir in sorted(version_dirs, reverse=True):
            for filename in ("best.ckpt", "last.ckpt", "interrupted.ckpt"):
                candidate = version_dir / filename
                if candidate.is_file():
                    return candidate

    raise FileNotFoundError(
        f"No checkpoint found at {path}. Expected a checkpoint file or "
        f"{path}/version_N/best.ckpt (or last.ckpt)."
    )
