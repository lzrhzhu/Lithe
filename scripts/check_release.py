"""Build isolated wheel/sdist artifacts and verify their version metadata."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--clean", action="store_true",
                        help="remove stale build/dist/egg-info before checking")
    parser.add_argument("--clean-only", action="store_true",
                        help="remove generated build/dist/egg-info and exit")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    version_file = root / args.package / "__init__.py"
    match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]',
                      version_file.read_text(encoding="utf-8"), re.MULTILINE)
    if not match:
        raise SystemExit(f"Could not find literal __version__ in {version_file}")
    version = match.group(1)

    def clean_generated() -> None:
        for generated in (root / "build", root / "dist",
                          root / f"{args.distribution.replace('-', '_')}.egg-info"):
            if generated.is_dir():
                shutil.rmtree(generated)
            elif generated.exists():
                generated.unlink()

    if args.clean or args.clean_only:
        clean_generated()
    if args.clean_only:
        return 0

    with tempfile.TemporaryDirectory(prefix="lithe-release-check-") as tmp:
        out = Path(tmp)
        subprocess.run([sys.executable, "-m", "build", "--outdir", str(out)],
                       cwd=root, check=True)
        artifacts = sorted(p for p in out.iterdir() if p.is_file())
        wheels = [p for p in artifacts if p.suffix == ".whl"]
        sdists = [p for p in artifacts if p.name.endswith(".tar.gz")]
        if len(wheels) != 1 or len(sdists) != 1 or len(artifacts) != 2:
            raise SystemExit(f"Expected exactly one wheel and sdist, got: {artifacts}")

        expected_stem = args.distribution.replace("-", "_")
        if wheels[0].name != f"{expected_stem}-{version}-py3-none-any.whl":
            raise SystemExit(f"Unexpected wheel name: {wheels[0].name}")
        if sdists[0].name != f"{expected_stem}-{version}.tar.gz":
            raise SystemExit(f"Unexpected sdist name: {sdists[0].name}")

        with zipfile.ZipFile(wheels[0]) as wheel:
            metadata_names = [n for n in wheel.namelist()
                              if n.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1:
                raise SystemExit("Wheel must contain exactly one METADATA file")
            wheel_metadata = wheel.read(metadata_names[0]).decode("utf-8")
        with tarfile.open(sdists[0], "r:gz") as archive:
            # Setuptools includes both the canonical top-level PKG-INFO and
            # its egg-info copy. Read only the canonical one.
            pkg_info = f"{expected_stem}-{version}/PKG-INFO"
            if pkg_info not in archive.getnames():
                raise SystemExit(f"Sdist lacks top-level {pkg_info}")
            stream = archive.extractfile(pkg_info)
            assert stream is not None
            sdist_metadata = stream.read().decode("utf-8")

        def metadata_field(metadata: str, name: str) -> str | None:
            for line in metadata.splitlines():
                if line.startswith(f"{name}:"):
                    return line.split(":", 1)[1].strip()
            return None

        for label, metadata in (("wheel", wheel_metadata),
                                ("sdist", sdist_metadata)):
            if metadata_field(metadata, "Name") != args.distribution:
                raise SystemExit(f"{label} has wrong distribution name")
            got_version = metadata_field(metadata, "Version")
            if got_version != version:
                raise SystemExit(f"{label} version differs from source "
                                 f"{version}: {got_version!r}")

        subprocess.run([sys.executable, "-m", "twine", "check", "--strict",
                        *(str(p) for p in artifacts)], check=True)
        print(f"Validated {args.distribution} {version}: wheel + sdist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
