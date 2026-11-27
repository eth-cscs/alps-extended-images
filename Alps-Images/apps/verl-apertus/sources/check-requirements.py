"""Fail when an installed distribution's own requirements are not met.

Used after a --no-deps install: pip does not resolve the package's requirements then, so this
checks each unconditional requirement (no extras, markers evaluated) against what is installed.
Usage: python check-requirements.py DIST [DIST ...]
"""

import importlib.metadata as metadata
import sys

from packaging.requirements import Requirement


def problems(dist: str) -> list[str]:
    found = []
    for line in metadata.requires(dist) or []:
        req = Requirement(line)
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        try:
            installed = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            found.append(f"{dist} needs {req}, not installed")
            continue
        if req.specifier and not req.specifier.contains(installed, prereleases=True):
            found.append(f"{dist} needs {req}, {req.name} {installed} is installed")
    return found


def main() -> int:
    found = [p for dist in sys.argv[1:] for p in problems(dist)]
    for p in found:
        print(p, file=sys.stderr)
    if not found:
        print("requirements met:", ", ".join(sys.argv[1:]))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
