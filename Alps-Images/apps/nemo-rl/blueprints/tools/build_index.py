#!/usr/bin/env python3
"""Check the NeMo-RL blueprints and build blueprints/index.json from them.

  build_index.py           write index.json
  build_index.py --check   fail if index.json is not what the blueprints produce

Blueprints are written in YAML (blueprints/<id>/blueprint.yaml, plus models.yaml and
datasets.yaml); tools such as the FirecREST UI read only the generated index.json. Parsing once,
here, means every reader sees the same values: YAML 1.1 (PyYAML) and YAML 1.2 parsers disagree
on scalars such as 1e-6 or no.

Besides the JSON Schemas in blueprints/schema/, every blueprint is checked against the files it
describes:
- each profile is a file <recipe>/<profile>.env under example/
- the parameters are exactly the values each profile lets a run override
  (VAR="${VAR:-default}"); their defaults are read from the profile, so they have one source
- the profile defaults are within the parameter ranges and satisfy the rules
- models and datasets exist in the catalogs

Requires PyYAML and jsonschema (uv run --with pyyaml --with jsonschema python build_index.py).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import jsonschema
import yaml

INDEX_SCHEMA_VERSION = 1
BLUEPRINTS_DIR = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = BLUEPRINTS_DIR.parent / "example"

# VAR="${VAR:-default}" lets a run override VAR; anything else is fixed by the profile
OVERRIDABLE_PATTERN = re.compile(r'^([A-Z][A-Z0-9_]*)="\$\{\1:-([^}"]*)\}"\s*(?:#.*)?$')
ASSIGNMENT_PATTERN = re.compile(r"^([A-Z][A-Z0-9_]*)=")
INT_PATTERN = re.compile(r"^-?\d+$")
FLOAT_PATTERN = re.compile(r"^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")

Json = dict[str, Any]


class BlueprintError(Exception):
    pass


class StrictLoader(yaml.SafeLoader):
    """Safe YAML without anchors, aliases or duplicate keys."""

    def compose_node(self, parent, index):  # type: ignore[override]
        event = self.peek_event()
        if isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None):
            raise BlueprintError(f"{self.name}: anchors and aliases are not allowed")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):  # type: ignore[override]
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        duplicates = sorted({str(key) for key in keys if keys.count(key) > 1})
        if duplicates:
            raise BlueprintError(f"{self.name}: duplicate keys {', '.join(duplicates)}")
        return super().construct_mapping(node, deep=deep)


def load_yaml(path: Path) -> Any:
    try:
        return yaml.load(path.read_text(), Loader=StrictLoader)  # noqa: S506 (a SafeLoader)
    except yaml.YAMLError as error:
        raise BlueprintError(f"{path}: {error}") from None


def validate(data: Any, schema: Json, path: Path) -> list[str]:
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{path}: {'/'.join(str(p) for p in error.absolute_path) or '(root)'}: {error.message}"
        for error in sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
    ]


def read_profile(path: Path) -> tuple[dict[str, str], set[str]]:
    """The overridable variables with their defaults, and the fixed variables, of a .env profile."""
    overridable: dict[str, str] = {}
    fixed: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = OVERRIDABLE_PATTERN.match(line)
        if match:
            overridable[match.group(1)] = match.group(2)
        elif assignment := ASSIGNMENT_PATTERN.match(line):
            fixed.add(assignment.group(1))
    return overridable, fixed


def parse_number(text: str, kind: str) -> int | float | None:
    if kind == "int":
        return int(text) if INT_PATTERN.match(text) else None
    return float(text) if FLOAT_PATTERN.match(text) else None


def product(factors: list[str | int], values: dict[str, int | float]) -> float:
    result: float = 1
    for factor in factors:
        result *= factor if isinstance(factor, int) else values[factor]
    return result


def rule_holds(rule: Json, values: dict[str, int | float]) -> bool:
    if rule["kind"] == "atMost":
        return values[rule["value"]] <= values[rule["limit"]]
    divisor = product(rule["divisor"], values)
    return divisor != 0 and product(rule["dividend"], values) % divisor == 0


def rule_variables(rule: Json) -> set[str]:
    if rule["kind"] == "atMost":
        return {rule["value"], rule["limit"], rule["target"]}
    factors = rule["dividend"] + rule["divisor"]
    return {factor for factor in factors if isinstance(factor, str)} | {rule["target"]}


def load_catalog(name: str, key: str, schema: Json, errors: list[str]) -> dict[str, Json]:
    path = BLUEPRINTS_DIR / name
    data = load_yaml(path)
    schema_errors = validate(data, schema, path)
    if schema_errors or key not in data:
        errors.extend(schema_errors or [f"{path}: missing {key}"])
        return {}
    entries: dict[str, Json] = {}
    for entry in data[key]:
        if entry["id"] in entries:
            errors.append(f"{path}: duplicate id {entry['id']}")
        entries[entry["id"]] = entry
    return entries


def check_blueprint(
    path: Path, blueprint: Json, models: dict[str, Json], datasets: dict[str, Json]
) -> tuple[Json | None, list[str]]:
    """The blueprint's index entry, or the reasons it doesn't match what it describes."""
    errors: list[str] = []
    where = path.parent.name
    if blueprint["id"] != where:
        errors.append(f"{where}: id {blueprint['id']} does not match its folder")

    for model_id in blueprint["models"]:
        if model_id not in models:
            errors.append(f"{where}: unknown model {model_id}")
    for dataset_id in blueprint["datasets"]:
        if dataset_id not in datasets:
            errors.append(f"{where}: unknown dataset {dataset_id}")

    parameters = {parameter["key"]: parameter for parameter in blueprint["parameters"]}
    if len(parameters) != len(blueprint["parameters"]):
        errors.append(f"{where}: duplicate parameter keys")
    profile_ids = [profile["id"] for profile in blueprint["profiles"]]
    if len(set(profile_ids)) != len(profile_ids):
        errors.append(f"{where}: duplicate profile ids")
    for key, parameter in parameters.items():
        if parameter["min"] > parameter["max"]:
            errors.append(f"{where}: {key} has min above max")
        for profile_id in parameter.get("profiles", []):
            if profile_id not in profile_ids:
                errors.append(f"{where}: {key} names unknown profile {profile_id}")
    for key in blueprint["resources"]["nodes"]:
        if key not in parameters:
            errors.append(f"{where}: node count uses undeclared parameter {key}")
    for rule in blueprint.get("rules", []):
        for key in sorted(rule_variables(rule) - parameters.keys()):
            errors.append(f"{where}: rule '{rule['message']}' uses undeclared parameter {key}")
    for reference in blueprint.get("references", []):
        if reference["profile"] not in profile_ids:
            errors.append(f"{where}: reference names unknown profile {reference['profile']}")
    if errors:
        return None, errors

    recipe_dir = (EXAMPLE_DIR / blueprint["launch"]["recipe"]).resolve()
    if not recipe_dir.is_relative_to(EXAMPLE_DIR.resolve()) or not recipe_dir.is_dir():
        return None, [f"{where}: recipe folder example/{blueprint['launch']['recipe']} not found"]

    time_limit = blueprint["resources"]["timeLimitVariable"]
    profiles = []
    for profile in blueprint["profiles"]:
        profile_path = recipe_dir / f"{profile['id']}.env"
        if not profile_path.is_file():
            errors.append(f"{where}: profile file {profile_path.name} not found")
            continue
        overridable, _ = read_profile(profile_path)
        if time_limit not in overridable:
            errors.append(f"{where}/{profile['id']}: {time_limit} is not overridable")
        applicable = {
            key
            for key, parameter in parameters.items()
            if profile["id"] in parameter.get("profiles", profile_ids)
        }
        for key in sorted(applicable - overridable.keys()):
            errors.append(f"{where}/{profile['id']}: parameter {key} is not overridable in the profile")
        for key in sorted(overridable.keys() - applicable - {time_limit}):
            errors.append(f"{where}/{profile['id']}: overridable {key} is not a declared parameter")

        defaults: dict[str, int | float] = {}
        for key in sorted(applicable & overridable.keys()):
            parameter = parameters[key]
            value = parse_number(overridable[key], parameter["type"])
            if value is None:
                errors.append(f"{where}/{profile['id']}: {key} default {overridable[key]!r} is not {parameter['type']}")
            elif not parameter["min"] <= value <= parameter["max"]:
                errors.append(f"{where}/{profile['id']}: {key} default {value} is outside its range")
            else:
                defaults[key] = value
        for rule in blueprint.get("rules", []):
            if rule_variables(rule) <= defaults.keys() and not rule_holds(rule, defaults):
                errors.append(f"{where}/{profile['id']}: defaults break rule '{rule['message']}'")
        profiles.append({**profile, "timeLimit": overridable.get(time_limit), "defaults": defaults})
    if errors:
        return None, errors

    entry = {
        **blueprint,
        "models": [models[model_id] for model_id in blueprint["models"]],
        "datasets": [datasets[dataset_id] for dataset_id in blueprint["datasets"]],
        "profiles": profiles,
    }
    del entry["schemaVersion"]
    return entry, []


def build_index() -> tuple[Json, list[str]]:
    errors: list[str] = []
    blueprint_schema = load_yaml(BLUEPRINTS_DIR / "schema" / "blueprint.schema.yaml")
    catalog_schema = load_yaml(BLUEPRINTS_DIR / "schema" / "catalog.schema.yaml")
    models = load_catalog("models.yaml", "models", catalog_schema, errors)
    datasets = load_catalog("datasets.yaml", "datasets", catalog_schema, errors)

    entries = []
    for path in sorted(BLUEPRINTS_DIR.glob("*/blueprint.yaml")):
        try:
            blueprint = load_yaml(path)
        except BlueprintError as error:
            errors.append(str(error))
            continue
        schema_errors = validate(blueprint, blueprint_schema, path.relative_to(BLUEPRINTS_DIR))
        if schema_errors:
            errors.extend(schema_errors)
            continue
        entry, blueprint_errors = check_blueprint(path, blueprint, models, datasets)
        errors.extend(blueprint_errors)
        if entry:
            entries.append(entry)
    if not entries and not errors:
        errors.append("no blueprints found")
    return {"schemaVersion": INDEX_SCHEMA_VERSION, "blueprints": entries}, errors


def render(index: Json) -> str:
    return json.dumps(index, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if index.json is out of date")
    args = parser.parse_args()
    try:
        index, errors = build_index()
    except BlueprintError as error:
        errors = [str(error)]
    if errors:
        for error in errors:
            print(f"::error::{error}", file=sys.stderr)
        return 1
    index_path = BLUEPRINTS_DIR / "index.json"
    content = render(index)
    if args.check:
        if not index_path.is_file() or index_path.read_text() != content:
            print(
                "::error::blueprints/index.json is out of date; run "
                "Alps-Images/apps/nemo-rl/blueprints/tools/build_index.py and commit the result",
                file=sys.stderr,
            )
            return 1
        print(f"{len(index['blueprints'])} blueprints checked; index.json is up to date")
        return 0
    index_path.write_text(content)
    print(f"Wrote {len(index['blueprints'])} blueprints to {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
