import importlib.util
import shutil
import sys
from pathlib import Path

import pytest


BUILDER = Path(__file__).resolve().parents[1] / "build_index.py"
spec = importlib.util.spec_from_file_location("build_index", BUILDER)
bi = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bi
spec.loader.exec_module(bi)

VLLM = "apertus2-moe-gsm8k-vllm"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    """A copy of the blueprints and the example they describe, to break on purpose."""
    shutil.copytree(bi.BLUEPRINTS_DIR, tmp_path / "blueprints", ignore=shutil.ignore_patterns("tools"))
    shutil.copytree(bi.EXAMPLE_DIR, tmp_path / "example")
    monkeypatch.setattr(bi, "BLUEPRINTS_DIR", tmp_path / "blueprints")
    monkeypatch.setattr(bi, "EXAMPLE_DIR", tmp_path / "example")
    return tmp_path


def edit(path, old, new):
    content = path.read_text()
    assert old in content, old
    path.write_text(content.replace(old, new, 1))


def errors_of(app):
    _, errors = bi.build_index()
    return "\n".join(errors)


def blueprint(app, blueprint_id=VLLM):
    return app / "blueprints" / blueprint_id / "blueprint.yaml"


def test_the_committed_blueprints_are_valid_and_the_index_is_up_to_date():
    index, errors = bi.build_index()
    assert errors == []
    assert (bi.BLUEPRINTS_DIR / "index.json").read_text() == bi.render(index)


def test_profile_defaults_come_from_the_env_files():
    index, _ = bi.build_index()
    vllm = next(entry for entry in index["blueprints"] if entry["id"] == VLLM)
    adamw, muon = vllm["profiles"]
    assert adamw["defaults"]["TRAIN_NODES"] == 8 and muon["defaults"]["TRAIN_NODES"] == 16
    assert adamw["defaults"]["BASE_LR"] == 1.0e-6
    # Profile-specific parameters only appear in their profiles
    assert "MATRIX_LR" not in adamw["defaults"] and muon["defaults"]["MATRIX_LR"] == 5.0e-6
    assert adamw["timeLimit"] == "07:00:00"
    # Catalog entries are expanded
    assert vllm["models"][0]["architecture"] == "moe"


def test_rejects_anchors_and_duplicate_keys(app):
    edit(blueprint(app), "datasets:\n  - gsm8k", "datasets: &d\n  - gsm8k")
    assert "anchors and aliases are not allowed" in errors_of(app)
    edit(blueprint(app), "datasets: &d", "datasets:")
    edit(blueprint(app), "algorithm: grpo", "algorithm: grpo\nalgorithm: grpo")
    assert "duplicate keys algorithm" in errors_of(app)


def test_numbers_must_be_unambiguous_yaml(app):
    # PyYAML reads 1e-6 as a string, YAML 1.2 parsers as a number: the schema rejects it
    edit(blueprint(app), "    min: 0\n    max: 1\n", "    min: 1e-6\n    max: 1\n")
    assert "'1e-6' is not of type 'number'" in errors_of(app)


def test_parameters_must_match_what_the_profiles_let_a_run_override(app):
    profile = app / "example" / "vllm" / "gsm8k" / "adamw.env"
    edit(profile, 'NUM_PROMPTS="${NUM_PROMPTS:-128}"', "NUM_PROMPTS=128")
    edit(profile, "WEIGHT_DECAY=0.01", 'WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"')
    errors = errors_of(app)
    assert f"{VLLM}/adamw: parameter NUM_PROMPTS is not overridable in the profile" in errors
    assert f"{VLLM}/adamw: overridable WEIGHT_DECAY is not a declared parameter" in errors


def test_profile_defaults_must_fit_the_parameters_and_rules(app):
    profile = app / "example" / "vllm" / "gsm8k" / "muon.env"
    edit(profile, 'TRAIN_NODES="${TRAIN_NODES:-16}"', 'TRAIN_NODES="${TRAIN_NODES:-200}"')
    edit(profile, 'MIN_LR="${MIN_LR:-0.0}"', 'MIN_LR="${MIN_LR:-1.0}"')
    edit(profile, 'UPDATES_PER_ROLLOUT="${UPDATES_PER_ROLLOUT:-4}"', 'UPDATES_PER_ROLLOUT="${UPDATES_PER_ROLLOUT:-3}"')
    edit(profile, 'BASE_LR="${BASE_LR:-2.0e-7}"', 'BASE_LR="${BASE_LR:-fast}"')
    errors = errors_of(app)
    assert f"{VLLM}/muon: TRAIN_NODES default 200 is outside its range" in errors
    assert f"{VLLM}/muon: BASE_LR default 'fast' is not float" in errors
    assert "defaults break rule 'Prompts per step × generations per prompt" in errors


def test_rules_are_evaluated_against_the_defaults(app):
    profile = app / "example" / "vllm" / "gsm8k" / "adamw.env"
    edit(profile, 'TRAIN_EXPERT_PARALLEL_SIZE="${TRAIN_EXPERT_PARALLEL_SIZE:-32}"', 'TRAIN_EXPERT_PARALLEL_SIZE="${TRAIN_EXPERT_PARALLEL_SIZE:-24}"')
    edit(profile, 'MIN_LR="${MIN_LR:-5.0e-7}"', 'MIN_LR="${MIN_LR:-1.0e-5}"')
    errors = errors_of(app)
    assert f"{VLLM}/adamw: defaults break rule 'Must divide the training GPUs'" in errors
    assert f"{VLLM}/adamw: defaults break rule 'Minimum learning rate must not exceed" in errors


def test_references_to_unknown_things_are_reported(app):
    edit(blueprint(app), "models:\n  - apertus-2-chonk", "models:\n  - qwen3-8b")
    edit(blueprint(app), "dividend: [TRAIN_NODES, 4]", "dividend: [NODES, 4]")
    errors = errors_of(app)
    assert f"{VLLM}: unknown model qwen3-8b" in errors
    assert "uses undeclared parameter NODES" in errors


def test_missing_recipes_and_profiles_are_reported(app):
    edit(blueprint(app), "  recipe: vllm/gsm8k", "  recipe: vllm/math")
    assert f"{VLLM}: recipe folder example/vllm/math not found" in errors_of(app)
    edit(blueprint(app), "  recipe: vllm/math", "  recipe: vllm/gsm8k")
    (app / "example" / "vllm" / "gsm8k" / "muon.env").unlink()
    assert f"{VLLM}: profile file muon.env not found" in errors_of(app)


def test_the_folder_names_the_blueprint(app):
    shutil.move(blueprint(app).parent, app / "blueprints" / "renamed")
    assert f"renamed: id {VLLM} does not match its folder" in errors_of(app)


@pytest.mark.parametrize(
    ("rule", "values", "holds"),
    [
        ({"kind": "atMost", "value": "A", "limit": "B"}, {"A": 1, "B": 1}, True),
        ({"kind": "atMost", "value": "A", "limit": "B"}, {"A": 2, "B": 1}, False),
        ({"kind": "divisible", "dividend": ["A", 4], "divisor": ["B"]}, {"A": 8, "B": 32}, True),
        ({"kind": "divisible", "dividend": ["A", 4], "divisor": ["B"]}, {"A": 8, "B": 24}, False),
        ({"kind": "divisible", "dividend": ["A"], "divisor": ["B"]}, {"A": 8, "B": 0}, False),
    ],
)
def test_rule_evaluation(rule, values, holds):
    assert bi.rule_holds(rule, values) is holds


def test_check_mode_fails_on_a_stale_index(app, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_index.py"])
    assert bi.main() == 0
    monkeypatch.setattr(sys, "argv", ["build_index.py", "--check"])
    assert bi.main() == 0
    edit(app / "blueprints" / "datasets.yaml", "Grade school math", "Grade-school math")
    assert bi.main() == 1
    assert "index.json is out of date" in capsys.readouterr().err
