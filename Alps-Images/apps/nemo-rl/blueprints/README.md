# NeMo-RL blueprints

Machine-readable descriptions of the training setups in [`../example/`](../example/README.md),
so that tools such as the FirecREST UI can offer them without hard-coding each one.

A blueprint is one tested combination of topology, engines, models and dataset. Each lives in its
own folder; the combinations are fields, not nested folders, so only setups that have actually
been run are offered.

```text
blueprints/
  schema/blueprint.schema.yaml    JSON Schema for blueprint.yaml (written in YAML)
  schema/catalog.schema.yaml      JSON Schema for the catalogs
  models.yaml                     models blueprints can list
  datasets.yaml                   datasets blueprints can list
  <id>/blueprint.yaml             one setup
  index.json                      generated from all of the above; never edit by hand
  tools/build_index.py            checks the blueprints and writes index.json
```

## What a blueprint declares

- `topology`: `mode` (sync or async GRPO) and `placement` (training and generation on split or
  colocated nodes)
- `engines`: the training and generation backends
- `launch`: the recipe folder under `example/` that `train.sh` runs, and the container image
- `models` and `datasets`: ids from the catalogs. List only models the recipe's config supports:
  the current configs set Apertus 2 MoE options, so dense models need their own setup
- `resources`: GPUs per node, the parameters that add up to the node count, the profile variable
  holding the time limit, and fixed `sbatch` options
- `profiles`: one per `<recipe>/<id>.env`, the first being the default
- `parameters`: the values a run may change, with labels and ranges. They must be exactly the
  `VAR="${VAR:-default}"` lines of the profiles they apply to; defaults are read from the
  profiles, so they are never written twice
- `rules`: checks across parameters (`atMost`, `divisible`), which tools evaluate before launch
- `monitoring`: destinations, with the fixed and user-set environment variables each needs
- `references`: reference runs, if any (`status: validated` setups should have one)

A tool launching a blueprint exports the chosen parameter values, `HF_MODEL` (the model's path)
and `NEMO_RL_IMAGE`, and runs `train.sh <recipe> --profile <profile>`. Blueprints contain no
shell code, so a blueprint can't change what a job runs beyond these values.

## Adding or changing a blueprint

1. Add or edit the recipe and its `.env` profiles under `example/` as usual.
2. Add or edit `<id>/blueprint.yaml` (the folder name is the id), and catalog entries if needed.
3. Regenerate the index from the repository root and commit it:

   ```bash
   uv run --with pyyaml --with jsonschema python Alps-Images/apps/nemo-rl/blueprints/tools/build_index.py
   ```

The GitHub preflight runs the same script with `--check`: it fails when a blueprint doesn't match
its schema, its profiles or the catalogs, or when the committed `index.json` is out of date.

## YAML conventions

Only `build_index.py` parses the YAML; everything else reads `index.json`, so all readers see the
same values. Still, write YAML that every parser reads the same way:

- no anchors, aliases or tags (they are rejected), and no duplicate keys
- quote strings that look like other types: times such as `'07:00:00'`, `'false'`, `'no'`
- write floats with a dot and a signed exponent, such as `1.0e-6`, never `1e-6`
