# Data transfer

The Git repository contains source code, not APK samples, extracted method
sequences, experiment labels/splits, trained checkpoints or generated logs.

To reproduce the existing local split on a server, separately transfer:

- `data/prepared.json`
- `data/methods/` (JSON sequence caches referenced by the prepared records)

Both currently exist in the local SAS-Droid directory and are excluded from Git.
Keep the prepared file unchanged to retain the existing split and GA vocabulary.
After transfer, run `python server_preflight.py --device cuda`.

Alternatively, place your authorized APK dataset under `apk/benign` and
`apk/malradar`, extract with data.py and create a new prepared split. See
SERVER_RUN.md. Newly prepared data is a new experiment, not the old fixed split.
