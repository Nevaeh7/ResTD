#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
LOCALE=${1:?Usage: 03_gr.sh <us|es|jp> [options]}
shift
exec "${PYTHON:-python}" -m restd.train --locale "$LOCALE" "$@"
