#!/usr/bin/env bash
# Download the FER-2013 dataset from Kaggle.
#
# Requires the Kaggle CLI (`pip install kaggle`) with credentials in
# ~/.kaggle/kaggle.json. Saves fer2013.csv into ./data/.
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DATA_DIR"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "error: kaggle CLI not found. Install with 'pip install kaggle' and place ~/.kaggle/kaggle.json."
  exit 1
fi

echo "downloading FER-2013 to $DATA_DIR ..."
kaggle datasets download -d msambare/fer2013 -p "$DATA_DIR" --unzip || \
kaggle datasets download -d deadskull7/fer2013 -p "$DATA_DIR" --unzip

if [[ ! -f "$DATA_DIR/fer2013.csv" ]]; then
  echo "warning: expected $DATA_DIR/fer2013.csv but it is missing."
  echo "If the dataset extracted into class subdirectories instead of a CSV,"
  echo "use a different mirror or convert the folders to CSV format."
  exit 1
fi

echo "done."
