#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/link_data.sh [TARGET_DIR]
# Creates a symlink data/raw_campaigns -> TARGET_DIR (default: /home/vroston/data/raw_campaigns)

TARGET_DIR="${1:-/home/vroston/data/raw_campaigns}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="$REPO_ROOT/data/raw_campaigns"

if [ -e "$DEST_DIR" ] || [ -L "$DEST_DIR" ]; then
  echo "Removing existing $DEST_DIR"
  rm -rf "$DEST_DIR"
fi

mkdir -p "$(dirname "$DEST_DIR")"
ln -s "$TARGET_DIR" "$DEST_DIR"

echo "Linked $TARGET_DIR -> $DEST_DIR"
