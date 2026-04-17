#!/usr/bin/env bash

set -e

echo "⬇️ Downloading docking_vina..."

FILE_ID="1B7270Q4yG00TuJztwveHcLo9Mr6WaLU5"

pip install -q gdown
gdown $FILE_ID -O docking_vina.zip

echo "📦 Unzipping..."
unzip -q docking_vina.zip
rm docking_vina.zip

echo "✅ docking_vina ready"

echo ""
echo "⚠️ IMPORTANT: Install MGLTools manually:"
echo "https://ccsb.scripps.edu/mgltools/"