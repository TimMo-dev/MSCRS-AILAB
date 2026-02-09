#!/usr/bin/env bash
#SBATCH --job-name=build_pyg
#SBATCH --output=build_pyg.out
#SBATCH --error=build_pyg.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00

set -euo pipefail

export SINGULARITY_TMPDIR="$HOME/.singularity/tmp"
export SINGULARITY_CACHEDIR="$HOME/.singularity/cache"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

DEF="pytorch38_cu111_pyg.def"
SIF="pytorch38_cu111_pyg.sif"

# Build output on local filesystem (NOT /ceph)
LOCAL_OUT="/tmp/$USER"
mkdir -p "$LOCAL_OUT"
LOCAL_SIF="$LOCAL_OUT/$SIF"

echo "Building SIF to local path: $LOCAL_SIF"
singularity build --fakeroot --force "$LOCAL_SIF" "$DEF"

echo "Copying back to project dir: $PWD/$SIF"
cp -f "$LOCAL_SIF" "$PWD/$SIF"

echo "Done:"
ls -lh "$PWD/$SIF"

