#!/usr/bin/env bash
# Software stack of the paper runs = slime@8efb1166's own stack plus a small patch.
#
# 1. Build slime's environment first, exactly as slime@8efb1166 does it (CUDA 12.9, torch 2.9.1, SGLang,
#    Megatron-LM 3714d81d, TransformerEngine 2.10, flash-attn 2.7.4.post1, apex, mbridge, ...):
#       https://github.com/THUDM/slime/blob/8efb1166/build_conda.sh   (or slime's Docker image of that commit)
# 2. Then, inside that environment, run this script from the repository root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SLIME_COMMIT=8efb1166
mkdir -p "$ROOT/third_party"
if [ ! -d "$ROOT/third_party/slime/.git" ]; then
    git clone https://github.com/THUDM/slime.git "$ROOT/third_party/slime"
fi
cd "$ROOT/third_party/slime"
git checkout "$SLIME_COMMIT"
if git apply --check "$ROOT/patches/slime.patch" 2>/dev/null; then
    git apply "$ROOT/patches/slime.patch"
else
    echo "patches/slime.patch already applied (or the checkout is not at $SLIME_COMMIT)"
fi
pip install -e . --no-deps
pip install -r "$ROOT/requirements.txt"
python3 -c "import slime, litellm, convokit, openai; print('ok')"
