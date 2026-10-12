#!/usr/bin/env bash
set -euo pipefail
umask 077
cd /workspace
trap 'echo failed > /workspace/setup.status' ERR
echo installing > setup.status
export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq bubblewrap ripgrep
else
  yum install -y bubblewrap
fi
python3 -m pip install --disable-pip-version-check 'atrex-gateway-client==0.14.3' --index-url https://artlab.alibaba-inc.com/1/pypi/simple
python3 -m pip install --upgrade pip --index-url https://artlab.alibaba-inc.com/1/pypi/simple
python3 -m pip install --disable-pip-version-check 'torch==2.6.0+cpu' --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://artlab.alibaba-inc.com/1/pypi/simple
npm install -g @anthropic-ai/claude-code
python3 -c 'import torch; print("torch",torch.__version__)'
claude --version
agate --version
bwrap --ro-bind / / --proc /proc --dev /dev --unshare-pid -- bash -c 'echo bwrap-ok'
echo ready > setup.status
