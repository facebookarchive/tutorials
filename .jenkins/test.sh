#!/bin/bash

set -ex

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/.. && pwd)

# Figure out which Python to use
PYTHON="python"
if [ -n "$BUILD_ENVIRONMENT" ]; then
  if [[ "$BUILD_ENVIRONMENT" == py2* ]]; then
    PYTHON="python2"
  elif [[ "$BUILD_ENVIRONMENT" == py3* ]]; then
    PYTHON="python3"
  fi
fi

cd "$ROOT_DIR"
python tutorials_to_script_converter.py
git status
if git diff --quiet HEAD; then
  echo "Source tree is clean."
else
  echo "After running a tutorial -> script sync there are changes. This probably means you edited an ipython notebook without a proper sync to a script. Please see caffe2/python/tutorials/README.md for more information"
  if [ "$exit_code" -eq 0 ]; then
    exit_code=1
  fi
fi
