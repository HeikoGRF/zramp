#!/usr/bin/env bash
# Source this file before invoking legacy Python entry points directly.

legacy_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
zramp_repo_root=$(cd -- "$legacy_script_dir/../.." && pwd -P)
zramp_legacy_root="$zramp_repo_root/project_archive/code/legacy/campaign_implementations"
zramp_shared_root="$zramp_repo_root/paper_experiments/code/shared_runtime"

export PYTHONPATH="$zramp_legacy_root:$zramp_shared_root${PYTHONPATH:+:$PYTHONPATH}"
unset legacy_script_dir zramp_repo_root zramp_legacy_root zramp_shared_root

