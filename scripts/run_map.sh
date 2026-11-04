#!/usr/bin/env bash
set -eo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${repo_dir}/.map_runtime"
source /opt/ros/jazzy/setup.bash

missing=0
for package in rtabmap_sync rtabmap_odom rtabmap_slam octomap_msgs; do
  if ! ros2 pkg prefix "${package}" >/dev/null 2>&1; then
    missing=1
  fi
done
if (( missing )); then
  if [[ ! -f "${runtime_dir}/.ready" ]]; then
    mkdir -p "${runtime_dir}/debs"
    simulation="$(apt-get -s --no-install-recommends install \
      ros-jazzy-rtabmap-sync ros-jazzy-rtabmap-odom ros-jazzy-rtabmap-slam ros-jazzy-octomap-msgs)"
    mapfile -t packages < <(printf '%s\n' "${simulation}" | awk '$1 == "Inst" {print $2}')
    if (( ${#packages[@]} == 0 )); then
      echo "RTAB-Map packages are unavailable; check the ROS Jazzy apt repository." >&2
      exit 1
    fi
    (
      cd "${runtime_dir}/debs"
      missing_packages=()
      for package in "${packages[@]}"; do
        cached=0
        for deb in "${package}"_*.deb; do
          [[ -f "${deb}" ]] || continue
          if dpkg-deb --contents "${deb}" >/dev/null 2>&1; then
            cached=1
          else
            rm -f "${deb}"
          fi
        done
        if (( ! cached )); then
          missing_packages+=("${package}")
        fi
      done
      if (( ${#missing_packages[@]} )); then
        apt-get download "${missing_packages[@]}"
      fi
      for deb in ./*.deb; do
        dpkg-deb -x "${deb}" "${runtime_dir}"
      done
      rm -f ./*.deb
    )
    touch "${runtime_dir}/.ready"
  fi
  export AMENT_PREFIX_PATH="${runtime_dir}/opt/ros/jazzy:${AMENT_PREFIX_PATH:-}"
  export LD_LIBRARY_PATH="${runtime_dir}/opt/ros/jazzy/lib:${runtime_dir}/opt/ros/jazzy/lib/x86_64-linux-gnu:${runtime_dir}/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="${runtime_dir}/opt/ros/jazzy/lib/python3.12/site-packages:${PYTHONPATH:-}"
fi

if [[ ! -f "${repo_dir}/install/setup.bash" ]]; then
  (cd "${repo_dir}" && colcon build --symlink-install)
fi
source "${repo_dir}/install/setup.bash"
if [[ ${1:-} == "--setup-only" ]]; then
  exit 0
fi

mode=map
if [[ ${1:-} == "slam" || ${1:-} == "kick" || ${1:-} == "map" ]]; then
  mode="$1"
  shift
fi
if [[ ! -d "${runtime_dir}/python/toppra" || ! -d "${runtime_dir}/python/cyipopt" ]]; then
  python3 -m pip install --no-deps --target "${runtime_dir}/python" toppra==0.6.10 cyipopt==1.7.0
fi
export PYTHONPATH="${runtime_dir}/python:${PYTHONPATH:-}"
exec ros2 launch multirotor_cmd run.py "mode:=${mode}" "$@"
