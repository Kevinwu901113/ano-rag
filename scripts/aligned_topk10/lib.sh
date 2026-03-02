#!/usr/bin/env bash

# Shared helpers for aligned topk10 experiment scripts.

set -u

resolve_repo_root() {
  local this_dir
  this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${this_dir}/../.." && pwd
}

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

init_logging() {
  local script_name="$1"
  local root_dir="$2"
  local log_root="$3"

  RUN_ID="$(date +"%Y%m%d_%H%M%S")"
  LOG_DIR="${log_root}/${script_name}/${RUN_ID}"
  mkdir -p "${LOG_DIR}"
  SUMMARY_LOG="${LOG_DIR}/summary.log"

  echo "[$(timestamp)] script=${script_name}" | tee -a "${SUMMARY_LOG}"
  echo "[$(timestamp)] repo_root=${root_dir}" | tee -a "${SUMMARY_LOG}"
  echo "[$(timestamp)] log_dir=${LOG_DIR}" | tee -a "${SUMMARY_LOG}"
}

log_info() {
  local msg="$1"
  echo "[$(timestamp)] ${msg}" | tee -a "${SUMMARY_LOG}"
}

progress_bar() {
  local current="$1"
  local total="$2"
  local label="$3"

  if [[ "${total}" -le 0 ]]; then
    log_info "progress: ${label}"
    return
  fi

  local width=30
  local filled=$(( current * width / total ))
  local empty=$(( width - filled ))
  local percent=$(( current * 100 / total ))

  local bar
  local spaces
  printf -v bar "%*s" "${filled}" ""
  bar="${bar// /=}"
  printf -v spaces "%*s" "${empty}" ""

  log_info "progress [${bar}${spaces}] ${current}/${total} (${percent}%) ${label}"
}

run_task_logged() {
  local tag="$1"
  local logfile="$2"
  shift 2

  mkdir -p "$(dirname "${logfile}")"
  log_info "start ${tag}"

  (
    set -o pipefail
    stdbuf -oL -eL "$@" 2>&1 \
      | awk -v t="${tag}" '{ print "[" t "] " $0; fflush() }' \
      | tee -a "${logfile}"
  )
  local status=$?

  if [[ "${status}" -eq 0 ]]; then
    log_info "done ${tag}"
  else
    log_info "fail ${tag} exit=${status}"
  fi
  return "${status}"
}

wait_pair() {
  local pid_a="$1"
  local tag_a="$2"
  local pid_b="$3"
  local tag_b="$4"

  local status_a=0
  local status_b=0

  set +e
  wait "${pid_a}"
  status_a=$?
  wait "${pid_b}"
  status_b=$?
  set -e

  if [[ "${status_a}" -ne 0 || "${status_b}" -ne 0 ]]; then
    log_info "pair_failed ${tag_a}=${status_a} ${tag_b}=${status_b}"
    return 1
  fi
  log_info "pair_done ${tag_a} ${tag_b}"
  return 0
}

append_if_set() {
  local var_name="$1"
  local flag_name="$2"
  local -n out_ref="$3"
  local value="${!var_name:-}"
  if [[ -n "${value}" ]]; then
    out_ref+=("${flag_name}" "${value}")
  fi
}
