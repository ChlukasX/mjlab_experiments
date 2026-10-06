#!/usr/bin/env bash
# Quickstart trainer: wandb logging. Use --tmux to wrap in a tmux session.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

TASKS=(
  "Mjlab-Piplus-Upright"
  "Mjlab-Piplus-Locomotion"
  "Mjlab-Piplus-Locomotion-Arms"
  "Mjlab-Piplus-NonInertial"
  "Mjlab-Piplus-Platform"
  "Mjlab-Piplus-Ball"
  "Mjlab-Piplus-Ball-Arms"
  "Mjlab-Piplus-Ball-Small"
  "Mjlab-Piplus-Ball-Small-Turf"
  "Mjlab-Piplus-Ball-Mount"
  "Mjlab-Piplus-Ball-Mount-Flat"
  "Mjlab-Piplus-Ball-Mount-Size1"
  "Mjlab-Piplus-Ball-Mount-Size1-Flat"
  "Mjlab-Piplus-Ball-Balance-Size1"
  "Mjlab-Piplus-Ball-MountBalance"
  "Mjlab-Piplus-Ball-MountBalance-Mix30"
  "Mjlab-Piplus-Ball-MountBalance-Mix70"
  "Mjlab-Piplus-Ball-MountBalance-Mix0"
  "Mjlab-Piplus-Ball-MountBalance-Mix100"
)

TASK=""
NUM_ENVS=""
CPU=0
TMUX=0
NAME=""
EXTRA_ARGS=()

if [[ $# -gt 0 && "$1" != --* ]]; then
  TASK="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)
      CPU=1
      shift
      ;;
    --num-envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --tmux)
      TMUX=1
      shift
      ;;
    --name)
      NAME="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$TASK" ]]; then
  echo "Select task:"
  select choice in "${TASKS[@]}"; do
    [[ -n "$choice" ]] && TASK="$choice" && break
    echo "Invalid selection."
  done
fi

if [[ -z "$NUM_ENVS" ]]; then
  NUM_ENVS=64
  [[ "$CPU" -eq 0 ]] && NUM_ENVS=4096
fi

if [[ -z "${WANDB_API_KEY:-}" ]] && ! uv run wandb login --verify >/dev/null 2>&1; then
  echo "error: not logged in to wandb. Set WANDB_API_KEY or run 'uv run wandb login'." >&2
  exit 1
fi

if [[ "$CPU" -eq 1 ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

TRAIN_CMD=(uv run train "$TASK"
  --agent.logger wandb
  --agent.wandb-project "$TASK"
  --env.scene.num-envs "$NUM_ENVS"
  ${NAME:+--agent.run-name "$NAME"}
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

if [[ "$TMUX" -eq 1 ]]; then
  SESSION="mjlab-train"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session '$SESSION' already running: tmux attach -t $SESSION"
    exit 1
  fi

  tmux new-session -d -s "$SESSION" -n train
  tmux send-keys -t "$SESSION:train" "$(printf '%q ' "${TRAIN_CMD[@]}")" C-m

  tmux new-window -t "$SESSION" -n viewer
  tmux send-keys -t "$SESSION:viewer" "# uv run play $TASK --checkpoint-file logs/rsl_rl/<experiment>/<run>/model_N.pt --viewer viser --num-envs 1" C-m

  tmux attach -t "$SESSION"
else
  "${TRAIN_CMD[@]}"
fi
