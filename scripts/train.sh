#!/usr/bin/env bash
# Quickstart trainer: wandb logging. Use --tmux to wrap in a tmux session.
# --name is required. The task menu comes from the live task registry (uv run mjx tasks).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

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
      if [[ $# -lt 2 ]]; then
        echo "error: --name needs a value." >&2
        exit 1
      fi
      NAME="$2"
      shift 2
      ;;
    --name=*)
      NAME="${1#--name=}"
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$NAME" ]]; then
  echo "error: --name is required. Name every run, e.g. --name cur5-short-slug." >&2
  echo "usage: bash scripts/train.sh [task] --name <run-name> [--num-envs N] [--cpu] [--tmux]" >&2
  exit 1
fi

if [[ -z "$TASK" ]]; then
  TASK_LIST="$(uv run mjx tasks)"
  if [[ -z "$TASK_LIST" ]]; then
    echo "error: the task registry lists no Pi Plus task." >&2
    exit 1
  fi
  mapfile -t TASKS <<<"$TASK_LIST"
  echo "Select task:"
  select choice in "${TASKS[@]}"; do
    [[ -n "$choice" ]] && TASK="$choice" && break
    echo "Invalid selection."
  done
  if [[ -z "$TASK" ]]; then
    echo "error: no task selected." >&2
    exit 1
  fi
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
  --agent.run-name "$NAME"
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
