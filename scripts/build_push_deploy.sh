#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/build_push_deploy.sh [options]

Build (and optionally push / deploy) the app, trainer, and UI images.
Defaults to the `latest` tag so you can rebuild quickly without retagging.

Options:
  -t, --tag <tag>         Image tag to use (default: latest)
      --registry <name>   Registry prefix (default: ghcr.io/jimmyrisk)
      --namespace <ns>    Kubernetes namespace for deploy (default: mlops-poc)
      --push              Push the built images
      --deploy            Update Kubernetes deployment/cronjob to the new tag
      --no-build          Skip docker build step (useful after manual build)
  -h, --help              Show this message

The script expects to run from the repository root where Dockerfile.app exists.
EOF
}

TAG="latest"
REGISTRY="${REGISTRY:-ghcr.io/jimmyrisk}"
NAMESPACE="${NAMESPACE:-mlops-poc}"
DO_PUSH=0
DO_DEPLOY=0
DO_BUILD=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--tag)
      shift
      [[ $# -gt 0 ]] || { echo "Error: --tag requires an argument" >&2; exit 1; }
      TAG="$1"
      ;;
    --registry)
      shift
      [[ $# -gt 0 ]] || { echo "Error: --registry requires an argument" >&2; exit 1; }
      REGISTRY="$1"
      ;;
    --namespace)
      shift
      [[ $# -gt 0 ]] || { echo "Error: --namespace requires an argument" >&2; exit 1; }
      NAMESPACE="$1"
      ;;
    --push)
      DO_PUSH=1
      ;;
    --deploy)
      DO_DEPLOY=1
      ;;
    --no-build)
      DO_BUILD=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ $DO_DEPLOY -eq 1 ]]; then
  command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required for --deploy" >&2; exit 1; }

  detect_overlays() {
    local overlays
    overlays=$(kubectl get configmap -n "$NAMESPACE" --no-headers -o custom-columns=NAME:.metadata.name 2>/dev/null | grep -E '^rps-app-(static-)?patch-' || true)
    if [[ -n "$overlays" ]]; then
      echo "Detected ConfigMap overlays mounted into rps-app:"
      echo "$overlays"
      echo "Remove these overlays before deploying a new image."
      echo "You can run ./scripts/clear_configmap_overlays.sh --namespace $NAMESPACE to delete them."
      exit 1
    fi
  }

  detect_overlays
fi

command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 1; }

if [[ ! -f "Dockerfile.app" ]]; then
  echo "Run this script from the repository root (Dockerfile.app not found)." >&2
  exit 1
fi

IMAGES=(
  "rps-app:Dockerfile.app"
  "rps-trainer:Dockerfile.trainer"
  "rps-ui:Dockerfile.ui"
)

if [[ $DO_BUILD -eq 1 ]]; then
  echo "==> Building images with tag $TAG"
  for entry in "${IMAGES[@]}"; do
    NAME="${entry%%:*}"
    DOCKERFILE="${entry##*:}"
    IMAGE_REF="${REGISTRY}/${NAME}:${TAG}"
    echo "-- Building ${IMAGE_REF} using ${DOCKERFILE}"
    docker build -f "$DOCKERFILE" -t "$IMAGE_REF" .
  done
else
  echo "==> Skipping build step as requested"
fi

if [[ $DO_PUSH -eq 1 ]]; then
  echo "==> Pushing images with tag $TAG"
  for entry in "${IMAGES[@]}"; do
    NAME="${entry%%:*}"
    IMAGE_REF="${REGISTRY}/${NAME}:${TAG}"
    echo "-- Pushing ${IMAGE_REF}"
    docker push "$IMAGE_REF"
  done
fi

if [[ $DO_DEPLOY -eq 1 ]]; then
  echo "==> Updating Kubernetes workloads in namespace ${NAMESPACE}"
  kubectl set image deployment/rps-app app="${REGISTRY}/rps-app:${TAG}" -n "$NAMESPACE"
  kubectl set image deployment/rps-ui ui="${REGISTRY}/rps-ui:${TAG}" -n "$NAMESPACE"
  kubectl set image cronjob/rps-trainer trainer="${REGISTRY}/rps-trainer:${TAG}" -n "$NAMESPACE"

  echo "==> Restarting deployments to pull fresh images"
  kubectl rollout restart deployment/rps-app -n "$NAMESPACE"
  kubectl rollout restart deployment/rps-ui -n "$NAMESPACE"

  echo "==> Waiting for rps-app rollout (new image)"
  kubectl rollout status deployment/rps-app -n "$NAMESPACE"
  echo "==> Waiting for rps-ui rollout (new image)"
  kubectl rollout status deployment/rps-ui -n "$NAMESPACE"

  # CRITICAL: Set ASSET_VERSION *after* new image is deployed to bust browser cache
  NEW_ASSET_VERSION="${ASSET_VERSION_OVERRIDE:-$(date +%Y%m%d%H%M%S)}"
  echo "==> Bumping ASSET_VERSION to ${NEW_ASSET_VERSION} (cache bust)"
  kubectl set env deployment/rps-app ASSET_VERSION="${NEW_ASSET_VERSION}" -n "$NAMESPACE"

  echo "==> Waiting for rps-app rollout (asset version update)"
  kubectl rollout status deployment/rps-app -n "$NAMESPACE"
fi

echo "All requested operations completed successfully."