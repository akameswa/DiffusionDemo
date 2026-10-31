#!/usr/bin/env bash
#
# Install the cluster addons the diffusion demo needs on a fresh Magnum cluster.
# Versions match the live diffusion-demo-v2 cluster (verified 2026-10-31):
#   ingress-nginx      chart 4.15.1
#   cert-manager       v1.16.2 (static manifests; not installed via helm there)
#   nvidia gpu-operator chart v25.3.0 (provides the host driver + nvidia.com/gpu)
#
# Usage:
#   KUBECONFIG=~/.kube/diffusion-demo-test.yaml ./bootstrap.sh
#
set -euo pipefail

cd "$(dirname "$0")"

echo "== 1/4 ingress-nginx 4.15.1"
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx >/dev/null
helm repo update >/dev/null
helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx --create-namespace \
  --version 4.15.1 --wait --timeout 10m

echo "== 2/4 cert-manager v1.16.2"
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.16.2/cert-manager.yaml
kubectl -n cert-manager rollout status deploy/cert-manager --timeout=300s
kubectl -n cert-manager rollout status deploy/cert-manager-webhook --timeout=300s

echo "== 3/4 NVIDIA GPU operator v25.3.0 (installs the driver; takes ~10 min)"
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia >/dev/null
helm repo update >/dev/null
helm upgrade --install nvidia-gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator --create-namespace \
  --version v25.3.0 --wait --timeout 30m

echo "== 4/4 Let's Encrypt ClusterIssuer"
kubectl apply -f ../https_cluster_issuer.yaml

echo "== waiting for nvidia.com/gpu to show up on the worker"
for _ in $(seq 1 40); do
  gpus=$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' | grep -c '^1$' || true)
  [ "$gpus" -ge 1 ] && break
  sleep 30
done

kubectl get nodes -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
echo
kubectl get pods -n gpu-operator
