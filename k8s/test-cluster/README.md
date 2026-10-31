# Throwaway test cluster (embedding-plot same-origin fix)

A separate Magnum cluster so the live `diffusion-demo-v2` demo is never
touched while the branch `fix/embedding-plot-same-origin` is validated.

Cluster created 2026-10-31: **`diffusion-demo-test`** (id `06bb5f48-5e50-4438-976c-12e831b13283`),
same template and flavours as v2: `kubernetes-1-33-jammy-fixed-labels`,
master `m3.medium`, worker `g3.xl` (1× A100 40 GB, 32 vCPU, 120 GB RAM).

> **Cost:** the worker is a `g3.xl`, so this cluster burns SU at the same rate as
> the production one. Delete it when the test is over (last section).

## 1. Create the cluster (already done; kept for reference/recreation)

```bash
openstack coe cluster create \
  --cluster-template kubernetes-1-33-jammy-fixed-labels \
  --master-count 1 --node-count 1 \
  --master-flavor m3.medium --flavor g3.xl \
  --labels auto_scaling_enabled=true --labels min_node_count=1 --labels max_node_count=5 \
  --fixed-network auto_allocated_network \
  --docker-volume-size 10 --labels boot_volume_size=100 \
  diffusion-demo-test
```

Creation takes ~25–40 min. Watch it with:

```bash
openstack coe cluster show diffusion-demo-test -c status
```

## 2. Get a kubeconfig

```bash
openstack coe cluster config diffusion-demo-test --output ~/.kube/diffusion-demo-test.yaml
export KUBECONFIG=~/.kube/diffusion-demo-test.yaml
kubectl get nodes
```

## 3. Install the addons (versions match production)

```bash
./bootstrap.sh
```

Installs ingress-nginx 4.15.1, cert-manager v1.16.2, the NVIDIA GPU operator
v25.3.0 (this is what gives the node its driver and `nvidia.com/gpu`), and the
Let's Encrypt `ClusterIssuer`. No MIG: production runs `nvidia.com/mig.config=all-disabled`,
so `k8s/mig-node-group-labels.yaml` is not applied.

## 4. Build and push the image

The Dockerfile expects the repo's parent directory as build context:

```bash
cd /Users/akameswa/diffusiondemo
docker build -f DiffusionDemo/Dockerfile -t akameswa/diffusion-demo-cuda-slim:embedding-same-origin .
docker push akameswa/diffusion-demo-cuda-slim:embedding-same-origin
```

## 5. Deploy the demo on a DNS-free hostname

The ingress load balancer gets its own public VIP; `sslip.io` maps a hostname to
it, so no DNS record is needed and cert-manager can still do HTTP-01:

```bash
VIP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller \
        -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
TEST_HOST="$VIP.sslip.io"
echo "https://$TEST_HOST"

kubectl apply -f service.yaml
kubectl apply -f deployment.yaml
sed "s|TEST_HOST|$TEST_HOST|g" ingress.yaml | kubectl apply -f -

kubectl rollout status deployment/diffusion-demo-test-deployment --timeout=900s
kubectl get certificate diffusion-demo-test-tls     # READY=True
```

## 6. Acceptance tests

```bash
# a) aux mount attached — must print 1 (0 = mount did not attach)
curl -s https://$TEST_HOST/aux/plot/x | grep -c "Preparing the embedding plot"

# b) demo serves
curl -s -o /dev/null -w '%{http_code}\n' https://$TEST_HOST/
```

In the browser:

1. Open the demo, switch to **Embeddings** — the 3D plot renders.
2. Add/remove a word, change an axis — the iframe updates each time.
3. **The regression test:** leave the tab on Embeddings, then
   `kubectl rollout restart deployment/diffusion-demo-test-deployment`.
   The plot returns **by itself** within ~30 s (no hard refresh, no pod delete).
4. DevTools → Elements: the iframe src is `https://$TEST_HOST/aux/plot/<hash>`.

## 7. Delete the cluster (stops the SU burn)

```bash
openstack coe cluster delete diffusion-demo-test
```

## Promote to production once satisfied

```bash
# from the v2 cluster's kubeconfig
kubectl set image deployment/diffusion-demo-v2-deployment \
  diffusion-demo-v2=akameswa/diffusion-demo-cuda-slim:embedding-same-origin
kubectl rollout status deployment/diffusion-demo-v2-deployment --timeout=900s
```
