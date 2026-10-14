# Deployment Evidence

This document records the live validation evidence for the heart disease prediction MLOps project.

> **Evidence is historical:** The Docker and Kind checks below were performed with an earlier model artifact, before the preprocessing pipeline added categorical one-hot encoding. The recorded prediction probability (`0.444574`) does not match the current local model response (`0.455364`). Rebuild the image and repeat the Docker/Kubernetes checks before treating this as evidence for the current artifact or the BITS lab environment. The Kind Service's recorded external IP was `<pending>`; API access was through `kubectl port-forward`, not a public URL.

## 1. Docker Evidence

### Image build

Command used:

```bash
docker build -t heart-disease-api:latest .
```

Evidence:

- Docker image successfully built as `heart-disease-api:latest`
- Output showed the image built without Docker build errors

### Container run

Command used:

```bash
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest
```

Evidence:

- Container is running:
  - `heart-disease-api`
  - Status: `Up`

### Health endpoint

Command used:

```bash
curl.exe http://localhost:8000/health
```

Response:

```json
{"status":"ok"}
```

### Prediction endpoint

Example request payload:

```json
{
  "age": 63,
  "sex": 1,
  "cp": 1,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 2,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 3,
  "ca": 0,
  "thal": 6
}
```

Response:

```json
{
  "features": {
    "age": 63.0,
    "sex": 1.0,
    "cp": 1.0,
    "trestbps": 145.0,
    "chol": 233.0,
    "fbs": 1.0,
    "restecg": 2.0,
    "thalach": 150.0,
    "exang": 0.0,
    "oldpeak": 2.3,
    "slope": 3.0,
    "ca": 0.0,
    "thal": 6.0
  },
  "prediction": 0,
  "probability": 0.4445743342011674,
  "risk_label": "Low risk"
}
```

## 2. Kubernetes Evidence

### Local cluster creation

Command used:

```bash
kind create cluster --name heartdisease
```

Evidence:

- Local Kubernetes cluster created successfully
- Cluster name: `heartdisease`
- Kubernetes context set to `kind-heartdisease`

### Deployment

Manifest used:

```yaml
kubernetes/deployment.yaml
```

Command used:

```bash
kubectl apply -f kubernetes/deployment.yaml
```

Evidence:

- Deployment created: `heart-disease-api`
- Service created: `heart-disease-api`

### Pod status

Command used:

```bash
kubectl get pods -o wide
```

Evidence:

```text
NAME                                 READY   STATUS    RESTARTS   AGE   IP           NODE                         NOMINATED NODE   READINESS GATES
heart-disease-api-67ccf87445-4pcqv   1/1     Running   0          37s   10.244.0.6   heartdisease-control-plane   <none>           <none>
heart-disease-api-67ccf87445-5fzpc   1/1     Running   0          37s   10.244.0.5   heartdisease-control-plane   <none>           <none>
```

### Service status

Command used:

```bash
kubectl get svc -o wide
```

Evidence:

```text
NAME                TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)        AGE     SELECTOR
heart-disease-api   LoadBalancer   10.96.249.138   <pending>     80:32545/TCP   38s     app=heart-disease-api
```

### Live API validation in Kubernetes

Command used:

```bash
kubectl port-forward svc/heart-disease-api 8000:80 --address 127.0.0.1
```

Then validated via:

```bash
curl http://localhost:8000/health
```

Response:

```json
{"status":"ok"}
```

Prediction response:

```json
{
  "features": {
    "age": 63.0,
    "sex": 1.0,
    "cp": 1.0,
    "trestbps": 145.0,
    "chol": 233.0,
    "fbs": 1.0,
    "restecg": 2.0,
    "thalach": 150.0,
    "exang": 0.0,
    "oldpeak": 2.3,
    "slope": 3.0,
    "ca": 0.0,
    "thal": 6.0
  },
  "prediction": 0,
  "probability": 0.4445743342011674,
  "risk_label": "Low risk"
}
```

## 3. Additional Local Kubernetes Screenshot Evidence

The following screenshots capture a local Kubernetes deployment and health check:

- `screenshots/Kubernetes_Deployment1.png` — the `heart-disease-api` Deployment is available; its pod is `1/1 Running`; the Service is `NodePort` on `8000:30080`; and `http://127.0.0.1:8000/health` returned HTTP 200 with `{"status":"ok"}`.
- `screenshots/Kubernetes_Deployment2.png` — deployment, pod and Service status, followed by a successful FastAPI health check (`HTTP Status: 200`).

The command visible in the screenshots uses Kubernetes context `kind-heartdisease`. This evidence therefore documents a local Kind cluster, not Minikube. The displayed deployment has one replica and uses a NodePort Service. The screenshots demonstrate the health endpoint; they do not show a Kubernetes `/predict` request.

The screenshot-matching manifest is `kubernetes/local-nodeport.yaml`. It configures the `heart-disease-api:latest` image, one replica, a readiness probe on `/health`, and a NodePort Service on port 30080. The repository's `kubernetes/deployment.yaml` remains a separate two-replica LoadBalancer manifest.

## 4. Conclusion

The evidence includes BITS lab Docker validation of `/health`, `/predict` and `/metrics`, as well as local Kubernetes health validation documented in the screenshots above. The original Docker and Kind records in sections 1 and 2 are historical checks using an earlier model artifact; the additional Kubernetes screenshots establish a successful health check for the deployment shown there.
