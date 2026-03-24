# GLM OCR — Troubleshooting Guide

Quick reference for diagnosing issues with the `glm-ocr` Container App on Azure Container Apps.

## Prerequisites

```bash
# Verify CLI and extension
az --version
az extension show --name containerapp --query version -o tsv

# Set defaults (adjust to your environment)
RG=rg-aca-huggingface
APP=glm-ocr
```

---

## 1. Check Replica Status

```bash
# Is the app running?
az containerapp replica list -n $APP -g $RG -o table
```

If the list is **empty**, the app has scaled to zero (`minReplicas: 0`). The next
incoming request will trigger a cold start — which can take minutes for a large model
like GLM-OCR.

```bash
# Check scale settings
az containerapp show -n $APP -g $RG \
  --query "properties.template.scale" -o json
```

**Fix: prevent scale-to-zero** (keeps one replica always warm):

```bash
az containerapp update -n $APP -g $RG --min-replicas 1
```

---

## 2. View System Logs (Events)

System logs show startup/shutdown events, probe failures, and scaling decisions:

```bash
az containerapp logs show -n $APP -g $RG --type system --tail 50
```

### Common events and what they mean

| Reason | Meaning |
|---|---|
| `ContainerStarted` | Container started successfully |
| `ProbeFailed` | Health/startup probe returned non-200 or timed out |
| `KEDAScaleTargetDeactivated` | KEDA scaled replicas from 1 → 0 (cooldown expired) |
| `ContainerTerminated` / `ManuallyStopped` | Container was stopped (scale-down or crash) |

---

## 3. View Application (Console) Logs

Console logs show stdout/stderr from the vLLM process:

```bash
az containerapp logs show -n $APP -g $RG --type console --tail 100
```

Look for vLLM startup messages like:
- `INFO: Started server process` — vLLM is ready
- `Loading model weights` — model is still loading (not ready for traffic yet)
- `CUDA out of memory` — GPU memory insufficient

---

## 4. Startup Probe Failures (Root Cause of 503s)

Check the current probe configuration:

```bash
az containerapp show -n $APP -g $RG \
  --query "properties.template.containers[0].probes" -o json
```

**Current settings:**
```json
{
  "type": "Startup",
  "httpGet": { "path": "/health", "port": 8000 },
  "initialDelaySeconds": 30,
  "periodSeconds": 10,
  "failureThreshold": 60,
  "timeoutSeconds": 5
}
```

The startup probe allows up to `30 + (60 × 10) = 630 seconds` (~10.5 min) for the
model to load. If vLLM takes longer, the probe exhausts its retries and the
container is killed → 503 for any in-flight requests.

**If you see continuous `ProbeFailed` events**, the model is loading slower than
expected. Increase the failure threshold:

```bash
# Extend startup probe to ~15 minutes
az containerapp update -n $APP -g $RG --yaml - <<'EOF'
properties:
  template:
    containers:
      - name: glm-ocr
        probes:
          - type: Startup
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 10
            failureThreshold: 90
            timeoutSeconds: 10
EOF
```

---

## 5. Ingress & Timeout Settings

```bash
az containerapp show -n $APP -g $RG \
  --query "properties.configuration.ingress" -o json
```

Key fields:
- `targetPort: 8000` — must match vLLM's listening port
- `external: true` — publicly accessible
- `transport: Auto` — HTTP/1.1 or HTTP/2

Container Apps has a **default request timeout of 240 seconds**. For large PDFs
with many regions, OCR can exceed this. Consider setting a longer timeout if
requests are being cut off:

```bash
az containerapp ingress update -n $APP -g $RG --request-timeout 600
```

---

## 6. Resource Allocation

```bash
az containerapp show -n $APP -g $RG \
  --query "properties.template.containers[0].resources" -o json
```

Current: `8 vCPU / 56 GiB`. Verify the workload profile supports GPU if needed:

```bash
az containerapp env show -n aca-huggingface -g $RG \
  --query "properties.workloadProfiles" -o table
```

---

## 7. End-to-End Health Check

Quick smoke test against the running endpoint:

```bash
# Check if vLLM is responding
FQDN=$(az containerapp show -n $APP -g $RG --query "properties.configuration.ingress.fqdn" -o tsv)

# Health endpoint
curl -s "https://$FQDN/health"

# List models
curl -s "https://$FQDN/v1/models" | head -20

# Test OCR (tiny request)
curl -s "https://$FQDN/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/app/GLM-OCR",
    "messages": [{"role": "user", "content": "Text Recognition: hello"}]
  }' | head -20
```

---

## 8. Query Log Analytics with KQL (zsh)

KQL queries contain `|` pipe characters that zsh interprets as shell pipes. **Never
pass KQL inline** — always write the query to a `.kql` file first, then reference it
with the `@file` syntax.

### Setup

```bash
# Find the Log Analytics workspace ID for your Container App environment
ENV_NAME=aca-huggingface
WORKSPACE_ID=$(az containerapp env show -n $ENV_NAME -g $RG \
  --query "properties.appLogsConfiguration.logAnalyticsConfiguration.customerId" -o tsv)
echo "Workspace: $WORKSPACE_ID"
```

### Pattern: write `.kql` file, then query with `@file`

```bash
# Step 1: Write the KQL to a file (heredoc keeps pipes safe)
cat > /tmp/query.kql << 'KQLEOF'
ContainerAppConsoleLogs_CL
| where ContainerAppName_s == "glm-ocr"
| where TimeGenerated > ago(1h)
| project TimeGenerated, Log_s
| order by TimeGenerated desc
| take 20
KQLEOF

# Step 2: Run with @file — the @ prefix tells az CLI to read from a file
az monitor log-analytics query -w "$WORKSPACE_ID" \
  --analytics-query @/tmp/query.kql -o table
```

> **Important:** The heredoc delimiter must be **quoted** (`<< 'KQLEOF'`) to prevent
> zsh from expanding `$` variables inside the KQL. Without quotes, `$APP` would be
> expanded by the shell before the file is written.

### Useful KQL queries

**System events (scaling, probes, restarts):**

```bash
cat > /tmp/kql-system.kql << 'KQLEOF'
ContainerAppSystemLogs_CL
| where ContainerAppName_s == "glm-ocr"
| where TimeGenerated > ago(6h)
| project TimeGenerated, Log_s, Reason_s
| order by TimeGenerated desc
| take 50
KQLEOF
az monitor log-analytics query -w "$WORKSPACE_ID" \
  --analytics-query @/tmp/kql-system.kql -o table
```

**Console errors (vLLM crashes, OOM, encoder cache):**

```bash
cat > /tmp/kql-errors.kql << 'KQLEOF'
ContainerAppConsoleLogs_CL
| where ContainerAppName_s == "glm-ocr"
| where TimeGenerated > ago(6h)
| where Log_s has_any ("ERROR", "ValueError", "OutOfMemory", "killed", "EngineCore crashed")
| where Log_s !has "dump_input"
| project TimeGenerated, Log_s
| order by TimeGenerated desc
| take 30
KQLEOF
az monitor log-analytics query -w "$WORKSPACE_ID" \
  --analytics-query @/tmp/kql-errors.kql -o table
```

**HTTP status codes (find 503s):**

```bash
cat > /tmp/kql-503.kql << 'KQLEOF'
ContainerAppConsoleLogs_CL
| where ContainerAppName_s == "glm-ocr"
| where TimeGenerated > ago(6h)
| where Log_s has_any ("503", "502", "429", "status_code")
| project TimeGenerated, Log_s
| order by TimeGenerated desc
| take 30
KQLEOF
az monitor log-analytics query -w "$WORKSPACE_ID" \
  --analytics-query @/tmp/kql-503.kql -o table
```

**Encoder cache usage (detects overflows that cause 503s mid-request):**

```bash
cat > /tmp/kql-encoder-cache.kql << 'KQLEOF'
ContainerAppConsoleLogs_CL
| where ContainerAppName_s == "glm-ocr"
| where TimeGenerated > ago(6h)
| where Log_s has "encoder_cache_usage"
| project TimeGenerated, Log_s
| order by TimeGenerated desc
| take 10
KQLEOF
az monitor log-analytics query -w "$WORKSPACE_ID" \
  --analytics-query @/tmp/kql-encoder-cache.kql -o table
```

### Output formats

```bash
# Table (human-readable)
az monitor log-analytics query -w "$WORKSPACE_ID" --analytics-query @/tmp/query.kql -o table

# JSON (for scripting)
az monitor log-analytics query -w "$WORKSPACE_ID" --analytics-query @/tmp/query.kql -o json

# Save to file for analysis
az monitor log-analytics query -w "$WORKSPACE_ID" --analytics-query @/tmp/query.kql -o json > results.json
```

---

## 9. Why Did My Requests Return 503?

Based on our session's investigation using KQL (section 8), the 503 errors during
the OCR run were caused by **vLLM encoder cache overflow** — not cold start.

The key log line from `ContainerAppConsoleLogs_CL`:

```
(APIServer pid=1) ValueError: The decoder prompt contains a(n) image item with
length 6105, which exceeds the pre-allocated encoder cache size 6084. Please
reduce the input size or increase the encoder cache size by setting
--limit-mm-per-prompt at startup.
```

This was followed by:

```
(EngineCore pid=55) ERROR 03-23 23:22:19 [core.py:1101] EngineCore encountered a fatal error.
```

### What happened

1. Early pages processed fine — their images fit within vLLM's encoder cache (6084 tokens)
2. A later page produced an image with **6105 tokens** — exceeding the cache by just 21 tokens
3. vLLM raised a `ValueError` and the EngineCore crashed fatally
4. All subsequent in-flight and new requests returned **503 Service Unavailable**
5. The container restarted, startup probes began failing during model reload
6. After model reload completed, the fresh instance was idle → KEDA scaled to 0

### Fix: increase encoder cache or limit image size

**Option A — Increase `--limit-mm-per-prompt`** (server-side, in Dockerfile CMD):

`--limit-mm-per-prompt image=N` tells vLLM the max number of images per prompt to
pre-allocate encoder cache for. The default is `image=1`, which gave a cache of
6084 tokens — just 21 short. Setting `image=2` doubles the cache pool, easily
covering the overflow:

```bash
vllm serve /model/GLM-OCR \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --limit-mm-per-prompt image=2
```

> **Note:** This allocates encoder cache for 2 images worth of tokens per request,
> using more GPU memory. For GLM-OCR (0.9B) on 56 GiB this is fine.

**Option B — Reduce input image DPI** (client-side) to keep token count under the limit:

```bash
# Use a lower DPI (e.g. 200 instead of 300) to produce smaller images
uv run glm-ocr document.pdf --mode layout --dpi 200
```

This is the simplest fix — fewer pixels → fewer encoder tokens. The default 300 DPI
produced 6105 tokens on a dense page; 200 DPI should stay well under 6084.

**Option C — Add retry logic in the OCR client** for transient 503s so that
requests hitting a briefly restarting server recover automatically.

---

## Quick Reference

| Task | Command |
|---|---|
| Replica status | `az containerapp replica list -n $APP -g $RG -o table` |
| System events | `az containerapp logs show -n $APP -g $RG --type system --tail 50` |
| Console logs | `az containerapp logs show -n $APP -g $RG --type console --tail 100` |
| Scale config | `az containerapp show -n $APP -g $RG --query "properties.template.scale" -o json` |
| Probe config | `az containerapp show -n $APP -g $RG --query "properties.template.containers[0].probes" -o json` |
| Ingress config | `az containerapp show -n $APP -g $RG --query "properties.configuration.ingress" -o json` |
| Resource config | `az containerapp show -n $APP -g $RG --query "properties.template.containers[0].resources" -o json` |
| Set min replicas | `az containerapp update -n $APP -g $RG --min-replicas 1` |
| Set request timeout | `az containerapp ingress update -n $APP -g $RG --request-timeout 600` |
| KQL query (zsh-safe) | `cat > /tmp/q.kql << 'EOF'` ... `EOF` then `az monitor log-analytics query -w $WORKSPACE_ID --analytics-query @/tmp/q.kql -o table` |
