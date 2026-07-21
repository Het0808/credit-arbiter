# Deploy to AWS — full stack on one EC2 box (Docker Compose)

Runs the **entire** stack — frontend + FastAPI (trained model + vector RAG + LLM) +
PostgreSQL + Qdrant — as four containers on a single EC2 instance. Beginner-friendly.
Substitute the `ALL-CAPS` placeholders.

## What you'll run
```
web (nginx + frontend)  →  api (FastAPI: ML + Qdrant RAG + OpenRouter LLM)
                                   ├─ postgres   (app database)
                                   └─ qdrant     (vector store)
```

---

## Step 1 — Launch the server (AWS Console)
1. EC2 → **Launch instance**. Name `credit-arbiter`.
2. AMI **Ubuntu Server 24.04 LTS**; type **t3.medium** (4 GB RAM — needed for the ML libs + Qdrant).
3. Create/download a key pair `credit-key.pem`.
4. Security group inbound: **SSH 22** from *My IP*, **HTTP 80** from *Anywhere*.
5. Storage **30 GB gp3**. Launch → copy the **Public IPv4** = `SERVER_IP`.

## Step 2 — Connect
```bash
chmod 400 credit-key.pem
ssh -i credit-key.pem ubuntu@SERVER_IP
```

## Step 3 — Install Docker
```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
sudo usermod -aG docker ubuntu && newgrp docker
```

## Step 4 — Get the code
```bash
git clone https://github.com/Het0808/credit-arbiter.git
cd credit-arbiter
```

## Step 5 — Provide the trained model (it's gitignored, so not in the clone)
The image bakes in `models/production/risk_model_v1.pkl`. Copy it up from your laptop
(run this on **your machine**, not the server):
```bash
scp -i credit-key.pem models/production/risk_model_v1.pkl \
    ubuntu@SERVER_IP:~/credit-arbiter/models/production/
```
> If you skip this, the app still runs but falls back to the **rule-based** scorer
> (it won't use the trained model).

## Step 6 — Configure secrets
```bash
# on the server, in ~/credit-arbiter
cat > .env <<EOF
OPENROUTER_API_KEY=sk-or-...your-key...
JWT_SECRET_KEY=$(openssl rand -hex 32)
OPENROUTER_MODEL=openai/gpt-4o-mini
LLM_PROVIDER=openrouter
EOF
```
`docker compose` auto-reads `.env` for these values. Leave `OPENROUTER_API_KEY` blank to
run with the deterministic (non-LLM) explanation generator.

## Step 7 — Build & start everything
```bash
docker compose up -d --build      # first build ~3–5 min (installs ML/RAG/LLM deps)
docker compose ps                 # all services should be "running"/"healthy"
```

## Step 8 — Use it
Open **http://SERVER_IP** → register → **New Application** → Assess.
- Score comes from the **trained model** (`RISK_SCORER=ml`).
- Policy retrieval is **semantic** via Qdrant (`RETRIEVER=vector`).
- Explanations are **LLM-generated** via OpenRouter (if the key is set).

Smoke-check the API: `curl http://SERVER_IP/api/health` → `{"status":"ok",...}`.

## Updating later
```bash
cd ~/credit-arbiter && git pull && docker compose up -d --build
```

## Logs / troubleshooting
```bash
docker compose logs -f api        # backend logs
docker compose logs -f web        # nginx
docker compose restart api
```

## 💰 Cost & teardown
- t3.medium ≈ $30/mo if left on; **Stop** it when idle, **Terminate** to end all charges.
- Data persists in the `pgdata` / `qdrantdata` Docker volumes across restarts.

## Notes
- **Managed alternative:** swap the `postgres` container for **Amazon RDS** (set `DATABASE_URL`
  to the RDS endpoint) and/or Qdrant Cloud (set `QDRANT_URL`) — no code changes needed.
- **HTTPS:** put an AWS ALB or Caddy/Certbot in front of `web` for TLS + a domain.
- **Never** commit `.env` or bake `OPENROUTER_API_KEY` into an image.
