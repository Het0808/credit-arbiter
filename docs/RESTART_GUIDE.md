# Restart Guide — Halcyon Credit (Credit Arbiter)

Paste this whole file to an assistant (or follow it yourself) to bring the app back
up after you Stopped or Terminated the AWS server. It contains everything needed.

---

## Context (what this system is)
An agentic loan-underwriting app deployed on **one AWS EC2 instance** as **4 Docker
containers** via `docker compose`:

- **web** — nginx serving the frontend, proxies `/api` → api (port **80**, public)
- **api** — FastAPI: trained ML model scoring + Qdrant vector RAG + OpenRouter LLM explanations
- **postgres** — application database
- **qdrant** — vector store for policy retrieval

Key facts:
- **AWS account:** `851276831339` · **Region:** `us-east-1`
- **EC2:** name `credit-arbiter`, type **t3.medium**, Ubuntu 24.04
- **Security group:** `credit-arbiter-sg` (SSH 22 restricted, HTTP 80 open)
- **Key pair:** `credit-key` → private key file `credit-key.pem`
- **GitHub repo:** https://github.com/Het0808/credit-arbiter (branch `main`)
- **On the server:** code at `~/credit-arbiter`; secrets in `~/credit-arbiter/.env`
- **Runtime switches (set in `docker-compose.yml`):** `RISK_SCORER=ml`, `RETRIEVER=vector`, `LLM_PROVIDER=openrouter`
- **Trained model:** `models/production/risk_model_v1.pkl` (gitignored; lives on the
  instance disk — it PERSISTS across Stop/Start, so no re-upload needed after a Stop).

### Secrets you must have (rotate the old ones!)
- `OPENROUTER_API_KEY` — from https://openrouter.ai (Keys)
- `JWT_SECRET_KEY` — any long random string (`openssl rand -hex 32`)

---

## CASE A — You STOPPED the instance (fastest; data preserved)

### A1. Start the instance
AWS Console → **EC2 → Instances** → select `credit-arbiter` → **Instance state ▸ Start instance**.
Wait for **Running / 2 status checks passed**, then copy the **new Public IPv4** — call it `NEW_IP`
(⚠️ the IP changes on every Stop/Start unless you attached an Elastic IP).

### A2. Allow SSH from your current machine
The SSH rule is locked to an old IP. Get your machine's public IP and update the rule:
```bash
curl -s https://checkip.amazonaws.com          # = MY_IP
```
Console → **EC2 → Security Groups → `credit-arbiter-sg` → Inbound rules → Edit** →
the SSH (port 22) rule → Source = `MY_IP/32` → Save.

### A3. Locate your key file
You need `credit-key.pem` with strict permissions. If it's not already safe, put it in `~/.ssh`:
```bash
chmod 600 /path/to/credit-key.pem
```
> ⚠️ If you lost the `.pem`, you cannot SSH in. You'd have to Terminate + redeploy (Case B)
> with a new key pair.

### A4. Bring the containers back up (they don't auto-start)
```bash
ssh -i /path/to/credit-key.pem ubuntu@NEW_IP
cd credit-arbiter
sudo docker compose up -d        # starts web + api + postgres + qdrant
sudo docker compose ps           # all should be "running"/"healthy"
exit
```

### A5. Verify
```bash
curl http://NEW_IP/api/health    # -> {"status":"ok",...}
```
Open **http://NEW_IP** in a browser.

---

## CASE B — You TERMINATED the instance (full redeploy)

### B1. Launch a new instance (Console)
EC2 → **Launch instance**: name `credit-arbiter`, **Ubuntu Server 24.04 LTS**,
**t3.medium**, **30 GiB** gp3. Create/download a key pair `credit-key.pem`.
Security group inbound: **SSH 22** from your IP (`curl -s https://checkip.amazonaws.com`),
**HTTP 80** from anywhere. Launch → copy **Public IPv4** = `NEW_IP`.

### B2. Install Docker + clone the repo
```bash
chmod 600 /path/to/credit-key.pem
ssh -i /path/to/credit-key.pem ubuntu@NEW_IP
sudo apt-get update -y
sudo apt-get install -y docker.io docker-compose-v2 git
sudo systemctl enable --now docker
git clone https://github.com/Het0808/credit-arbiter.git
cd credit-arbiter
exit
```

### B3. Upload the trained model (gitignored — run on YOUR laptop)
The model isn't in git. Copy it up (or skip → app falls back to rule-based scoring):
```bash
scp -i /path/to/credit-key.pem /path/to/risk_model_v1.pkl \
    ubuntu@NEW_IP:~/credit-arbiter/models/production/
```
> If you don't have the `.pkl`, regenerate it on a machine with the training data:
> `python -m src.risk_model.train_hardened`.

### B4. Create the secrets file + start everything
```bash
ssh -i /path/to/credit-key.pem ubuntu@NEW_IP
cd credit-arbiter
cat > .env <<EOF
OPENROUTER_API_KEY=YOUR_OPENROUTER_KEY
JWT_SECRET_KEY=$(openssl rand -hex 32)
LLM_PROVIDER=openrouter
OPENROUTER_MODEL=openai/gpt-4o-mini
EOF
chmod 600 .env
sudo docker compose up -d --build      # first build ~4-6 min
sudo docker compose ps
exit
```

### B5. Verify
```bash
curl http://NEW_IP/api/health
```
Open **http://NEW_IP**.

---

## Using the app
Open `http://NEW_IP` → **Create an account** (DB is fresh unless preserved) →
**＋ New Application** → fill the form (include `EXT_SOURCE_1/2/3` bureau scores) → **Assess**.
Score comes from the trained model, policy retrieval is semantic (Qdrant), explanation is LLM-written.

To see an **Approve** (not a human referral): the applicant needs good bureau scores
(low model risk) **+** uploaded documents **+** an ID that passes the mock regulatory check.
Otherwise it correctly returns **Refer/Decline**.

## Deploy code updates
```bash
ssh -i /path/to/credit-key.pem ubuntu@NEW_IP
cd credit-arbiter && git pull && sudo docker compose up -d --build
```

## Troubleshooting
```bash
sudo docker compose logs -f api     # backend errors
sudo docker compose logs -f web     # nginx
sudo docker compose restart api
sudo docker compose down            # stop all (keeps volumes/data)
sudo docker compose up -d           # start all
```
- **api unhealthy on boot:** it waits for Postgres; give it ~15s, check `logs api`.
- **LLM explanations show `generator: deterministic`:** `OPENROUTER_API_KEY` missing/invalid in `.env` → fix → `docker compose up -d`.
- **Model not used (`scorer: rule_based`):** the `.pkl` isn't in `models/production/` → re-upload (B3).

## Optional: auto-start containers after a Stop/Start
Add `restart: unless-stopped` under the `api` and `web` services in `docker-compose.yml`,
commit, and redeploy — then Case A skips the `docker compose up -d` step.

## Cost control
- **Stop** the instance when idle (compute → ~$0; ~$2-3/mo for the disk).
- **Terminate** to end all charges (then Case B to redeploy).
