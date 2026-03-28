#!/bin/bash
# Script d'installation pour vast.ai
# Qwen3-TTS (7860) + vLLM/Mistral (8000) + Dashboard statut (3000)
# Compatible Alexandria mode TTS externe

echo "================================================"
echo "  Qwen3-TTS + vLLM + Dashboard — vast.ai"
echo "================================================"

apt-get update -y
apt-get install -y \
    python3-pip python3-venv python3.10-venv \
    git wget curl supervisor ffmpeg \
    software-properties-common

mkdir -p /workspace/logs

# ════════════════════════════════════════════════════════════
# 1. QWEN3-TTS (Gradio — port 7860)
# ════════════════════════════════════════════════════════════
echo ""
echo ">>> Installation de Qwen3-TTS..."

cd /workspace
git clone https://github.com/SUP3RMASS1VE/Qwen3-TTS.git
cd /workspace/Qwen3-TTS

python3 -m venv /workspace/tts_env
source /workspace/tts_env/bin/activate
pip install --upgrade pip
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r /workspace/Qwen3-TTS/requirements.txt
deactivate

# ════════════════════════════════════════════════════════════
# 2. vLLM + Mistral (OpenAI-compatible — port 8000)
# ════════════════════════════════════════════════════════════
echo ""
echo ">>> Installation de vLLM..."

python3 -m venv /workspace/llm_env
source /workspace/llm_env/bin/activate
pip install --upgrade pip
pip install numpy==1.26.4
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.4.2 transformers==4.40.2
deactivate

# ── Téléchargement de Mistral ──────────────────────────────
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN non defini - telechargement Mistral ignore."
else
    echo ""
    echo ">>> Telechargement de Mistral-7B-Instruct-v0.3..."
    mkdir -p /workspace/models/mistral
    cd /workspace/models/mistral
    for f in \
        config.json generation_config.json special_tokens_map.json \
        tokenizer.json tokenizer.model model.safetensors.index.json \
        model-00001-of-00003.safetensors \
        model-00002-of-00003.safetensors \
        model-00003-of-00003.safetensors
    do
        wget -q --show-progress \
            --header="Authorization: Bearer $HF_TOKEN" \
            "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/$f" \
            -O "$f" && echo "  OK $f" || echo "  FAIL $f"
    done
fi

# ════════════════════════════════════════════════════════════
# 3. DASHBOARD DE STATUT (Python/FastAPI — port 3000)
#    Vérifie en temps réel si chaque service répond
#    et affiche les dernières lignes de logs
# ════════════════════════════════════════════════════════════
echo ""
echo ">>> Installation du dashboard de statut..."

python3 -m venv /workspace/dashboard_env
source /workspace/dashboard_env/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn httpx
deactivate

# ── Serveur Python du dashboard ───────────────────────────
cat > /workspace/dashboard.py << 'PYEOF'
import asyncio, subprocess, os, time
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import httpx

app = FastAPI()

SERVICES = [
    {
        "name": "Qwen3-TTS",
        "port": 7860,
        "check_url": "http://localhost:7860/",
        "log": "/workspace/logs/qwen3-tts.log",
        "err": "/workspace/logs/qwen3-tts.err",
        "supervisor": "qwen3-tts",
        "color": "#4f8ef7",
        "desc": "TTS Gradio — Alexandria TTS URL"
    },
    {
        "name": "vLLM (Mistral)",
        "port": 8000,
        "check_url": "http://localhost:8000/health",
        "log": "/workspace/logs/vllm.log",
        "err": "/workspace/logs/vllm.err",
        "supervisor": "vllm",
        "color": "#7c5cbf",
        "desc": "LLM OpenAI-compatible — Alexandria LLM URL"
    },
]

def tail_log(path, n=30):
    if not os.path.exists(path):
        return "(pas encore de logs)"
    try:
        result = subprocess.run(["tail", f"-{n}", path], capture_output=True, text=True)
        return result.stdout or "(vide)"
    except Exception as e:
        return str(e)

def supervisor_status(name):
    try:
        r = subprocess.run(["supervisorctl", "status", name],
                           capture_output=True, text=True, timeout=3)
        return r.stdout.strip()
    except Exception:
        return "supervisorctl indisponible"

async def check_service(svc):
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(svc["check_url"])
            return r.status_code < 500
    except Exception:
        return False

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="10">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Alexandria — Statut serveur vast.ai</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; padding: 24px; }}
  h1 {{ font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }}
  .subtitle {{ font-size: 0.85rem; color: #64748b; margin-bottom: 28px; }}
  .subtitle span {{ color: #38bdf8; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
  @media (max-width: 700px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .card {{ background: #1e2130; border-radius: 12px; padding: 20px; border: 1px solid #2d3348; }}
  .card-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }}
  .dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
  .dot-ok {{ background: #22c55e; box-shadow: 0 0 8px #22c55e88; }}
  .dot-fail {{ background: #ef4444; box-shadow: 0 0 8px #ef444488; }}
  .dot-warn {{ background: #f59e0b; box-shadow: 0 0 8px #f59e0b88; }}
  .svc-name {{ font-weight: 700; font-size: 1rem; }}
  .svc-desc {{ font-size: 0.78rem; color: #64748b; margin-top: 2px; }}
  .badge {{ font-size: 0.72rem; padding: 2px 8px; border-radius: 20px; font-weight: 600; margin-left: auto; }}
  .badge-ok {{ background: #14532d; color: #86efac; }}
  .badge-fail {{ background: #450a0a; color: #fca5a5; }}
  .supervisor-line {{ font-size: 0.75rem; color: #94a3b8; font-family: monospace; margin-bottom: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .log-label {{ font-size: 0.72rem; color: #475569; margin-bottom: 4px; text-transform: uppercase; letter-spacing: .05em; }}
  .log-box {{ background: #0f1117; border-radius: 6px; padding: 10px; font-family: monospace; font-size: 0.72rem; color: #94a3b8; max-height: 160px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; border: 1px solid #1e2130; }}
  .info-card {{ background: #1e2130; border-radius: 12px; padding: 20px; border: 1px solid #2d3348; margin-bottom: 16px; }}
  .info-card h2 {{ font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 12px; }}
  .info-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; font-size: 0.85rem; }}
  .info-key {{ color: #64748b; width: 130px; flex-shrink: 0; }}
  .info-val {{ color: #e2e8f0; font-family: monospace; background: #0f1117; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; }}
  .refresh {{ font-size: 0.75rem; color: #334155; text-align: right; margin-top: 16px; }}
</style>
</head>
<body>
<h1>Alexandria — Serveurs vast.ai</h1>
<p class="subtitle">Mise à jour automatique toutes les 10s — <span>{time}</span></p>

<div class="info-card">
  <h2>Configuration Alexandria (Setup)</h2>
  <div class="info-row"><span class="info-key">TTS Mode</span><span class="info-val">external</span></div>
  <div class="info-row"><span class="info-key">TTS Server URL</span><span class="info-val">http://&lt;vast-ip&gt;:&lt;port-ext-7860&gt;</span></div>
  <div class="info-row"><span class="info-key">LLM Base URL</span><span class="info-val">http://&lt;vast-ip&gt;:&lt;port-ext-8000&gt;/v1</span></div>
  <div class="info-row"><span class="info-key">LLM API Key</span><span class="info-val">local</span></div>
  <div class="info-row"><span class="info-key">LLM Model</span><span class="info-val">mistralai/Mistral-7B-Instruct-v0.3</span></div>
</div>

<div class="grid">
{cards}
</div>

<p class="refresh">Page rafraîchie automatiquement • {time}</p>
</body></html>"""

CARD_TEMPLATE = """
<div class="card">
  <div class="card-header">
    <div class="dot {dot_class}"></div>
    <div>
      <div class="svc-name">{name}</div>
      <div class="svc-desc">{desc}</div>
    </div>
    <span class="badge {badge_class}">{status_text}</span>
  </div>
  <div class="supervisor-line">{supervisor}</div>
  <div class="log-label">Derniers logs</div>
  <div class="log-box">{log}</div>
</div>"""

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    cards = ""
    for svc in SERVICES:
        ok = await check_service(svc)
        sup = supervisor_status(svc["supervisor"])
        log = tail_log(svc["log"], 20)
        err = tail_log(svc["err"], 5)
        combined_log = log
        if err.strip() and err != "(vide)" and err != "(pas encore de logs)":
            combined_log += "\n--- STDERR ---\n" + err

        dot_class = "dot-ok" if ok else "dot-fail"
        badge_class = "badge-ok" if ok else "badge-fail"
        status_text = "EN LIGNE" if ok else "HORS LIGNE"

        cards += CARD_TEMPLATE.format(
            dot_class=dot_class,
            name=svc["name"],
            desc=svc["desc"],
            badge_class=badge_class,
            status_text=status_text,
            supervisor=sup,
            log=combined_log.replace("<", "&lt;").replace(">", "&gt;"),
        )

    html = HTML_TEMPLATE.format(cards=cards, time=now)
    return HTMLResponse(content=html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
PYEOF

# ════════════════════════════════════════════════════════════
# 4. SUPERVISOR CONFIG
# ════════════════════════════════════════════════════════════
echo ""
echo ">>> Configuration de supervisor..."

cat > /etc/supervisor/conf.d/alexandria-stack.conf << 'SUPEOF'

[program:qwen3-tts]
command=/workspace/tts_env/bin/python3 app.py --ip 0.0.0.0 --port 7860
directory=/workspace/Qwen3-TTS
autostart=true
autorestart=true
startsecs=15
stdout_logfile=/workspace/logs/qwen3-tts.log
stderr_logfile=/workspace/logs/qwen3-tts.err
environment=CUDA_VISIBLE_DEVICES="0"

[program:vllm]
command=/workspace/llm_env/bin/python3 -m vllm.entrypoints.openai.api_server --model /workspace/models/mistral --port 8000 --host 0.0.0.0 --tensor-parallel-size 1 --gpu-memory-utilization 0.85
directory=/workspace
autostart=true
autorestart=true
startsecs=30
stdout_logfile=/workspace/logs/vllm.log
stderr_logfile=/workspace/logs/vllm.err
environment=CUDA_VISIBLE_DEVICES="0"

[program:dashboard]
command=/workspace/dashboard_env/bin/python3 /workspace/dashboard.py
directory=/workspace
autostart=true
autorestart=true
startsecs=5
stdout_logfile=/workspace/logs/dashboard.log
stderr_logfile=/workspace/logs/dashboard.err

SUPEOF

supervisord -c /etc/supervisor/supervisord.conf 2>/dev/null || true
sleep 3
supervisorctl reread  2>/dev/null || true
supervisorctl update  2>/dev/null || true
supervisorctl start all 2>/dev/null || true

echo ""
echo "================================================"
echo "  INSTALLATION TERMINEE"
echo ""
echo "  Dashboard statut  -> http://<vast-ip>:<port-ext-3000>"
echo "  Qwen3-TTS Gradio  -> http://<vast-ip>:<port-ext-7860>"
echo "  vLLM API          -> http://<vast-ip>:<port-ext-8000>/v1"
echo "================================================"
