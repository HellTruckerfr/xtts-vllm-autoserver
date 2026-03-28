#!/bin/bash
set -e
echo "================================================"
echo "  Qwen3-TTS + vLLM + Status — Alexandria Stack"
echo "================================================"

apt-get update -y
apt-get install -y python3-pip python3-venv python3.10-venv git wget curl supervisor ffmpeg software-properties-common

mkdir -p /workspace/logs

echo ">>> Installation de Qwen3-TTS..."
cd /workspace
git clone https://github.com/SUP3RMASS1VE/Qwen3-TTS.git
python3 -m venv /workspace/tts_env
source /workspace/tts_env/bin/activate
pip install --upgrade pip
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r /workspace/Qwen3-TTS/requirements.txt
deactivate

echo ">>> Installation de vLLM..."
python3 -m venv /workspace/llm_env
source /workspace/llm_env/bin/activate
pip install --upgrade pip
pip install numpy==1.26.4
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.4.2 transformers==4.40.2
deactivate

if [ -z "$HF_TOKEN" ]; then
  echo "WARNING: HF_TOKEN non defini - telechargement Mistral ignore."
else
  echo ">>> Telechargement de Mistral-7B-Instruct-v0.3..."
  mkdir -p /workspace/models/mistral
  cd /workspace/models/mistral
  for f in config.json generation_config.json special_tokens_map.json tokenizer.json tokenizer.model model.safetensors.index.json model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors; do
    wget -q --show-progress --header="Authorization: Bearer $HF_TOKEN" "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/$f" -O "$f" && echo "  OK $f" || echo "  FAIL $f"
  done
  cd /workspace
fi

echo ">>> Installation de la page de statut..."
python3 -m venv /workspace/status_env
source /workspace/status_env/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn
deactivate

cat > /workspace/status_server.py << 'PYEOF'
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import subprocess, os, time

app = FastAPI()

SERVICES = {
    "qwen3-tts": {"port": 7860, "log": "/workspace/logs/qwen3-tts.log", "err": "/workspace/logs/qwen3-tts.err"},
    "vllm":      {"port": 8000, "log": "/workspace/logs/vllm.log",      "err": "/workspace/logs/vllm.err"},
}

def supervisor_status(name):
    try:
        out = subprocess.check_output(["supervisorctl", "status", name], stderr=subprocess.DEVNULL, text=True)
        for s in ["RUNNING", "STARTING", "STOPPED", "FATAL"]:
            if s in out:
                return s.lower()
        return "unknown"
    except Exception:
        return "unknown"

def tail_log(path, n=30):
    if not os.path.exists(path):
        return "(pas encore de log)"
    try:
        return subprocess.check_output(["tail", f"-{n}", path], stderr=subprocess.DEVNULL, text=True).strip() or "(log vide)"
    except Exception:
        return "(erreur lecture log)"

def gpu_info():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, text=True).strip()
        p = [x.strip() for x in out.split(",")]
        return {"name": p[0], "temp": p[1]+"°C", "util": p[2]+"%", "mem": p[3]+" / "+p[4]+" MiB"}
    except Exception:
        return None

STATUS_COLOR = {"running": "#22c55e", "starting": "#f59e0b", "stopped": "#6b7280", "fatal": "#ef4444", "unknown": "#6b7280"}
STATUS_ICON  = {"running": "●", "starting": "◌", "stopped": "○", "fatal": "✗", "unknown": "?"}

@app.get("/", response_class=HTMLResponse)
def index():
    gpu = gpu_info()
    gpu_html = ""
    if gpu:
        gpu_html = f'<div class="gpu-card"><span class="gpu-title">GPU</span><span class="gpu-name">{gpu["name"]}</span><span class="gpu-stat">🌡 {gpu["temp"]}</span><span class="gpu-stat">⚡ {gpu["util"]}</span><span class="gpu-stat">🧠 {gpu["mem"]}</span></div>'

    services_html = ""
    for name, info in SERVICES.items():
        st = supervisor_status(name)
        color = STATUS_COLOR.get(st, "#6b7280")
        icon  = STATUS_ICON.get(st, "?")
        log_out = tail_log(info["log"])
        log_err = tail_log(info["err"], n=10)
        services_html += f"""
        <div class="service">
          <div class="service-header">
            <span class="dot" style="color:{color}">{icon}</span>
            <span class="service-name">{name}</span>
            <span class="badge" style="background:{color}">{st}</span>
            <span class="port">:{info['port']}</span>
          </div>
          <div class="log-tabs">
            <button class="tab-btn active" onclick="showTab(this,'stdout-{name}')">stdout</button>
            <button class="tab-btn" onclick="showTab(this,'stderr-{name}')">stderr</button>
          </div>
          <pre class="log-box" id="stdout-{name}">{log_out}</pre>
          <pre class="log-box hidden" id="stderr-{name}">{log_err}</pre>
        </div>"""

    ts = time.strftime("%H:%M:%S")
    return f"""<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8"><meta http-equiv="refresh" content="10">
<title>Alexandria Stack</title><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0f172a;color:#e2e8f0;font-family:'Segoe UI',sans-serif;padding:24px}}
h1{{font-size:1.4rem;font-weight:700;margin-bottom:4px;color:#f8fafc}}
.subtitle{{font-size:.8rem;color:#64748b;margin-bottom:20px}}
.gpu-card{{display:flex;align-items:center;gap:16px;background:#1e293b;border-radius:10px;padding:12px 16px;margin-bottom:20px;border:1px solid #334155;flex-wrap:wrap}}
.gpu-title{{font-weight:700;color:#94a3b8;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em}}
.gpu-name{{font-weight:600;color:#f1f5f9}}
.gpu-stat{{font-size:.85rem;color:#cbd5e1}}
.service{{background:#1e293b;border-radius:10px;padding:16px;margin-bottom:16px;border:1px solid #334155}}
.service-header{{display:flex;align-items:center;gap:10px;margin-bottom:12px}}
.dot{{font-size:1.1rem}}
.service-name{{font-weight:700;font-size:1rem;flex:1}}
.badge{{font-size:.7rem;font-weight:700;padding:2px 8px;border-radius:20px;color:white;text-transform:uppercase;letter-spacing:.05em}}
.port{{font-size:.8rem;color:#475569;font-family:monospace}}
.log-tabs{{display:flex;gap:6px;margin-bottom:8px}}
.tab-btn{{background:#0f172a;border:1px solid #334155;color:#94a3b8;padding:3px 10px;border-radius:6px;font-size:.75rem;cursor:pointer}}
.tab-btn.active{{background:#334155;color:#f1f5f9;border-color:#475569}}
.log-box{{background:#0f172a;border-radius:6px;padding:10px 12px;font-size:.72rem;line-height:1.5;max-height:220px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;border:1px solid #1e293b;color:#94a3b8}}
.hidden{{display:none}}
</style></head><body>
<h1>Alexandria Stack</h1>
<div class="subtitle">Rafraichissement auto toutes les 10s — {ts}</div>
{gpu_html}{services_html}
<script>
function showTab(btn,id){{
  const p=btn.closest('.service');
  p.querySelectorAll('.log-box').forEach(e=>e.classList.add('hidden'));
  p.querySelectorAll('.tab-btn').forEach(e=>e.classList.remove('active'));
  document.getElementById(id).classList.remove('hidden');
  btn.classList.add('active');
  document.getElementById(id).scrollTop=document.getElementById(id).scrollHeight;
}}
document.querySelectorAll('.log-box:not(.hidden)').forEach(e=>e.scrollTop=e.scrollHeight);
</script></body></html>"""

@app.get("/health")
def health():
    statuses = {name: supervisor_status(name) for name in SERVICES}
    return {"ok": all(s=="running" for s in statuses.values()), "services": statuses, "gpu": gpu_info()}
PYEOF

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

[program:status]
command=/workspace/status_env/bin/uvicorn status_server:app --host 0.0.0.0 --port 3000
directory=/workspace
autostart=true
autorestart=true
startsecs=5
stdout_logfile=/workspace/logs/status.log
stderr_logfile=/workspace/logs/status.err
SUPEOF

supervisord -c /etc/supervisor/supervisord.conf 2>/dev/null || true
sleep 3
supervisorctl reread  2>/dev/null || true
supervisorctl update  2>/dev/null || true
supervisorctl start all 2>/dev/null || true

echo "================================================"
echo "  INSTALLATION TERMINEE"
echo "  Status  -> http://<ip>:<port-ext-3000>"
echo "  TTS URL -> http://<ip>:<port-ext-7860>"
echo "  LLM URL -> http://<ip>:<port-ext-8000>/v1"
echo "================================================"
