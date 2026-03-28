#!/bin/bash
set -e
echo "================================================"
echo "  Qwen3-TTS + vLLM + Status - Alexandria Stack"
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

# Ecriture du script de telechargement Mistral
# Lance par supervisor au demarrage — lit HF_TOKEN depuis l'environnement du conteneur
mkdir -p /workspace/models/mistral
cat > /workspace/download_mistral.sh << 'DLEOF'
#!/bin/bash
# Lit HF_TOKEN depuis /root/.hf_token (ecrit par le on-start script vast.ai)
# Si pas de fichier, tente la variable d'environnement en fallback
if [ -f /root/.hf_token ]; then
  HF_TOKEN=$(cat /root/.hf_token)
fi
if [ -z "$HF_TOKEN" ]; then
  echo "ERREUR: HF_TOKEN introuvable."
  echo "Ajoute dans le On-start Script vast.ai, AVANT le curl:"
  echo "  echo 'hf_TON_TOKEN' > /root/.hf_token &&"
  exit 1
fi
# Verifie si deja telechargé
if [ -f /workspace/models/mistral/model-00003-of-00003.safetensors ]; then
  echo "Mistral deja present, skip download"
  exit 0
fi
echo "Telechargement de Mistral-7B-Instruct-v0.3..."
mkdir -p /workspace/models/mistral
cd /workspace/models/mistral
for f in config.json generation_config.json special_tokens_map.json tokenizer.json tokenizer.model model.safetensors.index.json model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors; do
  wget -q --show-progress --header="Authorization: Bearer $HF_TOKEN" "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/$f" -O "$f" && echo "OK $f" || echo "FAIL $f"
done
echo "Download termine."
DLEOF
chmod +x /workspace/download_mistral.sh

echo ">>> Installation page de statut..."
python3 -m venv /workspace/status_env
source /workspace/status_env/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn
deactivate

# Ecriture status_server.py
cat > /workspace/status_server.py << 'PYEOF'
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import subprocess, os, time
app = FastAPI()
SERVICES = {
    "mistral-dl": {"port": 0,    "log": "/workspace/logs/mistral-download.log", "err": "/workspace/logs/mistral-download.err"},
    "qwen3-tts": {"port": 7860, "log": "/workspace/logs/qwen3-tts.log", "err": "/workspace/logs/qwen3-tts.err"},
    "vllm":      {"port": 8000, "log": "/workspace/logs/vllm.log",      "err": "/workspace/logs/vllm.err"},
}
def supervisor_status(name):
    try:
        out = subprocess.check_output(["supervisorctl","status",name],stderr=subprocess.DEVNULL,text=True)
        for s in ["RUNNING","STARTING","STOPPED","FATAL"]:
            if s in out: return s.lower()
        return "unknown"
    except: return "unknown"
def tail_log(path, n=30):
    if not os.path.exists(path): return "(pas encore de log)"
    try: return subprocess.check_output(["tail",f"-{n}",path],stderr=subprocess.DEVNULL,text=True).strip() or "(log vide)"
    except: return "(erreur)"
def gpu_info():
    try:
        out = subprocess.check_output(["nvidia-smi","--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total","--format=csv,noheader,nounits"],stderr=subprocess.DEVNULL,text=True).strip()
        p=[x.strip() for x in out.split(",")]
        return {"name":p[0],"temp":p[1]+"C","util":p[2]+"%","mem":p[3]+" / "+p[4]+" MiB"}
    except: return None
C={"running":"#22c55e","starting":"#f59e0b","stopped":"#6b7280","fatal":"#ef4444","unknown":"#6b7280"}
I={"running":"RUNNING","starting":"STARTING","stopped":"STOPPED","fatal":"ERROR","unknown":"?"}
@app.get("/", response_class=HTMLResponse)
def index():
    gpu=gpu_info()
    gh=""
    if gpu: gh=f'<div class="g"><b>GPU</b> {gpu["name"]} | Temp:{gpu["temp"]} | Load:{gpu["util"]} | VRAM:{gpu["mem"]}</div>'
    sh=""
    for name,info in SERVICES.items():
        st=supervisor_status(name)
        col=C.get(st,"#6b7280"); ico=I.get(st,"?")
        lo=tail_log(info["log"]); le=tail_log(info["err"],n=10)
        lo=lo.replace("<","&lt;").replace(">","&gt;")
        le=le.replace("<","&lt;").replace(">","&gt;")
        sh+=f'<div class="s"><div class="h"><span class="b" style="background:{col}">{ico}</span> <b>{name}</b> <span class="p">:{info["port"]}</span></div><div class="t"><button class="tb a" onclick="sT(this,\'o{name}\')">stdout</button><button class="tb" onclick="sT(this,\'e{name}\')">stderr</button></div><pre class="l" id="o{name}">{lo}</pre><pre class="l hd" id="e{name}">{le}</pre></div>'
    ts=time.strftime("%H:%M:%S")
    css="*{box-sizing:border-box;margin:0;padding:0}body{background:#0f172a;color:#e2e8f0;font-family:sans-serif;padding:24px}h1{font-size:1.4rem;font-weight:700;color:#f8fafc;margin-bottom:4px}.sub{font-size:.8rem;color:#64748b;margin-bottom:20px}.g{background:#1e293b;border-radius:8px;padding:12px;margin-bottom:20px;border:1px solid #334155}.s{background:#1e293b;border-radius:10px;padding:16px;margin-bottom:16px;border:1px solid #334155}.h{display:flex;align-items:center;gap:10px;margin-bottom:10px}.b{font-size:.7rem;font-weight:700;padding:3px 10px;border-radius:20px;color:#fff;text-transform:uppercase}.p{font-size:.8rem;color:#475569;font-family:monospace;margin-left:auto}.t{display:flex;gap:6px;margin-bottom:8px}.tb{background:#0f172a;border:1px solid #334155;color:#94a3b8;padding:3px 10px;border-radius:6px;font-size:.75rem;cursor:pointer}.tb.a{background:#334155;color:#f1f5f9}.l{background:#0f172a;border-radius:6px;padding:10px;font-size:.72rem;line-height:1.5;max-height:200px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;color:#94a3b8}.hd{display:none}"
    js="function sT(b,id){var p=b.closest('.s');p.querySelectorAll('.l').forEach(function(e){e.classList.add('hd')});p.querySelectorAll('.tb').forEach(function(e){e.classList.remove('a')});document.getElementById(id).classList.remove('hd');b.classList.add('a');document.getElementById(id).scrollTop=9999;}document.querySelectorAll('.l:not(.hd)').forEach(function(e){e.scrollTop=9999;});"
    return f"<!DOCTYPE html><html><head><meta charset=UTF-8><meta http-equiv=refresh content=10><title>Alexandria Stack</title><style>{css}</style></head><body><h1>Alexandria Stack</h1><div class=sub>Rafraichissement 10s - {ts}</div>{gh}{sh}<script>{js}</script></body></html>"
@app.get("/health")
def health():
    s={n:supervisor_status(n) for n in SERVICES}
    return {"ok":all(v=="running" for v in s.values()),"services":s,"gpu":gpu_info()}
PYEOF

# Ecriture config supervisor
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

[program:mistral-download]
command=/workspace/download_mistral.sh
directory=/workspace
autostart=true
autorestart=false
startsecs=2
stdout_logfile=/workspace/logs/mistral-download.log
stderr_logfile=/workspace/logs/mistral-download.err

[program:vllm]
command=/bin/bash -c "while [ ! -f /workspace/models/mistral/model-00003-of-00003.safetensors ]; do echo 'Attente du telechargement Mistral...'; sleep 10; done && /workspace/llm_env/bin/python3 -m vllm.entrypoints.openai.api_server --model /workspace/models/mistral --port 8000 --host 0.0.0.0 --tensor-parallel-size 1 --gpu-memory-utilization 0.85"
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
