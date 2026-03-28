#!/bin/bash
set -e

echo "==============================================="
echo " XTTS + vLLM + Grafana + Prometheus Installer "
echo "==============================================="

apt update
apt install -y python3-pip python3-venv python3.10-venv wget git curl supervisor

mkdir -p /workspace/models/mistral
mkdir -p /workspace/logs
mkdir -p /workspace/monitoring
cd /workspace

###############################################
# XTTS ENV
###############################################
python3 -m venv /workspace/xtts_env
source /workspace/xtts_env/bin/activate

pip install --upgrade pip
pip install numpy==1.22.0 scipy==1.11.2
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.37.2
pip install TTS==0.22.0 fastapi uvicorn soundfile pydantic

echo "I have purchased a commercial license from Coqui: licensing@coqui.ai" > ~/.coqui_license
echo "Otherwise, I agree to the terms of the non-commercial CPML: https://coqui.ai/cpml" >> ~/.coqui_license

cat << 'EOF' > /workspace/server_xtts.py
from fastapi import FastAPI
from pydantic import BaseModel
from TTS.api import TTS
import base64, io, soundfile as sf

app = FastAPI()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

class Req(BaseModel):
    text: str
    speaker_wav: str | None = None
    language: str = "fr"

@app.post("/tts")
def tts_endpoint(req: Req):
    speaker = None
    if req.speaker_wav:
        wav_bytes = base64.b64decode(req.speaker_wav)
        with open("voice.wav", "wb") as f:
            f.write(wav_bytes)
        speaker = "voice.wav"

    wav = tts.tts(text=req.text, speaker_wav=speaker, language=req.language)

    buf = io.BytesIO()
    sf.write(buf, wav, 24000, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"audio": audio_b64}
EOF

deactivate

###############################################
# vLLM ENV
###############################################
python3 -m venv /workspace/llm_env
source /workspace/llm_env/bin/activate

pip install --upgrade pip
pip install numpy==1.26.4
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.4.2 transformers==4.40.2 lm-format-enforcer==0.9.8

deactivate

###############################################
# DOWNLOAD MISTRAL
###############################################
HF_TOKEN="TON_HF_TOKEN_ICI"

cd /workspace/models/mistral

for f in config.json generation_config.json special_tokens_map.json tokenizer.json tokenizer.model model.safetensors.index.json model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors; do
    wget --header="Authorization: Bearer $HF_TOKEN" \
         "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/$f" \
         -O $f
done

###############################################
# PROMETHEUS + NODE EXPORTER (CPU/RAM)
###############################################
cd /workspace/monitoring

wget https://github.com/prometheus/prometheus/releases/download/v2.51.0/prometheus-2.51.0.linux-amd64.tar.gz
tar -xzf prometheus-2.51.0.linux-amd64.tar.gz
mv prometheus-2.51.0.linux-amd64 prometheus

wget https://github.com/prometheus/node_exporter/releases/download/v1.7.0/node_exporter-1.7.0.linux-amd64.tar.gz
tar -xzf node_exporter-1.7.0.linux-amd64.tar.gz
mv node_exporter-1.7.0.linux-amd64 node_exporter

cat << 'EOF' > /workspace/monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
EOF

###############################################
# GRAFANA
###############################################
cd /workspace/monitoring
wget https://dl.grafana.com/oss/release/grafana-11.0.0.linux-amd64.tar.gz
tar -xzf grafana-11.0.0.linux-amd64.tar.gz
mv grafana-11.0.0 grafana

###############################################
# SUPERVISOR CONFIG
###############################################
cat << 'EOF' > /etc/supervisor/conf.d/stack.conf
[program:xtts]
command=/workspace/xtts_env/bin/uvicorn server_xtts:app --host 0.0.0.0 --port 7860
directory=/workspace
autostart=true
autorestart=true
stdout_logfile=/workspace/logs/xtts.log
stderr_logfile=/workspace/logs/xtts.err

[program:vllm]
command=/workspace/llm_env/bin/python3 -m vllm.entrypoints.openai.api_server --model /workspace/models/mistral --port 8000 --host 0.0.0.0 --tensor-parallel-size auto --gpu-memory-utilization 0.95
directory=/workspace
autostart=true
autorestart=true
stdout_logfile=/workspace/logs/vllm.log
stderr_logfile=/workspace/logs/vllm.err

[program:prometheus]
command=/workspace/monitoring/prometheus/prometheus --config.file=/workspace/monitoring/prometheus/prometheus.yml --web.listen-address=:9090
directory=/workspace/monitoring/prometheus
autostart=true
autorestart=true
stdout_logfile=/workspace/logs/prometheus.log
stderr_logfile=/workspace/logs/prometheus.err

[program:node_exporter]
command=/workspace/monitoring/node_exporter/node_exporter --web.listen-address=:9100
directory=/workspace/monitoring/node_exporter
autostart=true
autorestart=true
stdout_logfile=/workspace/logs/node_exporter.log
stderr_logfile=/workspace/logs/node_exporter.err

[program:grafana]
command=/workspace/monitoring/grafana/bin/grafana-server --homepath=/workspace/monitoring/grafana --http-port=3000
directory=/workspace/monitoring/grafana
autostart=true
autorestart=true
stdout_logfile=/workspace/logs/grafana.log
stderr_logfile=/workspace/logs/grafana.err
EOF

supervisorctl reread || true
supervisorctl update || true
supervisorctl start all || true

echo "==============================================="
echo " INSTALLATION COMPLETE "
echo " XTTS:      http://<IP>:7860"
echo " vLLM:      http://<IP>:8000/v1"
echo " Grafana:   http://<IP>:3000"
echo " Prometheus http://<IP>:9090"
echo " NodeExp:   http://<IP>:9100/metrics"
echo "==============================================="
