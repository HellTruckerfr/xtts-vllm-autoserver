#!/bin/bash
# Script d'installation pour vast.ai
# Lance Qwen3-TTS (port 7860) + vLLM/Mistral (port 8000)
# Compatible avec Alexandria en mode TTS externe

echo "================================================"
echo "  Qwen3-TTS + vLLM — Serveurs externes Alexandria"
echo "================================================"

# ── Dépendances système ──────────────────────────────────────
apt-get update -y
apt-get install -y \
    python3-pip python3-venv python3.10-venv \
    git wget curl supervisor ffmpeg \
    software-properties-common

mkdir -p /workspace/logs

# ════════════════════════════════════════════════════════════
# 1. QWEN3-TTS (Gradio — port 7860)
#    Utilise SUP3RMASS1VE/Qwen3-TTS, le serveur Gradio
#    qu'Alexandria attend en mode "external"
# ════════════════════════════════════════════════════════════
echo ""
echo ">>> Installation de Qwen3-TTS..."

cd /workspace
git clone https://github.com/SUP3RMASS1VE/Qwen3-TTS.git
cd /workspace/Qwen3-TTS

python3 -m venv /workspace/tts_env
source /workspace/tts_env/bin/activate

pip install --upgrade pip

# PyTorch CUDA 12.1
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Dépendances Qwen3-TTS
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
    echo "Ajoute HF_TOKEN dans les Environment Variables vast.ai."
else
    echo ""
    echo ">>> Telechargement de Mistral-7B-Instruct-v0.3..."
    mkdir -p /workspace/models/mistral
    cd /workspace/models/mistral

    for f in \
        config.json \
        generation_config.json \
        special_tokens_map.json \
        tokenizer.json \
        tokenizer.model \
        model.safetensors.index.json \
        model-00001-of-00003.safetensors \
        model-00002-of-00003.safetensors \
        model-00003-of-00003.safetensors
    do
        wget -q --show-progress \
            --header="Authorization: Bearer $HF_TOKEN" \
            "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/$f" \
            -O "$f" \
            && echo "  OK $f" || echo "  FAIL $f"
    done
fi

# ════════════════════════════════════════════════════════════
# 3. SUPERVISOR — lance les deux services au démarrage
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

SUPEOF

# Lance supervisord puis recharge
supervisord -c /etc/supervisor/supervisord.conf 2>/dev/null || true
sleep 3
supervisorctl reread  2>/dev/null || true
supervisorctl update  2>/dev/null || true
supervisorctl start all 2>/dev/null || true

echo ""
echo "================================================"
echo "  INSTALLATION TERMINEE"
echo ""
echo "  Dans Alexandria (Setup) :"
echo "  TTS Mode     -> external"
echo "  TTS URL      -> http://<vast-ip>:<port-ext-7860>"
echo "  LLM Base URL -> http://<vast-ip>:<port-ext-8000>/v1"
echo "  LLM API Key  -> local"
echo "  LLM Model    -> mistralai/Mistral-7B-Instruct-v0.3"
echo ""
echo "  Logs : /workspace/logs/"
echo "================================================"
