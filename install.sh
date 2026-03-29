#!/bin/bash
set -e
echo "================================================"
echo "  Qwen3-TTS - Alexandria TTS Server"
echo "================================================"

apt-get update -y
apt-get install -y python3-pip python3-venv python3.10-venv git wget curl supervisor ffmpeg sox zip

mkdir -p /workspace/logs

echo ">>> Installation de Qwen3-TTS + serveur Gradio Alexandria-compatible..."
python3 -m venv /workspace/tts_env
source /workspace/tts_env/bin/activate
pip install --upgrade pip
# numpy 1.26.4 requis — numpy 2.x incompatible avec torch 2.5.1
pip install numpy==1.26.4
# torch 2.5.1+cu124 requis pour compatibilite avec qwen-tts/transformers 4.57.3
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install qwen-tts gradio accelerate soundfile
deactivate

# Telecharge le serveur wrapper compatible Alexandria
curl -s https://raw.githubusercontent.com/HellTruckerfr/xtts-vllm-autoserver/main/alexandria_tts_server.py \
  -o /workspace/alexandria_tts_server.py

# Script de lancement TTS avec le bon environnement Python
cat > /workspace/start_tts.sh << 'STARTEOF'
#!/bin/bash
export PATH="/workspace/tts_env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export VIRTUAL_ENV="/workspace/tts_env"
export PYTHONPATH=""
exec /workspace/tts_env/bin/python3 /workspace/alexandria_tts_server.py --host 0.0.0.0 --port 7860
STARTEOF
chmod +x /workspace/start_tts.sh

# Configuration supervisor
cat > /etc/supervisor/conf.d/alexandria-tts.conf << 'SUPEOF'
[program:qwen3-tts]
command=/workspace/start_tts.sh
directory=/workspace
autostart=true
autorestart=true
startsecs=15
stdout_logfile=/workspace/logs/qwen3-tts.log
stderr_logfile=/workspace/logs/qwen3-tts.err
SUPEOF

supervisorctl reread && supervisorctl update && supervisorctl start qwen3-tts || true

echo "================================================"
echo " INSTALLATION COMPLETE"
echo " TTS: http://<IP>:7860"
echo "================================================"
