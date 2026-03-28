# XTTS + vLLM Auto Server  
### UltraSpeed + AutoMonitoring + Prometheus + Grafana

Ce projet installe automatiquement un serveur complet pour :

- XTTS v2 (TTS multilingue GPU)
- vLLM (serveur OpenAI-compatible)
- Mistral 7B Instruct v0.3
- Prometheus (metrics)
- Node Exporter (CPU/RAM/GPU)
- Grafana (dashboard de monitoring)
- Supervisor (auto-restart XTTS + vLLM + monitoring)

Le tout est optimisé pour Vast.ai, avec :

- CUDA 12.1  
- GPU full speed  
- FlashAttention  
- CUDA Graphs  
- Tensor parallel auto  
- Auto-restart en cas de crash  
- Auto-installation à chaque démarrage  

---

## 🚀 Installation sur Vast.ai

Dans **On-start Script**, collez :

curl -s https://raw.githubusercontent.com/HellTruckerfr/xtts-vllm-autoserver/main/install.sh | bash

Ensuite configurez :

### Image Path
nvidia/cuda:12.1.1-runtime-ubuntu22.04

### Docker Options
--gpus all --shm-size=16g -p 7860:7860 -p 8000:8000 -p 9090:9090 -p 9100:9100 -p 3000:3000

### Ports
7860 (XTTS)  
8000 (vLLM)  
9090 (Prometheus)  
9100 (Node Exporter)  
3000 (Grafana)

### Launch Mode
Docker ENTRYPOINT

---

## 🔥 Services disponibles

XTTS : http://<IP>:7860  
vLLM : http://<IP>:8000/v1  
Prometheus : http://<IP>:9090  
Node Exporter : http://<IP>:9100/metrics  
Grafana : http://<IP>:3000  

---

## 📊 Dashboard Grafana

Le dashboard Grafana est fourni dans :

grafana/dashboard.json

Pour l’importer :

1. Ouvrez Grafana : http://<IP>:3000  
2. Connectez-vous (admin / admin)  
3. Allez dans Dashboards → Import  
4. Collez le contenu du fichier JSON  
5. Sélectionnez la source Prometheus  
6. Validez  

---

## 🛡️ Auto-monitoring

Supervisor surveille :

- XTTS  
- vLLM  
- Prometheus  
- Node Exporter  
- Grafana  

En cas de crash → redémarrage automatique.

---

## 📦 Structure du repo

xtts-vllm-autoserver/  
├── install.sh  
├── README.md  
└── grafana/  
    └── dashboard.json

---

## 🧠 Auteur

HellTruckerfr — GPU pipeline architect
