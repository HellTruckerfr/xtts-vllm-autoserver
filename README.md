# XTTS + vLLM Auto Server  
### UltraSpeed + AutoMonitoring + Prometheus + Grafana

Ce projet installe automatiquement un serveur complet pour :

- **XTTS v2** (TTS multilingue GPU)
- **vLLM** (serveur OpenAI-compatible)
- **Mistral 7B Instruct v0.3**
- **Prometheus** (metrics)
- **Node Exporter** (CPU/RAM/GPU)
- **Grafana** (dashboard de monitoring)
- **Supervisor** (auto-restart XTTS + vLLM + monitoring)

Le tout est optimisé pour **Vast.ai**, avec :

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

```bash
curl -s https://raw.githubusercontent.com/HellTruckerfr/xtts-vllm-autoserver/main/install.sh | bash
