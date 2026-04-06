# xtts-vllm-autoserver

Serveur Gradio Qwen3-TTS optimisé pour la génération d'audiobooks en batch GPU, conçu pour tourner sur un serveur vast.ai RTX 4090 et s'intégrer avec [Alexandria Audiobook Generator](https://github.com/HellTruckerfr/alexandria-audiobook).

---

## Architecture

```
Alexandria (PC local)
    ↕ SSH tunnel port 7860
Serveur vast.ai RTX 4090
    └── Gradio (port 7860)
        └── Qwen3-TTS 1.7B Base
            └── LoRA adapters (narrator_fr_v3, feminine_fr)
```

---

## Installation sur vast.ai

### Image Docker recommandée
`nvidia/cuda:12.1.1-runtime-ubuntu22.04`

### On-start script (vast.ai)
```bash
echo "hf_TON_TOKEN" > /root/.hf_token && curl -s https://raw.githubusercontent.com/HellTruckerfr/xtts-vllm-autoserver/main/install.sh | bash
```

### Installation manuelle
```bash
echo "hf_TON_TOKEN" > /root/.hf_token
curl -s https://raw.githubusercontent.com/HellTruckerfr/xtts-vllm-autoserver/main/install.sh | bash
```

---

## Démarrage

Le serveur démarre automatiquement via supervisor après l'installation.

Après un reboot :
```bash
supervisord -c /etc/supervisor/supervisord.conf
sleep 5
supervisorctl start qwen3-tts
```

Vérifier le statut :
```bash
supervisorctl status qwen3-tts
tail -f /workspace/logs/qwen3-tts.log
```

---

## Tunnel SSH (depuis le PC local)

```powershell
ssh -p PORT root@IP -L 7860:localhost:7860 -N
```

Ou avec auto-reconnect :
```powershell
powershell -ExecutionPolicy Bypass -File "tunnel_watchdog.ps1"
```

---

## Endpoints Gradio

| Endpoint | Description |
|----------|-------------|
| `/generate_custom_voice` | Génération avec voix built-in + instruct |
| `/generate_voice_clone` | Clone vocal depuis audio de référence |
| `/generate_batch_clone` | **Batch GPU** — génère N textes simultanément via LoRA |

### generate_batch_clone (principal)

```python
result = client.predict(
    texts_json=json.dumps(["Texte 1", "Texte 2", ...]),
    lora_name="narrator_fr_v3",  # nom du dossier dans /workspace/lora_models/
    language="French",
    api_name="/generate_batch_clone"
)
# Retourne (chemin_json, erreur)
# Le JSON contient une liste de WAVs encodés en base64
```

---

## LoRA Models

Stockés dans `/workspace/lora_models/` — téléchargés automatiquement depuis HuggingFace privé `Helltrucker/audiobook-lora-models`.

| Adapter | Description |
|---------|-------------|
| `narrator_fr_v3` | Narrateur français — 1697 samples, 8 epochs, loss 3.15 |
| `feminine_fr` | Voix féminine française |
| `narrator_fr_v2` | Version précédente (conservée pour compatibilité) |

Chaque dossier contient :
```
narrator_fr_v3/
├── adapter_config.json
├── adapter_model.safetensors
├── ref_sample.wav
└── training_meta.json
```

---

## Paramètres de génération

```python
model.generate_voice_clone(
    text=texts,
    voice_clone_prompt=prompt,
    non_streaming_mode=True,
    language="French",
    do_sample=True,
    temperature=1.2,      # Expressivité (défaut: 1.2)
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.1
)
```

---

## Optimisations

- **torch.cuda.empty_cache()** avant et après chaque batch pour éviter les OOM
- **_trim_start(300ms)** — supprime l'artefact "ur" en début de WAV
- **_add_silence(300ms)** — silence en fin de WAV pour éviter les coupures
- **Nettoyage automatique** — fichiers `/tmp/gradio/` de plus d'1h supprimés après chaque batch
- **Cache prompts LoRA** — le prompt de clonage est calculé une seule fois par LoRA et mis en cache

---

## Monitoring

```bash
# GPU en temps réel
nvitop

# Logs serveur
tail -f /workspace/logs/qwen3-tts.log
tail -f /workspace/logs/qwen3-tts.err

# Espace disque
df -h /workspace
du -sh /workspace/* 2>/dev/null | sort -rh | head -10
```

---

## Nettoyage manuel

```bash
# Vide les tmp Gradio
rm -rf /tmp/gradio/* /tmp/batch_wavs_*

# Vide les logs
> /workspace/logs/qwen3-tts.log
> /workspace/logs/qwen3-tts.err
```

---

## License

MIT
