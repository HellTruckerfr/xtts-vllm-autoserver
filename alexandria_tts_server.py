#!/usr/bin/env python3
import argparse, os, tempfile, numpy as np
import gradio as gr

_custom_model = None
_clone_model  = None
_lora_models  = {}
LORA_DIR = "/workspace/lora_models"
_clone_prompt_cache = {}  # cache: ref_audio_path -> prompt

def _get_custom_model():
    global _custom_model
    if _custom_model is None:
        from qwen_tts import Qwen3TTSModel
        print("Loading CustomVoice model...")
        _custom_model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda")
        print("CustomVoice model loaded.")
    return _custom_model

def _get_lora_model(lora_name):
    global _lora_models
    if lora_name not in _lora_models:
        from qwen_tts import Qwen3TTSModel
        lora_path = os.path.join(LORA_DIR, lora_name)
        if not os.path.exists(lora_path):
            print(f"LoRA not found: {lora_path}, falling back to custom")
            return None
        print(f"Loading LoRA: {lora_name}...")
        model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda")
        model.model.load_adapter(lora_path)
        _lora_models[lora_name] = model
        print(f"LoRA {lora_name} loaded.")
    return _lora_models[lora_name]

VOICE_MAP = {
    "Aiden": "Aiden", "Dylan": "Dylan", "Eric": "Eric",
    "Ono_anna": "Ono_anna", "Ryan": "Ryan", "Serena": "Serena",
    "Sohee": "Sohee", "Uncle_fu": "Uncle_fu", "Vivian": "Vivian",
}
LORA_VOICES = ["narrator_fr", "narrator_fr_v2", "feminine_fr"]

def _pad_text(text):
    text = text.strip()
    if text and text[-1] not in ".!?\u2026\xbb\"\u2019":
        text = text + "."
    return text

def _add_silence(wav, sr, ms=300):
    import numpy as np
    silence = np.zeros(int(sr * ms / 1000), dtype=wav.dtype)
    return np.concatenate([wav, silence])

def generate_custom_voice(text: str, language: str, speaker: str, instruct: str, model_size: str = "1.7B", seed: int = -1):
    """fn_index=0 — api_name=/generate_custom_voice"""
    try:
        import soundfile as sf
        import numpy as np
        lang = language if language and language != "Auto" else None
        instr = instruct.strip() if instruct and instruct.strip() else "Neutral delivery."
        text_padded = _pad_text(text)
        print(f"[custom] speaker={speaker} | instruct={instr[:60]} | text={text[:60]}")
        if speaker in LORA_VOICES:
            model = _get_lora_model(speaker)
            if model is not None:
                wavs, sr = model.generate_custom_voice(text=text_padded, instruct_text=instr, speaker="Ryan", language=lang)
                wav = np.concatenate(wavs) if isinstance(wavs, list) else wavs
                wav = _add_silence(wav, sr)
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp.name, wav, sr)
                return tmp.name, None
        model = _get_custom_model()
        spk_id = VOICE_MAP.get(speaker, "Ryan")
        wavs, sr = model.generate_custom_voice(text=text_padded, instruct_text=instr, speaker=spk_id, language=lang)
        wav = np.concatenate(wavs) if isinstance(wavs, list) else wavs
        wav = _add_silence(wav, sr)
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, wav, sr)
        return tmp.name, None
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, str(e)

def generate_clone_voice(ref_audio, ref_text: str, text: str, language: str):
    """fn_index=1 — api_name=/generate_voice_clone"""
    try:
        import soundfile as sf, numpy as np, tempfile
        from qwen_tts import Qwen3TTSModel
        global _clone_model, _clone_prompt_cache
        if _clone_model is None:
            print("Loading Base/Clone model...")
            _clone_model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda")
        model = _clone_model
        ref_path = ref_audio if isinstance(ref_audio, str) else ref_audio.get("path", ref_audio)
        lang = language if language and language != "Auto" else None
        text_padded = _pad_text(text)
        # Cache du prompt clone par ref_audio — evite de recalculer a chaque ligne
        if ref_path not in _clone_prompt_cache:
            print(f"[cache] Calcul prompt clone pour {os.path.basename(ref_path)}...")
            audio_arr, sample_rate = sf.read(ref_path)
            if audio_arr.ndim > 1:
                audio_arr = audio_arr.mean(axis=1)
            prompt = model.create_voice_clone_prompt(ref_audio=(audio_arr, sample_rate), ref_text=ref_text)
            _clone_prompt_cache[ref_path] = prompt
            print(f"[cache] Prompt mis en cache pour {os.path.basename(ref_path)}")
        else:
            print(f"[cache] Reutilisation prompt pour {os.path.basename(ref_path)}")
        prompt = _clone_prompt_cache[ref_path]
        wavs, sr = model.generate_voice_clone(
            text=text_padded, voice_clone_prompt=prompt,
            non_streaming_mode=True, language=lang
        )
        wav = np.concatenate(wavs) if isinstance(wavs, list) else wavs
        wav = _add_silence(wav, sr)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, wav, sr)
        return tmp.name, None
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, str(e)


def generate_batch_clone(texts_json: str, ref_audio, ref_text: str, language: str):
    """fn_index=2 — api_name=/generate_batch_clone — Batch voice clone"""
    try:
        import soundfile as sf, numpy as np, tempfile, json, os
        from qwen_tts import Qwen3TTSModel
        global _clone_model
        if _clone_model is None:
            print("Loading Base/Clone model...")
            _clone_model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda")
        model = _clone_model
        ref_path = ref_audio if isinstance(ref_audio, str) else ref_audio.get("path", ref_audio)
        lang = language if language and language != "Auto" else None
        texts = json.loads(texts_json)
        texts_padded = [_pad_text(t) for t in texts]
        print(f"[batch_clone] {len(texts)} textes | ref={ref_path}")
        wavs_list, sr = model.generate_voice_clone(
            text=texts_padded,
            ref_audio=ref_path,
            ref_text=ref_text,
            language=lang,
            non_streaming_mode=True,
        )
        # Concatene tous les wavs avec silence entre eux
        result_wavs = []
        silence = np.zeros(int(sr * 0.3), dtype=np.float32)
        for wav in wavs_list:
            result_wavs.append(wav)
            result_wavs.append(silence)
        final_wav = np.concatenate(result_wavs)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, final_wav, sr)
        return tmp.name, None
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, str(e)

def build_app():
    import gradio as gr
    all_voices = list(VOICE_MAP.keys()) + LORA_VOICES
    with gr.Blocks() as app:
        with gr.Tab("Custom Voice"):
            txt0  = gr.Textbox(label="text")
            lang0 = gr.Dropdown(["Auto","English","French","Chinese","Japanese","Korean","German","Spanish","Portuguese","Russian"], value="Auto", label="language")
            spk0  = gr.Dropdown(all_voices, value="Ryan", label="speaker")
            ins0  = gr.Textbox(label="instruct")
            mdl0  = gr.Dropdown(["1.7B", "0.6B"], value="1.7B", label="model_size")
            sed0  = gr.Number(value=-1, label="seed")
            btn0  = gr.Button("Generate")
            out0  = gr.Audio(label="Output", type="filepath")
            err0  = gr.Textbox(label="Error", visible=False)
            btn0.click(generate_custom_voice, inputs=[txt0, lang0, spk0, ins0, mdl0, sed0], outputs=[out0, err0], api_name="generate_custom_voice")
        with gr.Tab("Batch Clone"):
            btexts = gr.Textbox(label="texts_json")
            bref   = gr.Audio(label="ref_audio", type="filepath")
            brtxt  = gr.Textbox(label="ref_text")
            blang  = gr.Dropdown(["Auto","English","French","Chinese"], value="Auto", label="language")
            bbtn   = gr.Button("Batch Clone")
            bout   = gr.Audio(label="Output", type="filepath")
            berr   = gr.Textbox(label="Error", visible=False)
            bbtn.click(generate_batch_clone, inputs=[btexts, bref, brtxt, blang], outputs=[bout, berr], api_name="generate_batch_clone")

        with gr.Tab("Voice Clone"):
            ref1  = gr.Audio(label="ref_audio", type="filepath")
            rtxt1 = gr.Textbox(label="ref_text")
            txt1  = gr.Textbox(label="text")
            lang1 = gr.Dropdown(["Auto","English","French","Chinese"], value="Auto", label="language")
            btn1  = gr.Button("Clone")
            out1  = gr.Audio(label="Output", type="filepath")
            err1  = gr.Textbox(label="Error", visible=False)
            btn1.click(generate_clone_voice, inputs=[ref1, rtxt1, txt1, lang1], outputs=[out1, err1], api_name="generate_voice_clone")
    return app

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=False)
