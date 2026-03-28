#!/usr/bin/env python3
"""
Serveur Gradio compatible Alexandria (mode external).
Expose fn_index=0 (custom voice) et fn_index=1 (voice clone)
avec les api_name corrects attendus par Alexandria/tts.py.
"""
import argparse, os, tempfile, numpy as np
import gradio as gr

_custom_model = None
_clone_model  = None

def _get_custom_model():
    global _custom_model
    if _custom_model is None:
        from qwen_tts import Qwen3TTSModel
        print("Loading CustomVoice model...")
        _custom_model = Qwen3TTSModel("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        print("CustomVoice model loaded.")
    return _custom_model

def _get_clone_model():
    global _clone_model
    if _clone_model is None:
        from qwen_tts import Qwen3TTSModel
        print("Loading Base/Clone model...")
        _clone_model = Qwen3TTSModel("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        print("Base/Clone model loaded.")
    return _clone_model

VOICE_MAP = {
    "Aiden": "Aiden", "Dylan": "Dylan", "Eric": "Eric",
    "Ono_anna": "Ono_anna", "Ryan": "Ryan", "Serena": "Serena",
    "Sohee": "Sohee", "Uncle_fu": "Uncle_fu", "Vivian": "Vivian",
}

def generate_custom_voice(text: str, lang: str, speaker: str, instruct: str):
    """fn_index=0 — api_name=/generate_custom_voice"""
    try:
        import soundfile as sf
        model  = _get_custom_model()
        spk_id = VOICE_MAP.get(speaker, "Ryan")
        instr  = instruct.strip() if instruct and instruct.strip() else "Neutral delivery."
        print(f"[custom] speaker={spk_id} | instruct={instr[:60]} | text={text[:60]}")
        wavs, sr = model.generate_custom_voice(
            text=text, instruct_text=instr, speaker=spk_id,
            language=lang if lang and lang != "Auto" else None,
        )
        wav = np.concatenate(wavs) if isinstance(wavs, list) else wavs
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, wav, sr)
        return tmp.name, None
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, str(e)

def generate_clone_voice(ref_audio, ref_text: str, text: str, lang: str):
    """fn_index=1 — api_name=/generate_clone_voice"""
    try:
        import soundfile as sf
        model    = _get_clone_model()
        ref_path = ref_audio if isinstance(ref_audio, str) else ref_audio.get("path", ref_audio)
        print(f"[clone] ref_text={ref_text[:40]} | text={text[:60]}")
        wavs, sr = model.generate_voice_clone(
            text=text, ref_audio=ref_path, ref_text=ref_text,
            language=lang if lang and lang != "Auto" else None,
        )
        wav = np.concatenate(wavs) if isinstance(wavs, list) else wavs
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, wav, sr)
        return tmp.name, None
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, str(e)

def build_app():
    with gr.Blocks() as app:
        # fn_index=0 — Custom Voice
        # Les noms des composants doivent correspondre exactement
        # aux kwargs envoyés par Alexandria/tts.py
        with gr.Tab("Custom Voice"):
            txt0  = gr.Textbox(label="text")
            lang0 = gr.Dropdown(["Auto","English","French","Chinese","Japanese","Korean","German","Spanish","Portuguese","Russian"], value="Auto", label="language")
            spk0  = gr.Dropdown(list(VOICE_MAP.keys()), value="Ryan", label="speaker")
            ins0  = gr.Textbox(label="instruct")
            btn0  = gr.Button("Generate")
            out0  = gr.Audio(label="Output", type="filepath")
            err0  = gr.Textbox(label="Error", visible=False)
            btn0.click(
                generate_custom_voice,
                inputs=[txt0, lang0, spk0, ins0],
                outputs=[out0, err0],
                api_name="generate_custom_voice"
            )

        # fn_index=1 — Voice Clone
        with gr.Tab("Voice Clone"):
            ref1  = gr.Audio(label="ref_audio", type="filepath")
            rtxt1 = gr.Textbox(label="ref_text")
            txt1  = gr.Textbox(label="text")
            lang1 = gr.Dropdown(["Auto","English","French","Chinese"], value="Auto", label="language")
            btn1  = gr.Button("Clone")
            out1  = gr.Audio(label="Output", type="filepath")
            err1  = gr.Textbox(label="Error", visible=False)
            btn1.click(
                generate_clone_voice,
                inputs=[ref1, rtxt1, txt1, lang1],
                outputs=[out1, err1],
                api_name="generate_clone_voice"
            )
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=False)
