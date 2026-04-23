
import argparse
import csv
import os
import sys
import subprocess
from pathlib import Path
import time
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import datetime
import math
import csv
import imageio_ffmpeg
import gradio as gr


def bootstrap_audio_path():
    """Make ffmpeg/ffprobe discoverable before importing pydub."""
    existing_path = os.environ.get("PATH", "")
    path_entries = []

    winget_path = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links")
    if os.path.exists(winget_path):
        path_entries.append(winget_path)

    try:
        bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        path_entries.append(os.path.dirname(bundled_ffmpeg))
    except Exception:
        pass

    for entry in path_entries:
        if entry and entry not in existing_path.split(os.pathsep):
            existing_path = entry + os.pathsep + existing_path if existing_path else entry

    os.environ["PATH"] = existing_path


bootstrap_audio_path()

import pydub

# --- STARTUP DIAGNOSTICS & ENVIRONMENT STANDARDIZATION ---
print("="*50)
print(f"OmniVoice Startup Diagnostics")
print(f"Python Version: {sys.version}")
print(f"Python Executable: {sys.executable}")
print(f"Working Directory: {os.getcwd()}")
print("="*50)

def configure_audio_binaries():
    """FAANG-level robust binary detection for ffmpeg/ffprobe"""
    resolved_ffmpeg = None
    resolved_ffprobe = None
    
    # 1. Check WinGet Links (Highest priority on this user's machine)
    winget_path = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links")
    if os.path.exists(winget_path):
        os.environ["PATH"] += os.pathsep + winget_path
        potential_ffmpeg = os.path.join(winget_path, "ffmpeg.exe")
        potential_ffprobe = os.path.join(winget_path, "ffprobe.exe")
        if os.path.exists(potential_ffmpeg): resolved_ffmpeg = potential_ffmpeg
        if os.path.exists(potential_ffprobe): resolved_ffprobe = potential_ffprobe

    # 2. Fallback to imageio-ffmpeg
    if not resolved_ffmpeg:
        try:
            resolved_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            os.environ["PATH"] += os.pathsep + os.path.dirname(resolved_ffmpeg)
            # Try to guess ffprobe location relative to ffmpeg
            guess_ffprobe = resolved_ffmpeg.replace("ffmpeg", "ffprobe")
            if os.path.exists(guess_ffprobe): resolved_ffprobe = guess_ffprobe
        except Exception: pass

    # 3. Final check via 'where' command
    if not resolved_ffmpeg:
        try:
            resolved_ffmpeg = subprocess.check_output(["where", "ffmpeg"], text=True).splitlines()[0]
        except Exception: pass
    if not resolved_ffprobe:
        try:
            resolved_ffprobe = subprocess.check_output(["where", "ffprobe"], text=True).splitlines()[0]
        except Exception: pass

    # Apply to pydub
    if resolved_ffmpeg:
        pydub.AudioSegment.converter = resolved_ffmpeg
        print(f"✅ ffmpeg resolved: {resolved_ffmpeg}")
    else:
        print("⚠️ Warning: ffmpeg not found. Voice cloning and non-WAV uploads will fail.")
        
    if resolved_ffprobe:
        # pydub doesn't have a direct probe setter, but adding to PATH handles it for shell outs
        print(f"✅ ffprobe resolved: {resolved_ffprobe}")
    else:
        print("⚠️ Warning: ffprobe not found. Gradio Audio components may crash on non-WAV files.")

configure_audio_binaries()

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder
import whisper
from whisper.tokenizer import LANGUAGES




def format_language_name(language_code):
    if not language_code:
        return "Unknown"
    return LANGUAGES.get(language_code.lower(), language_code).title()


def calculate_confidence_score(result):
    segments = result.get("segments") or []
    avg_logprobs = [
        segment.get("avg_logprob")
        for segment in segments
        if segment.get("avg_logprob") is not None
    ]

    if not avg_logprobs:
        return "N/A"

    confidence = float(np.exp(np.mean(avg_logprobs)) * 100.0)
    confidence = max(0.0, min(100.0, confidence))
    return f"{confidence:.1f}%"


def update_translation(input_lang, output_lang, audio_file="input_audio.wav"):
    selected_language = None if input_lang == "Auto-detect" else input_lang.lower()
    task = "transcribe" if selected_language and output_lang.lower() == input_lang.lower() else "translate"

    try:
        result = whisper_model.transcribe(audio_file, language=selected_language, task=task)
        translated_text = result["text"].strip()
        detected_language = format_language_name(result.get("language"))
        confidence_score = calculate_confidence_score(result)

        translation_result = {
            "translated_text": translated_text,
            "input_text": f"[{detected_language}] {translated_text}",
            "detected_language": detected_language,
            "confidence_score": confidence_score,
        }
        print(
            "Translation result:",
            translation_result["translated_text"],
            f"| detected={translation_result['detected_language']}",
            f"| confidence={translation_result['confidence_score']}",
        )
        return translation_result
    except Exception as e:
        print(f"Error during Whisper transcription: {repr(e)}")
        return {
            "translated_text": "",
            "input_text": "",
            "detected_language": "Unknown",
            "confidence_score": "N/A",
        }

enc_model_fpath="saved_models/default/encoder.pt"
syn_model_fpath="saved_models/default/synthesizer.pt"
voc_model_fpath="saved_models/default/vocoder.pt"
seed = None
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-e", "--enc_model_fpath", type=Path,
                    default=enc_model_fpath,
                    help="Path to a saved encoder")
parser.add_argument("-s", "--syn_model_fpath", type=Path,
                    default=syn_model_fpath,
                    help="Path to a saved synthesizer")
parser.add_argument("-v", "--voc_model_fpath", type=Path,
                    default=voc_model_fpath,
                    help="Path to a saved vocoder")
parser.add_argument("--seed", type=int, default=seed, help=\
    "Optional random number seed value to make toolbox deterministic.")
    
args = parser.parse_args()
arg_dict = vars(args)
print_args(args, parser)
    

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
        "%.1fGb total memory.\n" %
        (torch.cuda.device_count(),
        device_id,
        gpu_properties.name,
        gpu_properties.major,
        gpu_properties.minor,
        gpu_properties.total_memory / 1e9))
else:
    print("Using CPU for inference.\n")

print("Preparing the encoder, the synthesizer and the vocoder...")
ensure_default_models(Path("saved_models"))
encoder.load_model(args.enc_model_fpath)
synthesizer = Synthesizer(args.syn_model_fpath)
vocoder.load_model(args.voc_model_fpath)

print("Loading Whisper model...")
whisper_model = whisper.load_model("small")


def generate_voice(text):
    try:
        # Load and preprocess input voice (reference speaker)
        in_fpath = "input_audio.wav"
        original_wav, sampling_rate = librosa.load(in_fpath, sr=None)
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        print("Loaded and preprocessed input audio")

        # Create embedding from speaker
        embed = encoder.embed_utterance(preprocessed_wav)
        print("Created speaker embedding")

        # It generates spectrograms for each text
        texts = [text]
        embeds = [embed]
        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
        print("Created mel spectrogram")

        # Convert spectrogram to waveform
        generated_wav = vocoder.infer_waveform(spec)
        print("Synthesized the waveform")

        # Pad a bit for playback
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

        # Save directly without preprocessing again
        filename = "output_audio.wav"
        sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
        print(f"Saved output as {filename}")

    except Exception as e:
        print(f"Caught exception: {repr(e)}")


def plot_waveform(audio_path, filename, color):
    try:
        plt.figure(figsize=(8, 2), facecolor='none')
        ax = plt.axes()
        ax.set_facecolor('none')
        y, sr = librosa.load(audio_path, sr=None)
        librosa.display.waveshow(y, sr=sr, color=color, alpha=0.8)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(filename, transparent=True, format='png', dpi=100)
        plt.close()
        return filename
    except Exception as e:
        print(f"Waveform error: {e}")
        return None

def prepare_input_audio(audio_mic, audio_up):
    """Normalizes input audio (mic or file) to input_audio.wav using ffmpeg directly"""
    audio_source = audio_mic if audio_mic is not None else audio_up
    if not audio_source:
        return None, "No audio provided"

    # Gradio 3.x gr.File can return a file path string or a file object
    if isinstance(audio_source, list): audio_source = audio_source[0]
    input_path = audio_source.name if hasattr(audio_source, 'name') else audio_source

    if not os.path.exists(input_path):
        return None, f"File not found: {input_path}"

    target_path = "input_audio.wav"
    if os.path.exists(target_path):
        try: os.remove(target_path)
        except: pass

    try:
        # Use ffmpeg directly for maximum robustness across formats (mp3, m4a, webm, etc)
        command = [
            pydub.AudioSegment.converter,
            "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1",
            target_path
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return target_path, None
    except Exception as e:
        print(f"Normalization error: {repr(e)}")
        # Fallback to soundfile for basic WAVs
        try:
            data, sr = sf.read(input_path)
            sf.write(target_path, data, 16000)
            return target_path, None
        except:
            return None, f"Could not process audio format. Please try a standard WAV or MP3. Error: {str(e)}"


def build_history_updates(history_state):
    history_updates = []
    for i in range(5):
        if i < len(history_state):
            history_updates.append(gr.update(value=history_state[i]["text"], visible=True))
            history_updates.append(gr.update(value=history_state[i]["audio"], visible=True))
        else:
            history_updates.append(gr.update(visible=False))
            history_updates.append(gr.update(visible=False))
    return history_updates


def make_pipeline_output(
    status_text,
    history_state,
    audio_out=None,
    transcript_text="",
    detected_language="",
    confidence_score="",
    package_file=None,
    input_wave=None,
    output_wave=None,
    feedback_message="",
    feedback_visible=False,
    feedback_input_text="",
    feedback_translated_text="",
):
    history_updates = build_history_updates(history_state)
    package_update = gr.update(value=package_file, visible=bool(package_file))
    feedback_update = gr.update(value=feedback_message, visible=feedback_visible)

    return (
        status_text,
        audio_out,
        transcript_text,
        detected_language,
        confidence_score,
        package_update,
        input_wave,
        output_wave,
        feedback_update,
        *history_updates,
        history_state,
        feedback_input_text,
        feedback_translated_text,
    )


def log_feedback(vote, input_text, translated_text):
    if not translated_text:
        gr.Warning("Please generate a translation before leaving feedback.")
        return gr.update(
            value="Please generate a translation before leaving feedback.",
            visible=True,
        )

    feedback_path = Path("feedback_log.csv")
    file_exists = feedback_path.exists()

    with feedback_path.open("a", newline="", encoding="utf-8") as feedback_file:
        writer = csv.DictWriter(
            feedback_file,
            fieldnames=["timestamp", "vote", "input_text", "translated_text"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "vote": vote,
                "input_text": input_text,
                "translated_text": translated_text,
            }
        )

    gr.Info("Thank you for the feedback!")
    return gr.update(value=f"Thanks for sharing your feedback: {vote}.", visible=True)

def run_g(input_lang, output_lang, audio_mic, audio_up, history_state):
    # Initialize history if empty
    if history_state is None:
        history_state = []

    yield make_pipeline_output(
        "⏳ **Status**: Receiving audio...",
        history_state,
    )

    # Normalize audio
    input_wav, error = prepare_input_audio(audio_mic, audio_up)

    if error:
        yield make_pipeline_output(
            f"❌ **Error**: {error}",
            history_state,
        )
        return

    try:
        if os.path.exists("output_audio.wav"):
            try: os.remove("output_audio.wav")
            except: pass

        yield make_pipeline_output(
            "⏳ **Status**: Generating input waveform...",
            history_state,
        )
        in_wave = plot_waveform(input_wav, 'in_wave.png', '#38bdf8')

        yield make_pipeline_output(
            "⏳ **Status**: Transcribing audio...",
            history_state,
            input_wave=in_wave,
        )
        translation_result = update_translation(input_lang, output_lang)
        translated_text = translation_result["translated_text"]
        detected_language = translation_result["detected_language"]
        confidence_score = translation_result["confidence_score"]

        if not translated_text or translated_text.lower().strip() == "thank you for watching":
            yield make_pipeline_output(
                "❌ **Error**: Could not understand or empty transcription! Please try again.",
                history_state,
                input_wave=in_wave,
                detected_language=detected_language,
                confidence_score=confidence_score,
            )
            return

        yield make_pipeline_output(
            "⏳ **Status**: Cloning voice and generating English audio...",
            history_state,
            transcript_text=translated_text,
            detected_language=detected_language,
            confidence_score=confidence_score,
            input_wave=in_wave,
            feedback_input_text=translation_result["input_text"],
            feedback_translated_text=translated_text,
        )
        generate_voice(translated_text)

        if not os.path.exists("output_audio.wav"):
            yield make_pipeline_output(
                "❌ **Error**: Voice generation failed.",
                history_state,
                transcript_text=translated_text,
                detected_language=detected_language,
                confidence_score=confidence_score,
                input_wave=in_wave,
                feedback_input_text=translation_result["input_text"],
                feedback_translated_text=translated_text,
            )
            return

        out_wave = plot_waveform('output_audio.wav', 'out_wave.png', '#a855f7')

        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        history_language = detected_language if input_lang == "Auto-detect" else input_lang
        new_entry = {
            "text": f"[{timestamp}] {history_language} → {output_lang}\n{translated_text}",
            "audio": "output_audio.wav"
        }
        history_state.insert(0, new_entry)
        if len(history_state) > 5:
            history_state.pop()

        yield make_pipeline_output(
            "✅ **Status**: Done!",
            history_state,
            audio_out="output_audio.wav",
            transcript_text=translated_text,
            detected_language=detected_language,
            confidence_score=confidence_score,
            package_file="output_audio.wav",
            input_wave=in_wave,
            output_wave=out_wave,
            feedback_input_text=translation_result["input_text"],
            feedback_translated_text=translated_text,
        )

    except Exception as e:
        print(f"Pipeline error: {repr(e)}")
        yield make_pipeline_output(
            f"❌ **Error**: An unexpected error occurred: {str(e)}",
            history_state,
        )

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

body, .gradio-container {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #0b0f19 !important;
    color: #f8fafc !important;
    background-image: radial-gradient(circle at 15% 50%, rgba(99, 102, 241, 0.08), transparent 25%),
                      radial-gradient(circle at 85% 30%, rgba(56, 189, 248, 0.08), transparent 25%);
    background-attachment: fixed;
}

.glass-card {
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 24px !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3) !important;
    padding: 30px !important;
    overflow: hidden;
}

.title-text {
    font-size: 3rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #f8fafc 0%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.03em;
}

.subtitle-text {
    font-size: 1.1rem !important;
    color: #94a3b8 !important;
    text-align: center;
    max-width: 600px;
    margin: 0 auto 2rem auto !important;
    line-height: 1.6;
}

.primary-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    border-radius: 14px !important;
    padding: 14px 28px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 10px 20px -10px rgba(99, 102, 241, 0.6) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 15px 25px -10px rgba(99, 102, 241, 0.8) !important;
    filter: brightness(1.1);
}

.status-box {
    background: rgba(56, 189, 248, 0.1) !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 12px !important;
    padding: 12px 20px !important;
    color: #38bdf8 !important;
    font-weight: 500 !important;
    margin-bottom: 20px !important;
}

.gr-input, .gr-box, select, input {
    background: rgba(0, 0, 0, 0.2) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
}

.gr-input:focus, .gr-box:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
}

.history-item {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    padding: 16px !important;
    margin-bottom: 16px !important;
    transition: transform 0.2s ease;
}

.history-item:hover {
    background: rgba(255, 255, 255, 0.06) !important;
    transform: translateX(4px);
}
"""

with gr.Blocks(css=custom_css, title="OmniVoice AI") as demo:
    history_state = gr.State([])
    feedback_input_state = gr.State("")
    feedback_translated_state = gr.State("")

    # Landing Page
    with gr.Column(visible=True, elem_classes="glass-card", elem_id="landing_page") as landing_page:
        gr.Markdown("<h1 class='title-text'>OmniVoice AI</h1>")
        gr.Markdown("<p class='subtitle-text'>Experience seamless, zero-shot voice cloning and translation. Speak in your native tongue and hear it rendered instantly in English, with your exact vocal tone and emotion preserved.</p>")
        
        with gr.Row():
            gr.Markdown("")
            try_btn = gr.Button("Try it Now ✨", elem_classes="primary-btn")
            gr.Markdown("")

    # Main App
    with gr.Column(visible=False) as main_app:
        with gr.Row():
            # Left Column (Main functionality)
            with gr.Column(scale=2, elem_classes="glass-card"):
                status_md = gr.Markdown("🟢 **Ready**: Please provide audio input.", elem_classes="status-box")
                
                with gr.Row():
                    input_lang = gr.Dropdown(
                        ["Auto-detect", "Tamil", "Malayalam", "Hindi", "Kannada", "Telugu"], 
                        label="Input Language", value="Auto-detect", interactive=True
                    )
                    output_lang = gr.Dropdown(
                        ["English"], label="Output Language", value="English", interactive=True
                    )
                gr.Markdown(
                    "`Auto-detect` is slightly slower because Whisper first analyzes the opening audio to identify the language."
                )
                
                with gr.Tabs():
                    with gr.Tab("🎙️ Record Voice"):
                        audio_mic = gr.Audio(source="microphone", type="filepath", label="Record Your Voice")
                    with gr.Tab("📁 Upload File"):
                        audio_up = gr.File(label="Upload Audio File", file_types=[".wav", ".mp3", ".m4a", ".webm", ".aac", ".ogg", ".mp4"])
                
                in_waveform = gr.Image(label="Input Waveform", visible=True)
                
                translate_btn = gr.Button("Translate & Clone Voice 🚀", elem_classes="primary-btn")
                
                gr.Markdown("### Result")
                audio_out = gr.Audio(
                    type="filepath",
                    label="🔊 Synthesized Voice",
                    autoplay=True,
                    show_download_button=True,
                )
                out_waveform = gr.Image(label="Output Waveform", visible=True)
                text_out = gr.Textbox(label="Translated Transcript")
                with gr.Row():
                    detected_lang_out = gr.Textbox(label="Detected Language", interactive=False)
                    confidence_out = gr.Textbox(label="Confidence Score", interactive=False)
                package_file = gr.File(label="Download Full Package", visible=False, interactive=False)
                with gr.Row():
                    helpful_btn = gr.Button("👍 Helpful")
                    not_helpful_btn = gr.Button("👎 Not Helpful")
                feedback_status = gr.Markdown(visible=False)

            # Right Column (History)
            with gr.Column(scale=1, elem_classes="glass-card"):
                gr.Markdown("### 🕒 Recent Translations")
                history_boxes = []
                history_audios = []
                for i in range(5):
                    with gr.Column(visible=False, elem_classes="history-item") as hist_col:
                        hist_text = gr.Textbox(label=f"Translation {i+1}", interactive=False)
                        hist_audio = gr.Audio(label="Replay", interactive=False)
                        history_boxes.append(hist_text)
                        history_audios.append(hist_audio)

    # Navigation logic
    def show_app():
        return gr.update(visible=False), gr.update(visible=True)
        
    try_btn.click(fn=show_app, inputs=None, outputs=[landing_page, main_app])

    # Combine history outputs
    hist_outputs = []
    for t, a in zip(history_boxes, history_audios):
        hist_outputs.extend([t, a])

    translate_btn.click(
        fn=run_g,
        inputs=[input_lang, output_lang, audio_mic, audio_up, history_state],
        outputs=[
            status_md,
            audio_out,
            text_out,
            detected_lang_out,
            confidence_out,
            package_file,
            in_waveform,
            out_waveform,
            feedback_status,
        ] + hist_outputs + [
            history_state,
            feedback_input_state,
            feedback_translated_state,
        ]
    )

    helpful_btn.click(
        fn=lambda input_text, translated_text: log_feedback("Helpful", input_text, translated_text),
        inputs=[feedback_input_state, feedback_translated_state],
        outputs=[feedback_status],
    )
    not_helpful_btn.click(
        fn=lambda input_text, translated_text: log_feedback("Not Helpful", input_text, translated_text),
        inputs=[feedback_input_state, feedback_translated_state],
        outputs=[feedback_status],
    )

demo.queue(concurrency_count=1, max_size=10).launch(share=False)
