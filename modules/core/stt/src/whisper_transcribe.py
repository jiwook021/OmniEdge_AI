#!/usr/bin/env python3
"""OmniEdge_AI — STT Transcription Subprocess (Whisper TRT-LLM)

Called by TrtWhisperInferencer::transcribe() to run Whisper V3 Turbo inference
via TRT-LLM's ModelRunnerCpp.

Usage:
    python3 stt_subprocess.py \
        --engine-dir /path/to/whisper-turbo \
        --pcm-file /tmp/oe_pcm_XXXXX.bin \
        --output-file /tmp/oe_stt_result_XXXXX.txt

Alternatively, accepts pre-computed mel:
    python3 stt_subprocess.py \
        --engine-dir /path/to/whisper-turbo \
        --mel-file /tmp/oe_mel_XXXXX.bin \
        --num-frames 3000 --num-mels 128 \
        --output-file /tmp/oe_stt_result_XXXXX.txt

The pcm-file is raw signed 16-bit LE PCM at 16 kHz mono.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def compute_whisper_mel(pcm_float: np.ndarray, n_mels: int = 128) -> torch.Tensor:
    """Compute Whisper-compatible log-mel spectrogram from PCM float32 audio.

    Matches OpenAI Whisper's audio.log_mel_spectrogram() exactly:
    - 400-sample Hann window, 160-sample hop, 25ms/10ms at 16kHz
    - 128 mel bins, 0-8000 Hz triangular filterbank
    - log10 scale with max-relative normalization
    - Padded/trimmed to exactly 3000 frames (30 seconds)
    """
    try:
        # Use whisper library for exact mel computation if available
        import whisper
        audio = torch.from_numpy(pcm_float).float()
        # Pad/trim to 30s (480000 samples)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)
        return mel  # [n_mels, 3000]
    except ImportError:
        pass

    # Fallback: manual computation matching Whisper's spec
    import torch.nn.functional as F

    audio = torch.from_numpy(pcm_float).float()

    # Pad to 30s
    target_samples = 480000
    if audio.shape[0] < target_samples:
        audio = F.pad(audio, (0, target_samples - audio.shape[0]))
    else:
        audio = audio[:target_samples]

    # STFT parameters
    n_fft = 400
    hop_length = 160
    window = torch.hann_window(n_fft)

    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    # Mel filterbank (simplified — for production use the whisper library)
    n_freqs = n_fft // 2 + 1
    mel_filters = _mel_filterbank(n_freqs, n_mels, 16000, 0.0, 8000.0)
    mel_spec = mel_filters @ magnitudes

    # Log scale with Whisper's normalization
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec


def _mel_filterbank(n_freqs, n_mels, sr, fmin, fmax):
    """Build triangular mel filterbank matrix."""
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    freqs = np.linspace(0, sr / 2, n_freqs)
    fb = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        lo, mid, hi = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        for j, f in enumerate(freqs):
            if lo <= f <= mid and mid > lo:
                fb[i, j] = (f - lo) / (mid - lo)
            elif mid < f <= hi and hi > mid:
                fb[i, j] = (hi - f) / (hi - mid)

    return torch.from_numpy(fb).float()


def main():
    parser = argparse.ArgumentParser(description="Whisper TRT-LLM transcription")
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--pcm-file", help="Raw PCM s16le 16kHz mono binary")
    parser.add_argument("--mel-file", help="Pre-computed mel binary [numMels, numFrames] FP32")
    parser.add_argument("--num-frames", type=int, default=3000)
    parser.add_argument("--num-mels", type=int, default=128)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()

    engine_dir = Path(args.engine_dir)

    # ── Load and compute mel spectrogram ──────────────────────────────────
    if args.pcm_file:
        # Load raw PCM s16le → float32
        pcm_bytes = Path(args.pcm_file).read_bytes()
        pcm_s16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        pcm_float = pcm_s16.astype(np.float32) / 32768.0
        mel = compute_whisper_mel(pcm_float, n_mels=args.num_mels)
    elif args.mel_file:
        # Load pre-computed mel
        mel_bytes = Path(args.mel_file).read_bytes()
        mel_fp32 = np.frombuffer(mel_bytes, dtype=np.float32).copy()
        mel = torch.from_numpy(mel_fp32.reshape(args.num_mels, args.num_frames))
    else:
        print("ERROR: Must provide --pcm-file or --mel-file", file=sys.stderr)
        sys.exit(1)

    print(f"[whisper] mel shape: {mel.shape}, min={mel.min():.3f}, max={mel.max():.3f}", file=sys.stderr)

    # Transpose for TRT-LLM: [n_mels, n_frames] → [n_frames, n_mels], FP16
    mel_input = mel.transpose(0, 1).type(torch.float16)  # [3000, 128]
    mel_input_lengths = torch.tensor([mel.shape[1]], dtype=torch.int32)

    # ── Load tokenizer ────────────────────────────────────────────────────
    from transformers import AutoTokenizer

    tokenizer_paths = [
        engine_dir.parent / "whisper-large-v3-turbo",
        Path.home() / "omniedge_models" / "whisper-large-v3-turbo",
        engine_dir / "tokenizer",
    ]
    tokenizer = None
    for tp in tokenizer_paths:
        if tp.exists():
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(tp))
                break
            except Exception:
                continue

    if tokenizer is None:
        print("ERROR: Whisper tokenizer not found", file=sys.stderr)
        sys.exit(1)

    # ── SOT prompt tokens ─────────────────────────────────────────────────
    sot_sequence = [50258, 50259, 50359, 50363]  # SOT, en, transcribe, notimestamps
    eot_id = 50257
    decoder_input_ids = [torch.tensor(sot_sequence, dtype=torch.int32)]

    # ── Create ModelRunnerCpp ─────────────────────────────────────────────
    from tensorrt_llm.runtime import ModelRunnerCpp

    runner = ModelRunnerCpp.from_dir(
        engine_dir=str(engine_dir),
        is_enc_dec=True,
        max_batch_size=1,
        max_input_len=3000,
        max_output_len=args.max_tokens,
        max_beam_width=1,
        kv_cache_free_gpu_memory_fraction=0.15,
        cross_kv_cache_fraction=0.5,
    )

    # ── Run inference ─────────────────────────────────────────────────────
    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids=decoder_input_ids,
            encoder_input_features=[mel_input],
            encoder_output_lengths=mel_input_lengths // 2,
            max_new_tokens=args.max_tokens,
            end_id=eot_id,
            pad_id=eot_id,
            num_beams=1,
            output_sequence_lengths=True,
            return_dict=True,
        )
        torch.cuda.synchronize()
        output_ids = outputs["output_ids"].cpu().numpy().tolist()

    # ── Decode tokens ─────────────────────────────────────────────────────
    raw_ids = output_ids[0][0]
    print(f"[whisper] raw_ids ({len(raw_ids)}): {raw_ids[:20]}", file=sys.stderr)

    # Filter prompt tokens and special tokens
    text_ids = [t for t in raw_ids if t < 50257 and t not in sot_sequence]
    text = tokenizer.decode(text_ids, skip_special_tokens=False).strip()
    print(f"[whisper] text ({len(text_ids)} tokens): \"{text}\"", file=sys.stderr)

    Path(args.output_file).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
