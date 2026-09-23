#!/usr/bin/env python3
"""DuplexPipe: separation-free full-duplex conversation reconstruction from mono recordings.

Turns a single-channel recording of a two-speaker conversation into a two-channel file,
one speaker per channel, using a speaker-attributed, time-stamped transcript. Speech the
transcript marks as non-overlapped is copied to its speaker's channel. Only the overlapped
spans of each utterance are regenerated, with F5-TTS infilling conditioned on the same
speaker's neighbouring speech and the transcript; each generated span keeps the original
duration, so the two channels stay sample-aligned. No source separation is used.

    python duplexpipe.py --wav call.wav --transcript call.json --out out/
    python duplexpipe.py --wav-dir audio/ --transcript-dir transcripts/ --out out/ --shard 0/4
"""
import argparse
import glob
import json
import os
import re
from importlib.resources import files

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from omegaconf import OmegaConf
from f5_tts.infer.utils_infer import device, load_model, load_vocoder, mel_spec_type, sway_sampling_coef
from f5_tts.model import DiT
from f5_tts.model.utils import convert_char_to_pinyin

MARGIN = 8         # known mel frames decoded on each side of a hole (the vocoder returns (n-1)*hop samples)
XFADE = 0.020      # s, cross-fade between a regenerated span and the real audio next to it
EDGE = 0.005       # s, fade at both ends of every utterance placed on a channel
MERGE_GAP = 0.05   # s, holes closer than this are merged
LOW_FRAC = 0.15    # fully overlapped utterances whose hole is below this share of the sequence are flagged
PUNCT = "。，？！、；：.,?!;:"
AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3")
RAW_SEG = re.compile(r"\[(\d+\.?\d*)\]\[(S\d+)\]([^\[]*)\[(\d+\.?\d*)\]")


# ----------------------------------------------------------------------------- transcript

def load_transcript(path):
    """Return utterances [{"s", "e", "spk", "t"}] and the covered duration in seconds.

    Two formats are accepted:
      * MOSS-Transcribe-Diarize output: a JSON object whose "raw" field reads
        "[start][S01]text[end][start][S02]text[end]...". A long recording may be split into
        <stem>_0.json, <stem>_1.json, ..., each with its "segment_start" offset in seconds.
      * A JSON list of {"start": float, "end": float, "speaker": str, "text": str}.
    """
    m = re.match(r"(.*)_\d+\.json$", path)
    parts = sorted(glob.glob(glob.escape(m.group(1)) + "_[0-9]*.json")) if m else []
    docs = [json.load(open(p, encoding="utf-8")) for p in parts or [path]]
    if isinstance(docs[0], list):
        segs = [{"s": float(u["start"]), "e": float(u["end"]), "spk": str(u["speaker"]), "t": str(u["text"]).strip()}
                for u in docs[0]]
        segs = [u for u in segs if u["e"] > u["s"] and u["t"]]
        return segs, max((u["e"] for u in segs), default=0.0)
    docs.sort(key=lambda d: int(d.get("segment_index") or 0))
    segs = []
    for d in docs:
        off = float(d.get("segment_start") or 0.0)
        for g in RAW_SEG.finditer(d["raw"]):
            s, e, t = float(g.group(1)) + off, float(g.group(4)) + off, g.group(3).strip()
            if e > s and t:
                segs.append({"s": s, "e": e, "spk": g.group(2), "t": t})
    end = max([float(d.get("segment_end") or 0.0) for d in docs] + [max((u["e"] for u in segs), default=0.0)])
    return segs, end


def check_transcript(segs, duration, max_repeat, max_rate):
    """Reject transcripts caught in a decoding loop: many identical consecutive utterances,
    or implausibly many utterances per second."""
    if not segs:
        return False, "no utterances"
    run = longest = 1
    for a, b in zip(segs, segs[1:]):
        run = run + 1 if b["t"] == a["t"] else 1
        longest = max(longest, run)
    rate = len(segs) / max(duration, 1e-6)
    if longest >= max_repeat:
        return False, f"{longest} identical consecutive utterances"
    if rate > max_rate:
        return False, f"{rate:.2f} utterances per second"
    return True, f"longest repeat {longest}, {rate:.2f} utterances/s"


def holes(segs, u, pad):
    """Spans of utterance u (seconds from its start) overlapped by the other speaker,
    widened by `pad` on both sides, clipped to the utterance and merged."""
    dur = u["e"] - u["s"]
    hs = []
    for o in segs:
        if o["spk"] != u["spk"]:
            a, b = max(u["s"], o["s"]), min(u["e"], o["e"])
            if b > a:
                hs.append((max(0.0, a - u["s"] - pad), min(dur, b - u["s"] + pad)))
    merged = []
    for a, b in sorted(hs):
        if merged and a <= merged[-1][1] + MERGE_GAP:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def join_text(parts):
    """Concatenate utterance texts, adding a comma only where the previous text lacks final punctuation."""
    out = ""
    for t in parts:
        t = t.strip()
        if not t:
            continue
        if out:
            if out[-1] not in PUNCT:
                out += "，" if 0x4E00 <= ord(out[-1]) <= 0x9FFF else ", "
            elif t[0].isascii() and t[0].isalnum():
                out += " "
        out += t
    return out


def pick_context(segs, idx, side, cut, full, used, args):
    """Choose same-speaker neighbours of utterance idx as context on one side ("pre" or "post").

    Neighbours are used whole (audio and text); their own overlapped spans become extra holes
    that are regenerated with the target and discarded. A partially overlapped utterance
    collects neighbours, nearest first, until ctx_min seconds of clean speech (at most ctx_max
    seconds per side, at most ctx_max_gap away), borrowing from the other side if needed.
    A fully overlapped utterance gets one clean utterance per side.
    """
    g, used = segs[idx], set(used)
    same = lambda k, c: k != idx and c["spk"] == g["spk"]
    gap = lambda c: (g["s"] - c["e"]) if side == "pre" else (c["s"] - g["e"])
    clean = lambda c: (c["e"] - c["s"]) - sum(b - a for a, b in holes(segs, c, args.ovl_pad))
    if side == "pre":
        cands = sorted([(k, c) for k, c in enumerate(segs) if same(k, c) and c["e"] <= g["s"] + 0.05],
                       key=lambda kc: -kc[1]["e"])
    else:
        cands = sorted([(k, c) for k, c in enumerate(segs) if same(k, c) and c["s"] >= g["e"] - 0.05],
                       key=lambda kc: kc[1]["s"])

    def block(k, c, borrowed):
        used.add(k)
        return {"wav": cut(c["s"], c["e"]), "text": c["t"], "holes": holes(segs, c, args.ovl_pad), "borrowed": borrowed}

    if full:
        def one_clean(pool, nearest):
            fully = [(k, c) for k, c in pool if not holes(segs, c, 0.0) and c["e"] - c["s"] <= args.ctx_max]
            partly = [(k, c) for k, c in pool if clean(c) >= args.min_clean_sec and c["e"] - c["s"] <= args.ctx_max]
            for lst in (fully, partly):
                lst = [(k, c) for k, c in lst if k not in used]
                if lst:
                    if not nearest:
                        lst.sort(key=lambda kc: -clean(kc[1]))
                    return lst[0]
            return None
        pick, borrowed = one_clean([(k, c) for k, c in cands if gap(c) <= args.ctx_max_gap], True), False
        if pick is None:
            pick = one_clean([(k, c) for k, c in enumerate(segs) if same(k, c) and k not in used], False)
            borrowed = True
        return ([block(*pick, borrowed)] if pick else []), used

    out, total, acc, need = [], 0.0, 0.0, min(args.ctx_min, args.ctx_max)

    def take(k, c, borrowed):
        nonlocal total, acc
        cl = clean(c)
        if cl >= args.min_clean_sec:
            out.append(block(k, c, borrowed))
            total += c["e"] - c["s"]
            acc += cl

    for k, c in cands:
        if gap(c) > args.ctx_max_gap:
            break
        d = c["e"] - c["s"]
        if k in used or (out and total + d > args.ctx_max) or (not out and d > 1.5 * args.ctx_max):
            continue
        take(k, c, False)
        if acc >= need:
            break
    if acc < need:                                  # borrow the cleanest utterances from anywhere
        pool = [(-(c["e"] - c["s"] > args.ctx_max), clean(c), k, c) for k, c in enumerate(segs)
                if same(k, c) and k not in used]
        for _, cl, k, c in sorted(pool, reverse=True):
            if cl < args.min_clean_sec:
                break
            if out and total + (c["e"] - c["s"]) > args.ctx_max:
                continue
            take(k, c, True)
            if acc >= need:
                break
    if side == "pre":
        out.reverse()                               # back to chronological order
    else:
        out.sort(key=lambda b: b["borrowed"])       # true following context first
    return out, used


# ----------------------------------------------------------------------------- infilling

class Infiller:
    """F5-TTS infilling: regenerates masked mel frames of a sequence, keeping all known frames."""

    def __init__(self, ckpt=None, vocoder_dir=None, nfe=32, cfg=2.0, seed=1234):
        conf = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
        self.sr = conf.model.mel_spec.target_sample_rate
        self.hop = conf.model.mel_spec.hop_length
        self.n_mel = conf.model.mel_spec.n_mel_channels
        if not ckpt:
            from huggingface_hub import hf_hub_download
            ckpt = hf_hub_download("SWivid/F5-TTS", "F5TTS_v1_Base/model_1250000.safetensors")
        self.vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=bool(vocoder_dir),
                                    local_path=vocoder_dir or "", device=device)
        self.model = load_model(DiT, conf.model.arch, ckpt, mel_spec_type=mel_spec_type, device=device)
        self.nfe, self.cfg, self.seed = nfe, cfg, (seed if seed > 0 else None)

    def masked_mel(self, wav, hole_spans):
        """Mel of `wav` with the hole spans zeroed, the known-frame mask, the frame count and the hole frames."""
        m = self.model.mel_spec(torch.from_numpy(np.ascontiguousarray(wav)).float().unsqueeze(0).to(device))
        m = m.permute(0, 2, 1)
        T = m.shape[1]
        spans = [(max(0, round(a * self.sr / self.hop)), min(T, round(b * self.sr / self.hop))) for a, b in hole_spans]
        cond = torch.zeros(1, 0, self.n_mel, device=device)
        known = torch.zeros(1, 0, dtype=torch.bool, device=device)
        off = 0
        for a, b in spans:
            a = max(a, off)
            if b <= a:
                continue
            cond = torch.cat((cond, m[:, off:a], torch.zeros(1, b - a, self.n_mel, device=device)), dim=1)
            known = torch.cat((known, torch.ones(1, a - off, dtype=torch.bool, device=device),
                               torch.zeros(1, b - a, dtype=torch.bool, device=device)), dim=-1)
            off = b
        cond = torch.cat((cond, m[:, off:]), dim=1)
        known = F.pad(known, (0, cond.shape[1] - known.shape[-1]), value=True)
        return cond, known, T, spans

    @torch.inference_mode()
    def regenerate(self, real, hole_spans, text, pres, posts):
        """Regenerate the holes of `real` (model rate); every sample outside them stays real.
        Returns the new waveform, the text prompt and the hole's share of the sequence."""
        pres = [b for b in pres if len(b["wav"]) > self.hop]
        posts = [b for b in posts if len(b["wav"]) > self.hop]
        before = [self.masked_mel(b["wav"], b["holes"]) for b in pres]
        target = self.masked_mel(real, hole_spans)
        parts = before + [target] + [self.masked_mel(b["wav"], b["holes"]) for b in posts]
        cond = torch.cat([p[0] for p in parts], dim=1)
        known = torch.cat([p[1] for p in parts], dim=-1)
        prompt = join_text([b["text"] for b in pres] + [text] + [b["text"] for b in posts])
        lead = sum(p[0].shape[1] for p in before)   # frames before the target utterance
        _, _, T, spans = target
        share = sum(b - a for a, b in spans) / cond.shape[1]
        gen, _ = self.model.sample(cond=cond, text=convert_char_to_pinyin([prompt]), duration=cond.shape[1],
                                   steps=self.nfe, cfg_strength=self.cfg, sway_sampling_coef=sway_sampling_coef,
                                   edit_mask=known, seed=self.seed)
        gen = gen.to(torch.float32)
        out, cf, hop = real.copy(), int(XFADE * self.sr), self.hop
        for a, b in spans:
            if b <= a:
                continue
            a2, b2 = max(0, lead + a - MARGIN), min(gen.shape[1], lead + b + MARGIN)
            wf = self.vocoder.decode(gen[:, a2:b2, :].permute(0, 2, 1)).squeeze().cpu().numpy()
            ia, i0, i1 = (lead + a - a2) * hop, a * hop, min(len(out), b * hop)
            n = i1 - i0
            if n <= 0:
                continue
            out[i0:i1] = wf[ia:ia + n]
            fade = min(cf, ia, i0)
            if a > 0 and fade > 0:                  # real audio before the hole: cross-fade
                r = np.linspace(0, 1, fade)
                out[i0 - fade:i0] = out[i0 - fade:i0] * (1 - r) + wf[ia - fade:ia] * r
            elif min(cf, n // 3) > 0:               # hole starts the utterance: fade in
                out[i0:i0 + min(cf, n // 3)] *= np.linspace(0, 1, min(cf, n // 3))
            fade = min(cf, len(wf) - (ia + n), len(out) - i1)
            if b < T and fade > 0:                  # real audio after the hole: cross-fade
                r = np.linspace(0, 1, fade)
                out[i1:i1 + fade] = wf[ia + n:ia + n + fade] * (1 - r) + out[i1:i1 + fade] * r
            elif min(cf, n // 3) > 0:               # hole ends the utterance: fade out
                out[i1 - min(cf, n // 3):i1] *= np.linspace(1, 0, min(cf, n // 3))
        return out, prompt, share


# ----------------------------------------------------------------------------- one recording

def edge_fade(y, length):
    y = y.copy()
    L = min(length, len(y) // 4)
    if L > 0:
        y[:L] *= np.linspace(0, 1, L)
        y[-L:] *= np.linspace(1, 0, L)
    return y


def process(wav_path, transcript_path, args, infiller):
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    out_wav = os.path.join(args.out, stem + ".wav")
    if args.skip_done and os.path.exists(out_wav):
        return print("  already done")
    segs, covered = load_transcript(transcript_path)
    speakers = sorted({u["spk"] for u in segs})
    if len(speakers) < 2:
        return print(f"  skipped: {len(speakers)} speaker(s) in the transcript")
    ok, check = check_transcript(segs, covered, args.max_repeat, args.max_rate)
    if not ok and not args.no_check:
        with open(os.path.join(args.out, "skipped.tsv"), "a", encoding="utf-8") as f:
            f.write(f"{stem}\t{check}\n")
        return print("  skipped: " + check)

    x, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    rate = infiller.sr
    audio = torchaudio.transforms.Resample(sr, rate)(torch.from_numpy(x.mean(axis=1)).float()).numpy()
    N, out_sr = len(audio), args.out_sr or min(sr, rate)
    cut = lambda s, e: audio[max(0, int(s * rate)):min(N, int(e * rate))].copy()

    regen, report = {}, []
    for i, u in enumerate(segs):
        hs = holes(segs, u, args.ovl_pad)
        if not hs:
            continue
        hole = sum(b - a for a, b in hs)
        full = hole >= u["e"] - u["s"] - 0.01
        pres, used = pick_context(segs, i, "pre", cut, full, set(), args)
        posts, _ = pick_context(segs, i, "post", cut, full, used, args)
        item = {"index": i, "speaker": u["spk"], "start": u["s"], "end": u["e"], "hole": round(hole, 3),
                "spans": [[round(u["s"] + a, 3), round(u["s"] + b, 3)] for a, b in hs],
                "fully_overlapped": full, "context": [len(pres), len(posts)],
                "borrowed": sum(b["borrowed"] for b in pres + posts)}
        try:
            regen[i], item["prompt"], share = infiller.regenerate(cut(u["s"], u["e"]), hs, u["t"], pres, posts)
            item["hole_share"], item["high_risk"] = round(share, 3), full and share < LOW_FRAC
        except Exception as ex:                     # keep the original audio for this utterance
            item["error"] = str(ex)
        report.append(item)
        print(f"  {i:4d} {u['spk']} hole {hole:.2f}s" + ("  HIGH RISK" if item.get("high_risk") else "")
              + (f"  FAILED: {item['error']}" if "error" in item else ""))

    tracks = {spk: np.zeros(N, np.float32) for spk in speakers}
    for i, u in enumerate(segs):
        i0, i1 = max(0, int(u["s"] * rate)), min(N, int(u["e"] * rate))
        if i1 <= i0:
            continue
        y = regen[i] if i in regen else cut(u["s"], u["e"])
        y = y[:i1 - i0] if len(y) >= i1 - i0 else np.pad(y, (0, i1 - i0 - len(y)))
        tracks[u["spk"]][i0:i1] = edge_fade(y, int(EDGE * rate))
    chans = [librosa.resample(tracks[s], orig_sr=rate, target_sr=out_sr) for s in speakers[:2]]
    n = min(len(c) for c in chans)
    sf.write(out_wav, np.stack([c[:n] for c in chans], 1), out_sr)
    with open(os.path.join(args.out, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump({"audio": wav_path, "transcript": transcript_path, "channels": speakers[:2], "sample_rate": out_sr,
                   "duration": round(N / rate, 3), "utterances": len(segs), "transcript_check": check,
                   "high_risk": [r["index"] for r in report if r.get("high_risk")], "regenerated": report},
                  f, ensure_ascii=False, indent=1)
    print(f"  {len(regen)}/{len(segs)} utterances regenerated -> {out_wav}")


# ----------------------------------------------------------------------------- command line

def main():
    ap = argparse.ArgumentParser(description="Separation-free two-channel reconstruction of mono conversations.")
    ap.add_argument("--wav", help="one recording")
    ap.add_argument("--transcript", help="its transcript (JSON, see load_transcript)")
    ap.add_argument("--wav-dir", help="batch: directory of recordings")
    ap.add_argument("--transcript-dir", help="batch: transcripts named <stem>.json or <stem>_0.json")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--shard", metavar="I/N", help="process every N-th recording starting at I (multi-GPU)")
    ap.add_argument("--skip-done", action="store_true", help="skip recordings whose output exists")
    ap.add_argument("--out-sr", type=int, default=0, help="output sample rate (default: min(input rate, 24 kHz))")
    ap.add_argument("--ckpt", help="F5-TTS v1 Base checkpoint (default: download from Hugging Face)")
    ap.add_argument("--vocoder-dir", help="local Vocos directory (default: download charactr/vocos-mel-24khz)")
    ap.add_argument("--nfe", type=int, default=32, help="sampling steps")
    ap.add_argument("--cfg", type=float, default=2.0, help="classifier-free guidance strength")
    ap.add_argument("--seed", type=int, default=1234, help="sampling seed (0 = random)")
    ap.add_argument("--ovl-pad", type=float, default=0.10, help="widen each overlap by this many seconds per side")
    ap.add_argument("--ctx-min", type=float, default=3.0, help="clean context to collect per side (s)")
    ap.add_argument("--ctx-max", type=float, default=10.0, help="maximum context per side (s)")
    ap.add_argument("--ctx-max-gap", type=float, default=30.0, help="maximum distance of a context utterance (s)")
    ap.add_argument("--min-clean-sec", type=float, default=0.20, help="ignore neighbours with less clean speech (s)")
    ap.add_argument("--max-repeat", type=int, default=20, help="reject transcripts with this many identical utterances in a row")
    ap.add_argument("--max-rate", type=float, default=1.0, help="reject transcripts with more utterances per second")
    ap.add_argument("--no-check", action="store_true", help="process transcripts that fail the check")
    args = ap.parse_args()

    if args.wav_dir and args.transcript_dir:
        jobs = []
        for f in sorted(os.listdir(args.wav_dir)):
            stem, ext = os.path.splitext(f)
            if ext.lower() in AUDIO_EXT:
                t = next((p for p in (os.path.join(args.transcript_dir, stem + s) for s in (".json", "_0.json"))
                          if os.path.exists(p)), None)
                if t:
                    jobs.append((os.path.join(args.wav_dir, f), t))
    elif args.wav and args.transcript:
        jobs = [(args.wav, args.transcript)]
    else:
        ap.error("give --wav and --transcript, or --wav-dir and --transcript-dir")
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        jobs = jobs[i::n]
    os.makedirs(args.out, exist_ok=True)
    infiller = Infiller(args.ckpt, args.vocoder_dir, args.nfe, args.cfg, args.seed)
    for k, (w, t) in enumerate(jobs, 1):
        print(f"[{k}/{len(jobs)}] {w}")
        try:
            process(w, t, args, infiller)
        except Exception as ex:
            print("  failed:", ex)


if __name__ == "__main__":
    main()
