# DuplexPipe

Separation-free reconstruction of two-channel (full-duplex) conversations from single-channel recordings.

Given a mono recording of a two-speaker conversation and a speaker-attributed, time-stamped transcript, DuplexPipe writes a stereo file with one speaker per channel. Speech that the transcript marks as non-overlapped is copied to its speaker's channel. Only the overlapped spans of each utterance are regenerated, with [F5-TTS](https://github.com/SWivid/F5-TTS) infilling conditioned on the same speaker's neighbouring speech and the transcript. Each generated span keeps its original duration, so the two channels stay sample-aligned. No source separation is used.

## Install

```bash
pip install -r requirements.txt
```

A CUDA GPU is recommended. On first run the F5-TTS v1 Base checkpoint and the Vocos vocoder are downloaded from Hugging Face; use `--ckpt` and `--vocoder-dir` to point to local copies instead.

## Input

- **Audio**: wav or flac at any sample rate. Stereo input is mixed to mono; processing runs at 24 kHz.
- **Transcript**: a JSON file in one of two forms.
  1. The output of [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize), stored under the key `raw`:
     ```json
     {"raw": "[0.52][S01]Hello, how can I help?[2.10][1.95][S02]Hi, I have a question.[3.40]"}
     ```
     A long recording may be split into `<stem>_0.json`, `<stem>_1.json`, and so on. Each part carries its offset in seconds as `segment_start`; the parts are merged automatically.
  2. A list of utterances:
     ```json
     [{"start": 0.52, "end": 2.10, "speaker": "S01", "text": "Hello, how can I help?"},
      {"start": 1.95, "end": 3.40, "speaker": "S02", "text": "Hi, I have a question."}]
     ```

  Utterances of different speakers may overlap in time. Keep the punctuation in the text.

## Usage

One recording:

```bash
python duplexpipe.py --wav call.wav --transcript call.json --out out/
```

A directory of recordings, paired with transcripts named `<stem>.json` or `<stem>_0.json`:

```bash
python duplexpipe.py --wav-dir audio/ --transcript-dir transcripts/ --out out/ --skip-done
```

Several GPUs, one process each:

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python duplexpipe.py --wav-dir audio/ --transcript-dir transcripts/ \
      --out out/ --shard $i/4 --skip-done &
done; wait
```

## Output

For each recording `<stem>`:

- `<stem>.wav`: stereo, channel 0 is the first speaker label in sorted order (e.g. `S01`), channel 1 the second. Silence between a speaker's utterances. The sample rate is the input rate, capped at 24 kHz (`--out-sr` to change).
- `<stem>.json`: which utterances were regenerated, the regenerated spans (in seconds of the recording), the context used, and a `high_risk` flag.

Recordings whose transcript fails the sanity check (at least 20 identical consecutive utterances, or more than one utterance per second, typical of a decoding loop) are skipped and listed in `skipped.tsv`.

## Example

`examples/` holds a 65-second excerpt of a two-speaker telephone call from the Fisher English corpus:

- `fisher_example.wav`: the mono mixture (8 kHz).
- `fisher_example.json`: its speaker-attributed transcript, the MOSS-Transcribe-Diarize output for the call restricted to the excerpt.
- `output/fisher_example.wav` and `output/fisher_example.json`: the two-channel result and its report.
- `fisher_example.html`: an interactive view of the example with the input waveform, the transcript, the two output channels with the regenerated spans marked, and players for input and output. Open it in a browser from the `examples/` folder; if the browser blocks local audio, run `python -m http.server` in `examples/` and open `http://localhost:8000/fisher_example.html`.

To regenerate the output:

```bash
cd examples
python ../duplexpipe.py --wav fisher_example.wav --transcript fisher_example.json --out output
```

The Fisher English corpus is distributed by the Linguistic Data Consortium (Fisher English Training Speech Part 2, LDC2005S13).

## How it works

1. **Holes.** For each utterance, the spans overlapped by the other speaker are widened by 0.1 s on each side, clipped to the utterance and merged when closer than 50 ms.
2. **Context.** A partially overlapped utterance takes same-speaker neighbours, nearest first, until it has 3 s of clean speech per side (at most 10 s per side, at most 30 s away), borrowing from the other side at the start or end of a call. A fully overlapped utterance takes one clean utterance per side. Neighbours are used whole, audio and text; their own overlapped spans are regenerated as well and discarded.
3. **Infilling.** The mel spectrograms of the context and the utterance, with its holes zeroed, are concatenated, and F5-TTS regenerates the zeroed frames with the concatenated text as prompt. The sequence length is fixed, so no duration is predicted (32 steps, guidance 2.0, sway sampling, fixed seed).
4. **Splicing.** Each regenerated span is decoded with 8 known frames on either side and cross-faded into the real audio over 20 ms. No gain is applied.
5. **Assembly.** Every utterance is placed on its speaker's channel at its time-stamps, with 5 ms fades at its edges.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--ovl-pad` | 0.10 | seconds added to each side of an overlap |
| `--ctx-min` | 3.0 | clean context to collect per side (s) |
| `--ctx-max` | 10.0 | maximum context per side (s) |
| `--ctx-max-gap` | 30.0 | maximum distance of a context utterance (s) |
| `--nfe` / `--cfg` | 32 / 2.0 | sampling steps / guidance strength |
| `--seed` | 1234 | sampling seed, 0 for random |
| `--max-repeat` / `--max-rate` | 20 / 1.0 | transcript sanity check; `--no-check` disables it |

## Notes

- The timing of the output is the timing of the transcript. Speech the transcript misses is silent in both channels, and an utterance attributed to the wrong speaker lands on the wrong channel.
- Sub-second back-channels that are completely covered by the other speaker are the hardest case and can come out as noise. When such a hole is under 15% of the conditioning sequence, the utterance is kept and marked `high_risk` in the JSON report, so it can be filtered.
- F5-TTS v1 Base is a 24 kHz model; telephone audio at 8 kHz is upsampled for processing and returned at its own rate.

## Acknowledgements

DuplexPipe builds on [F5-TTS](https://github.com/SWivid/F5-TTS), [Vocos](https://github.com/gemelo-ai/vocos) and [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize). The pretrained F5-TTS weights are distributed by their authors under their own license (CC-BY-NC at the time of writing); check it before commercial use.

## License

The code is released under the [MIT License](LICENSE). The files in `examples/` are excerpts of the Fisher English corpus and are not covered by this license; they remain subject to the Linguistic Data Consortium's terms.

## Citation

The accompanying paper is under review; a citation will be added here.
