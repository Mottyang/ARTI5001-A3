# Assignment 3: fixed LJSpeech splits

Seed: 42. Ratios: train 80%, validation 10%, test 10%.
Counts: 10,480 / 1,310 / 1,310.

Both expA and expB must load these SAME saved files. Do not split again in either experiment. Share this entire folder with group members.

- *_ids.txt: one utterance ID per line, in fixed order.
- *_metadata.csv: UTF-8, pipe-delimited, no header; ID|original text|normalized text. Read with line.rstrip("\n").split("|") or disable CSV quote handling.
- split_manifest.json: seed, algorithm, counts and SHA-256 checksums.

Set DATA_ROOT to your local LJSpeech-1.1 folder. Resolve each audio as DATA_ROOT / "wavs" / (utterance_id + ".wav"). No audio is copied here.

Use train only for parameter updates. Use validation for model/checkpoint selection. Reserve test for final evaluation, not tuning. For a small pilot, select a fixed subset from train only and reuse it across experiments; do not redraw validation/test.

Pretrained LJSpeech models may already have seen these recordings. This split prevents overlap in your own fine-tuning, but does not establish an unseen test set relative to pretraining. Disclose this limitation and supplement with newly written prompts.
