# wakeword-trainer

Train a **custom [openWakeWord](https://github.com/dscripka/openWakeWord) wake word in your own language**,
with your own voice, on a normal PC (no Colab, no notebooks, no TensorFlow, no GPU). You get a plain
`.onnx` file that works anywhere openWakeWord models are loaded, for example
[GLaSSIST](https://github.com/SmolinskiP/GLaSSIST) (a one-command install into it is included).

It was built for Brazilian Portuguese (`bruninha`, `bruna`) and ships corpora for **English, Portuguese
and Spanish**; a language is one JSON file in `wakeword_trainer/corpora/`.

> Português: veja a seção [Em português](#em-português) no final.

## What it does

1. **generate** synthetic positives and negatives with neural voices of the chosen language
2. **record** your own voice (the word, normal speech, quiet room, an *exam* set) and, optionally,
   everyday audio (TV, music, conversation) so the model learns what is *not* the word
3. **train** a tiny dense head on openWakeWord embeddings and export it to ONNX (`[1,16,96] -> [1,1]`)
4. **evaluate** on audio never used for training and say honestly if the result is good, usable or weak,
   with an estimate of false triggers per hour of everyday sound
5. **install** the model into a GLaSSIST folder (never overwrites a different model silently)

A window (`wakeword-trainer gui`) walks through the steps; every step is also a CLI command.

## Privacy (read this)

* Your recordings and models stay in the workspace folder on your computer. Nothing of yours is uploaded.
* **Step 3 is the one exception, and it needs your explicit consent.** The synthetic voices are made by the
  `edge-tts` package, which sends the **text** of the training phrases (including your wake word) to an
  *unofficial* Microsoft Edge online text-to-speech endpoint. No account or key, but it is not local, it is
  not an official API and it can stop working. The window has a consent checkbox; the CLI refuses to run
  `generate` without `--allow-online-tts` (or `WAKEWORD_TRAINER_ALLOW_ONLINE_TTS=1`).
* The microphone is used only when you ask to record or test.
* The three openWakeWord base models are downloaded by openWakeWord and then **verified against pinned
  SHA-256 checksums** (`wakeword_trainer/base_models.json`); a mismatch stops everything.
* The feature cache is a plain `.npz` loaded with `allow_pickle=False`, so no file in the workspace can run
  code when loaded.

## Install

Python 3.9 to 3.13. Use a virtual environment.

```bash
git clone https://github.com/gustavomorozi/glassist-wakeword-trainer
cd glassist-wakeword-trainer
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
pip install openwakeword==0.6.0 --no-deps             # see note below
wakeword-trainer models                                # downloads + verifies the base models
```

* `openwakeword` is installed with `--no-deps` because its metadata requires `tflite-runtime` on Linux,
  which is not needed for ONNX models. Everything else it needs is already in the dependencies.
* For reproducible installs use the hash-pinned lock: `pip install --require-hashes -r requirements.lock.txt`
  (install `torch` first, as above).
* `PyAudio` needs PortAudio: on Linux `sudo apt install portaudio19-dev`; on Windows/macOS wheels exist.

## Use

```bash
wakeword-trainer gui                                  # the step-by-step window
```

or from the command line (about 30 to 40 minutes, mostly waiting):

```bash
wakeword-trainer init     --workspace ./bruninha --word Bruninha --language pt
wakeword-trainer generate --workspace ./bruninha --allow-online-tts
wakeword-trainer record   --workspace ./bruninha word        # say the word ~25+ times
wakeword-trainer record   --workspace ./bruninha speech      # talk normally for 3 minutes
wakeword-trainer record   --workspace ./bruninha quiet
wakeword-trainer record   --workspace ./bruninha exam        # separate recordings used only to measure
wakeword-trainer train    --workspace ./bruninha
wakeword-trainer record   --workspace ./bruninha everyday    # optional: TV/music/talk, then retrain
wakeword-trainer install  --workspace ./bruninha --glassist /path/to/GLaSSIST
```

The model is `<workspace>/out/<word>.onnx`. Add `--json` before the command for one JSON event per line.

### Fewer false triggers (measured)

A short word such as "Bruninha" tends to fire on everyday sounds. These tools were added and **measured** on one
real case (one word, one user, one microphone), always on audio the model never trained on:

* `train --ensemble 5`: five heads trained with different seeds, exported as ONE model that fires only when **all
  agree** (lowest score wins). A single run varied a lot on the same data (2.6% to 8.2% of 231 held-out everyday
  clips fired it); the 5-head consensus fired 1 of 231, still detecting 30 of 30 exam clips.
* `mine-hard`: generates thousands of random sounds (clicks, knocks, syllable-like bursts), lets the current model
  score them and keeps the ones that fool it as negatives. Training the consensus with them: **0 of 231** real
  clips, **0 of 1500** hard test sounds (the previous consensus: 1 of 231 and 1 of 1500; the original single
  model: 4 of 231 and 6 of 1500). The hard sounds come from the same generator family as the mined ones, so
  the real-clip number is the independent one.
* `import-detections`: turns the clips GLaSSIST saves on every trigger into training data, labelled by what the
  pipeline did next ("no text recognised" = false trigger = negative).
* `evaluate` now also prints `threshold_advice` (the lowest threshold that lets no everyday clip through).

What did **not** help, so you do not repeat it: random synthetic clicks (`transients`; no model reacted to them,
so they taught nothing) and speed/pitch perturbation of the positives (`speed_perturb`, now off by default; it made
a run worse: 0.87% vs 0.07% of hard test sounds). A 7-head bagged consensus looked best on real clips (0 of 231) but
was worse on hard sounds (0.87%). These are small samples from a single setup: treat the numbers as evidence for
this case, not as a guarantee.

### Honest expectations

* Short or common words (fewer than 3 syllables, ordinary names) trigger by mistake in conversation; the
  tool warns about them. A good model needs *your* voice and *your* everyday sound, not only synthetic voices.
* A verdict of "good" requires enough held-out everyday clips; a small sample can never earn it.
* **Verified end to end (2026-10-18, Windows, Python 3.13, fresh venv):** `init` -> `generate` (500 + 1056 synthetic clips, about 5 min) -> `train` (40 epochs, held-out recall 99%) -> `evaluate` (30/30 exam recordings detected, 4 of 201 normal-speech clips triggered it, verdict "good") -> `install` into a GLaSSIST folder (default `alexa` kept) -> the `.onnx` loads in onnxruntime with input `[1,16,96]`. The window opens, moves through the steps and closes cleanly. This run reused existing voice recordings of the author; **recording from the microphone inside the window was not exercised in this run**. That run found and fixed two missing dependencies (`flet-desktop`, `onnx`).
* The tests cover the logic (verdicts, collector, installer, CLI consent, window flow with a stubbed
  process). The full train-to-model pipeline was exercised by hand on Windows; Linux and macOS are untested.

## Tests

```bash
pip install -e ".[test]" && pytest
```

## License

MIT. openWakeWord (Apache-2.0) and its base models have their own licenses; the pre-trained base models
are released for non-commercial use, check [their terms](https://github.com/dscripka/openWakeWord).

## Em português

Treina uma **palavra de ativação personalizada do openWakeWord no seu idioma**, com a sua voz, num PC comum
(sem Colab, sem GPU). O resultado é um `.onnx` comum, que pode ser instalado no GLaSSIST com um comando.

* **Privacidade:** suas gravações ficam só no seu computador. A exceção, que exige seu consentimento
  (caixa na janela, ou `--allow-online-tts` na linha de comando), é a geração das vozes sintéticas: o
  **texto** das frases (incluindo a palavra) é enviado a um serviço online *não oficial* de voz da Microsoft
  Edge, via `edge-tts`.
* **Instalação:** veja a seção *Install* acima (Python 3.9 a 3.13). Depois, `wakeword-trainer gui` abre a
  janela passo a passo; a interface sai em português se o sistema estiver em português
  (ou `WAKEWORD_TRAINER_LANG=pt_BR`).
* **Expectativa realista:** palavras curtas ou comuns disparam sem querer; o veredito só chega a "bom" com
  amostra suficiente de áudio do dia a dia que o modelo nunca viu no treino.
