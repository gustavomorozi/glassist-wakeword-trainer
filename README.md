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

### Honest expectations

* Short or common words (fewer than 3 syllables, ordinary names) trigger by mistake in conversation; the
  tool warns about them. A good model needs *your* voice and *your* everyday sound, not only synthetic voices.
* A verdict of "good" requires enough held-out everyday clips; a small sample can never earn it.
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
