"""wakeword-trainer: command line.

  wakeword-trainer init     --workspace DIR --word Bruninha --language pt
  wakeword-trainer models                                # download + verify the base models
  wakeword-trainer generate --workspace DIR --allow-online-tts
  wakeword-trainer record   --workspace DIR word|speech|quiet|exam|everyday [--seconds N]
  wakeword-trainer transients --workspace DIR [--count 300]          # synthetic clicks/knocks as negatives
  wakeword-trainer import-detections --workspace DIR --detections FOLDER --log FILE [--include-positives]
  wakeword-trainer train    --workspace DIR [--epochs 40] [--rebuild]
  wakeword-trainer evaluate --workspace DIR
  wakeword-trainer scan     --workspace DIR [--tag env] [--all-clips]
  wakeword-trainer install  --workspace DIR --glassist FOLDER [--replace] [--force]
  wakeword-trainer info     --workspace DIR
  wakeword-trainer gui      [--improve]

Add --json before the command for machine-readable output (one JSON object per line: stage,
progress, log, result, error); the window uses it.
"""
import argparse
import asyncio
import json
import os
import sys

JSON_MODE = False

TTS_NOTICE = (
    'Generating the synthetic voices sends the TEXT of the training phrases, including your wake word, '
    'to an unofficial Microsoft Edge online text-to-speech endpoint (via the edge-tts package; no account '
    'or key, but it is not local). Your own recordings are never sent anywhere. '
    'Run again with --allow-online-tts (or set WAKEWORD_TRAINER_ALLOW_ONLINE_TTS=1) to agree.'
)


def emit(event, **data):
    if JSON_MODE:
        print(json.dumps({'event': event, **data}, ensure_ascii=False), flush=True)
        return
    if event == 'progress':
        done, total = data['done'], max(1, data['total'])
        print(f'\r  {done}/{total} ({done / total:.0%})', end='' if done < total else '\n', flush=True)
    elif event == 'log':
        print(data['msg'], flush=True)
    elif event == 'stage':
        if data['state'] == 'start':
            print(f'-> {data["name"]}...', flush=True)
    elif event == 'error':
        print(f'error: {data["msg"]}', file=sys.stderr, flush=True)
    elif event == 'result':
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str), flush=True)


def build_parser():
    ap = argparse.ArgumentParser(prog='wakeword-trainer', description='Train a custom openWakeWord wake word.')
    ap.add_argument('--json', action='store_true', help='machine-readable output (one JSON object per line)')
    sub = ap.add_subparsers(dest='command', required=True)

    def add(name, help_, workspace=True):
        p = sub.add_parser(name, help=help_)
        if workspace:
            p.add_argument('--workspace', required=True, help='project folder')
        return p

    p = add('init', 'create a project')
    p.add_argument('--word', required=True)
    p.add_argument('--language', default='en', help='corpus code (en, pt, es) or a path to a corpus JSON')
    add('models', 'download and verify the openWakeWord base models', workspace=False)
    p = add('generate', 'generate the synthetic voices')
    p.add_argument('--allow-online-tts', action='store_true', help='agree to send the phrase TEXT to the Edge TTS service')
    p = add('record', 'record your voice or everyday audio')
    p.add_argument('kind', choices=['word', 'exam', 'speech', 'quiet', 'everyday'])
    p.add_argument('--seconds', type=int, help='default: word 120, exam 75, speech 180, quiet 30, everyday 3600')
    p.add_argument('--max-clips', type=int, default=300, help='everyday: stop after this many clips with sound')
    p.add_argument('--device', type=int, help='input device index (default: system default, or WAKEWORD_TRAINER_MIC)')
    p = add('transients', 'make synthetic clicks, knocks, claps and key presses to use as negatives (no recording)')
    p.add_argument('--count', type=int, default=300)
    p = add('import-detections', 'use the clips GLaSSIST saved on each trigger, labelled by what the pipeline did next')
    p.add_argument('--detections', required=True, help='folder with trig_*.wav (e.g. glassist/detections/bruninha)')
    p.add_argument('--log', required=True, action='append', help='GLaSSIST log file (repeat for several)')
    p.add_argument('--include-positives', action='store_true', help='also add the triggers the assistant understood to real_pos')
    p.add_argument('--window', type=int, default=30, help='seconds after a trigger to look for the pipeline result')
    p = add('train', 'train the model and export the .onnx')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--rebuild', action='store_true', help='recompute the audio features')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ensemble', type=int, default=1, help='train N heads that must agree (fewer false triggers)')
    add('evaluate', 'check the model on audio it never trained on')
    p = add('scan', 'count how many everyday clips trigger the model')
    p.add_argument('--tag', default='env')
    p.add_argument('--all-clips', action='store_true', help='include the clips used for training')
    p = add('install', 'add the trained model to a GLaSSIST folder')
    p.add_argument('--glassist', required=True, help='GLaSSIST folder (a source checkout)')
    p.add_argument('--replace', action='store_true', help='make it the ONLY wake word (default: add to the list)')
    p.add_argument('--force', action='store_true', help='overwrite a different model with the same name')
    add('info', 'show what a project contains')
    p = add('gui', 'open the step-by-step window', workspace=False)
    p.add_argument('--improve', action='store_true', help='open on the "Improve" step')
    return ap


def _kind_defaults():
    return {'word': 120, 'exam': 75, 'speech': 180, 'quiet': 30, 'everyday': 3600}


def run(args):
    from .languages import load_corpus
    from .workspace import Project, Workspace
    cmd = args.command

    if cmd == 'gui':
        from .gui import wizard
        wizard.run(improve=args.improve)
        return 0
    if cmd == 'models':
        from . import models_check
        emit('stage', name='models', state='start')
        models_check.ensure(download=True, log=lambda m: emit('log', msg=m))
        emit('result', ok=True, verified=sorted(models_check.expected()))
        return 0
    if cmd == 'init':
        ws = Workspace(args.workspace, Project(word=args.word, language=args.language))
        ws.save_project()
        emit('result', **ws.summary(), model_name=ws.project.model_name)
        return 0

    ws = Workspace(args.workspace)
    emit('stage', name=cmd, state='start')
    progress = lambda done, total: emit('progress', done=done, total=total)
    log = lambda msg: emit('log', msg=msg)

    if cmd == 'info':
        emit('result', **ws.summary(), word=ws.project.word, model_name=ws.project.model_name)
    elif cmd == 'generate':
        if not (args.allow_online_tts or os.environ.get('WAKEWORD_TRAINER_ALLOW_ONLINE_TTS') == '1'):
            emit('error', msg=TTS_NOTICE, code='consent_required')
            return 2
        from . import models_check, tts_gen
        models_check.ensure(download=True, log=log)
        stats = asyncio.run(tts_gen.generate(ws, load_corpus(ws.project.language), progress))
        emit('result', **stats)
    elif cmd == 'record':
        from . import models_check, recorder
        seconds = args.seconds or _kind_defaults()[args.kind]
        emit('log', msg=f'Recording {args.kind} for up to {seconds} s. Speak now.' if args.kind in ('word', 'exam')
             else f'Recording {args.kind} for up to {seconds} s.')
        if args.kind == 'everyday':
            n = recorder.record_ambient(seconds, ws.real_neg, 'env', device=args.device, max_clips=args.max_clips,
                                        on_update=lambda e, lvl, saved: progress(saved, args.max_clips))
            emit('result', kind='everyday', clips=n)
        else:
            audio = recorder.record(seconds, device=args.device)
            if args.kind == 'word':
                n = recorder.save_word_clips(audio, ws.real_pos, 'cli')
            elif args.kind == 'exam':
                n = recorder.save_word_clips(audio, ws.test_pos, 'cli')
            elif args.kind == 'speech':
                n = recorder.save_speech_chunks(audio, ws.real_neg, 'cli')
            else:
                n = recorder.append_ambient(audio, ws.ambient)
            emit('result', kind=args.kind, clips=n if args.kind != 'quiet' else None,
                 seconds=round(len(audio) / 16000, 1))
    elif cmd == 'transients':
        from . import transients
        emit('result', **transients.generate(ws, args.count, progress=progress))
    elif cmd == 'import-detections':
        from . import detections
        emit('result', **detections.import_detections(ws, args.detections, args.log, args.include_positives, args.window))
    elif cmd == 'train':
        from . import models_check, train
        models_check.ensure(download=True, log=log)
        emit('result', **train.train(ws, args.epochs, progress=progress, log=log, rebuild=args.rebuild, seed=args.seed, ensemble=args.ensemble,
                                     stage=lambda name, state: emit('stage', name=name, state=state)))
    elif cmd == 'evaluate':
        from . import evaluate
        emit('stage', name='evaluate', state='start')
        result = evaluate.evaluate(ws, progress=progress)
        emit('stage', name='evaluate', state='done')
        emit('result', **result)
    elif cmd == 'scan':
        from . import evaluate
        emit('result', **evaluate.scan_clips(ws, tag=args.tag, holdout_only=not args.all_clips, progress=progress))
    elif cmd == 'install':
        from .integrations import glassist
        if not os.path.exists(ws.model_path):
            emit('error', msg=f'no trained model yet ({ws.model_path}); run `train` first')
            return 1
        try:
            emit('result', **glassist.install_model(ws.model_path, args.glassist, ws.project.model_name,
                                                    replace=args.replace, force=args.force))
        except glassist.ModelExistsError as exc:
            emit('error', msg=f'{exc}. Use --force to replace it.', code='model_exists')
            return 3
    return 0


def main(argv=None):
    global JSON_MODE
    args = build_parser().parse_args(argv)
    JSON_MODE = args.json
    try:
        return run(args)
    except Exception as exc:  # report instead of a bare traceback; the window shows this text
        emit('error', msg=f'{type(exc).__name__}: {exc}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
