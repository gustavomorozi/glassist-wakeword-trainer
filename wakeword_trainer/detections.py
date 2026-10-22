"""Turn the clips GLaSSIST saved on every wake-word trigger into training data, labelled by what happened next.

GLaSSIST (with "save detections" on) writes trig_YYYYMMDD_HHMMSS_<word>_<score>.wav on every trigger and
logs what the voice pipeline did afterwards:
  * "Pipeline error: stt-no-text-recognized"  -> nobody said anything: a FALSE trigger -> real_neg
  * the pipeline understood a command         -> a TRUE trigger (only with include_positives) -> real_pos
  * anything else / no log line               -> unknown: skipped (never guess)
This is "personalised negatives": the sounds YOUR house makes, found by the model itself.
"""
import datetime as dt
import glob
import os
import re
import shutil

CLIP = re.compile(r'trig_(\d{8})_(\d{6})_.+?_(\d{3})\.wav$')
STAMP = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
NO_TEXT = re.compile(STAMP + r'.*Pipeline error: stt-no-text-recognized')
UNDERSTOOD = re.compile(STAMP + r'.*(VOICE_ASSISTANT_INTENT_END|VOICE_ASSISTANT_TTS_START)')
OTHER_ERROR = re.compile(STAMP + r'.*Pipeline error: (?!stt-no-text-recognized)')


def _ts(s):
    return dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


def read_events(log_files):
    """[(time, kind)] with kind in 'no_text', 'understood', 'other_error', sorted by time."""
    events = []
    for path in log_files:
        with open(path, encoding='utf-8', errors='replace') as f:
            for line in f:
                for rx, kind in ((NO_TEXT, 'no_text'), (UNDERSTOOD, 'understood'), (OTHER_ERROR, 'other_error')):
                    m = rx.search(line)
                    if m:
                        events.append((_ts(m.group(1)), kind))
                        break
    return sorted(events)


def clip_time(path):
    m = CLIP.search(os.path.basename(path))
    return dt.datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S') if m else None


def label(clip_t, events, window_s=30):
    """'false' | 'true' | None. The first pipeline event within `window_s` seconds AFTER the clip decides."""
    for t, kind in events:
        if t < clip_t - dt.timedelta(seconds=2):
            continue
        if t > clip_t + dt.timedelta(seconds=window_s):
            break
        return {'no_text': 'false', 'understood': 'true'}.get(kind)
    return None


def import_detections(ws, detections_dir, log_files, include_positives=False, window_s=30):
    """Copy labelled clips into the workspace. Returns counts; never overwrites existing files."""
    events = read_events(log_files)
    out = {'false_to_real_neg': 0, 'true_to_real_pos': 0, 'unknown': 0, 'already_there': 0, 'no_timestamp': 0}
    for clip in sorted(glob.glob(os.path.join(detections_dir, 'trig_*.wav'))):
        t = clip_time(clip)
        if t is None:
            out['no_timestamp'] += 1
            continue
        verdict = label(t, events, window_s)
        stem = os.path.splitext(os.path.basename(clip))[0]
        if verdict == 'false':
            dest = os.path.join(ws.real_neg, f'real_env_det_{stem}.wav')
            key = 'false_to_real_neg'
        elif verdict == 'true' and include_positives:
            dest = os.path.join(ws.real_pos, f'real_det_{stem}.wav')
            key = 'true_to_real_pos'
        else:
            out['unknown'] += 1
            continue
        if os.path.exists(dest):
            out['already_there'] += 1
        else:
            shutil.copyfile(clip, dest)
            out[key] += 1
    return out
