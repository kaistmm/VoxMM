import os, glob, shutil, argparse, yaml, re, json
import librosa
import soundfile
import subprocess
import numpy as np

_TARGET_VOLUME = -16 # dBFS
_SAMPLE_RATE = 16000 # Hz

def normalize_volume(wav, target_rms):
    current_rms = np.sqrt(np.mean(wav**2))
    scaling_factor = target_rms / current_rms

    return wav * scaling_factor

def crop_wav(src_wav, start, end, sr=_SAMPLE_RATE, volume=_TARGET_VOLUME, trg_path=None):
    start = int(start*sr)
    end = int(end*sr)
    if end>len(src_wav):
        raise Exception(f"end timestamp longer than wav length: {end} vs {len(src_wav)}")

    cropped_wav = src_wav[start:end]
    target_rms = db2linear(volume)
    normalized_wav = normalize_volume(cropped_wav, target_rms)

    if not trg_path is None:
        soundfile.write(trg_path, normalized_wav, sr, 'PCM_24')
    else:
        return normalized_wav


def db2linear(db):
    return 10 ** (db/20)


def save_wav_from_video(vid_path, output_path, sample_rate=_SAMPLE_RATE):
    cmd = "ffmpeg -y -i {} -ac 1 -vn -acodec pcm_s16le -ar {} {}".format(vid_path, sample_rate, output_path)

    if subprocess.check_call(cmd,shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)!=0:
        raise Exception("wav extraction failed: {vid_path}")

