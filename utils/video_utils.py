import cv2
import os
import random
import json
import soundfile
import librosa
import subprocess
import numpy as np
from multiprocessing import Pool

from utils import wav_utils as wu
from utils import common_utils as cu


def get_frames(cap):
    frames = []
    while True:
        ret, frame = cap.read()
        if not frame is None:
            frames.append(frame)
        if not ret:
            break
        
    return frames


def resample_frames(frames, src_fps, trg_fps):
    src_len = len(frames)
    trg_len = int(src_len*trg_fps/src_fps)
    resampled = []
    for trg_f in range(trg_len):
        resampled.append(frames[min(int(trg_f*src_fps/trg_fps),src_len-1)])

    return resampled


def crop_frame(frame, bbox, frame_size=None, padding=True):
    H, W, _ = frame.shape
    x1 = int(bbox[0]*W)
    y1 = int(bbox[1]*H)
    x2 = int(bbox[2]*W)
    y2 = int(bbox[3]*H)
    
    if padding:
        out_of_frame = max(-x1,-y1,x2-W,y2-H,0)
        if out_of_frame>0:
            pad_size = out_of_frame+10
            frame_padded = np.pad(frame,((pad_size,pad_size),(pad_size,pad_size),(0,0)), 'constant', constant_values=(110,110))
            cropped_frame = frame_padded[y1+pad_size:y2+pad_size,x1+pad_size:x2+pad_size]
        else:
            cropped_frame = frame[y1:y2,x1:x2,:]
    else:
        cropped_frame = frame[y1:y2,x1:x2,:]

            
    if not frame_size is None:
        cropped_frame = cv2.resize(cropped_frame,frame_size)
    
    return cropped_frame


def convert_video(args):
    src_path, output_path, fps = args
    cmd = f"ffmpeg -y -i '{src_path}' -c:v libx264 -crf 18 -preset veryslow -r {fps} '{output_path}'"
    print(f"Converting {src_path} to {output_path}...")
    if subprocess.check_call(cmd,shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)!=0:
        raise Exception(f"Converting failed. {src_path}")


def convert_videos(src_dir, output_dir, fps, num_processes=32):
    if not os.path.isdir(src_dir):
        print(f"Source directory '{src_dir}' does not exist.")
        return
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    tasks = []
    for file in os.listdir(src_dir):
        if file.endswith('.mp4'):
            src_file = os.path.join(src_dir, file)
            output_file = os.path.join(output_dir, os.path.splitext(file)[0] + '.mp4')
            tasks.append((src_file, output_file, fps))

    with Pool(num_processes) as pool:
        pool.map(convert_video, tasks)

    print("Conversion completed.")


class Face_Track_Generator():
    def __init__(self,
        video_path,
        face_track_json_path,
                 
        track_size=256,
        track_framerate=25,
        codec=cv2.VideoWriter_fourcc(*'mp4v'),
        **kwargs
        ):      
        if os.path.isfile(video_path): 
            self.cap = cv2.VideoCapture(video_path)
            self.src_nm, self.src_ext = os.path.splitext(os.path.basename(video_path))
            self.src_path = video_path

        else:
            raise Exception(f"File not exist: {video_path}")  
            
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.src_len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.codec = codec

        with open(face_track_json_path,'r') as f:
            face_track_json = json.load(f)
            
        if int(face_track_json['FPS'])!=self.fps:
            face_track_json = cu.resample_face_track(face_track_json,self.fps)
        self.face_tracks = cu.parse_face_track_json(face_track_json)
        
        self.trg_size = (track_size, track_size)
        self.trg_fps = int(track_framerate)
        self.audio_in_track = False
        
    def load_audio_from_path(self, audio_path, sample_rate):
        if not os.path.isfile(audio_path):
            print(f"Cannot find wav file from path {audio_path}. Extract tmp wav from video")
            audio_path = self.src_path.replace(self.src_ext,'_t.wav')
            wu.save_wav_from_video(self.src_path, audio_path, sample_rate)
            print(f"Tmp wav extraction completed. {self.src_path} => {audio_path}")
            self.src_wav, self.sr = librosa.load(audio_path,sr=sample_rate)
            os.remove(audio_path)
            print(f"Tmp wav removed. {audio_path}")
            self.audio_in_track = True

        else:
            self.src_wav, self.sr = librosa.load(audio_path,sr=sample_rate)
            self.audio_in_track = True

    def load_audio(self, np_audio, sample_rate, **kwargs):
        self.src_wav = np_audio
        self.sr = sample_rate
        self.audio_in_track = True

    
    def get_face_crop(self, track_id, start, end):
        face_track = self.face_tracks[track_id]
        src_idxs = [f for f in range(self.src_len) if f>=int(start*self.fps) and f<=int(end*self.fps)]
        f_first = min(list(face_track.keys()))
        f_last = max(list(face_track.keys()))
        face_crops = []
        for idx in src_idxs:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            _, src_frame = self.cap.read()

            try:
                if src_frame is None:
                    raise Exception(f"No frame detected in the video. {track_id} {start}~{end}")
                bbox = face_track[min(max(idx,f_first),f_last)]
                face_crops.append(crop_frame(src_frame, bbox, self.trg_size))
                
            except:       
                raise Exception(f"face cropping failed. {str(e)}")
        
        if self.fps!=self.trg_fps:
            face_crops = resample_frames(face_crops, self.fps, self.trg_fps)
            
        return face_crops
        
                
    def save_video(self, frames, output_path, audio=None):
        # audio: numpy audio 
        vid_nm, vid_ext = os.path.splitext(output_path)
        tmp_wav_path = output_path.replace(vid_ext,"_t.wav")
        tmp_vid_path = output_path.replace(vid_ext,'_t'+vid_ext)
        frames_len = len(frames)
        
        if not audio is None and self.audio_in_track:
            out_writer = cv2.VideoWriter(tmp_vid_path, self.codec, self.trg_fps, self.trg_size)
        else:
            out_writer = cv2.VideoWriter(output_path, self.codec, self.trg_fps, self.trg_size)

        try:
            for frame in frames:
                out_writer.write(frame)
            out_writer.release()
            if not audio is None and self.audio_in_track:
                soundfile.write(tmp_wav_path, audio, self.sr,'PCM_24')
                
                cmd = 'ffmpeg -y -i {} -i {} -c:v copy -c:a aac {}'.format(tmp_vid_path,tmp_wav_path,output_path)
                if subprocess.check_call(cmd,shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)!=0:
                    raise Exception(f"merging audio & video failed. {output_path}")
        
        except Exception as e:
            raise Exception(f"video saving failed. {str(e)}")
        
        finally:            
            if os.path.isfile(tmp_wav_path):
                os.remove(tmp_wav_path)
            if os.path.isfile(tmp_vid_path):
                os.remove(tmp_vid_path)
                
    def __call__(self, track_id, start, end, output_path, volume=-16):
        face_crops = self.get_face_crop(track_id, start, end)
        if self.audio_in_track:
            wav_crop = wu.crop_wav(self.src_wav, start, end, self.sr, volume)
            self.save_video(face_crops, output_path, wav_crop)
            del wav_crop

        else:
            self.save_video(face_crops, output_path)
        del face_crops

