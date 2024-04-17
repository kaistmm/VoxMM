import os, glob, shutil, argparse, yaml, re, json
import librosa
from tqdm import tqdm

from multiprocessing import Pool

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import common_utils as cu
from utils import script_utils as su
from utils import wav_utils as wu
from utils import video_utils as vu

_warning = False

def worker(buff):
    args = buff['args']
    metadata = buff['metadata']
    fn = metadata['video_infos']['file_name']
    fidx = metadata['video_infos']['index']
    output_dir = buff['output_dir']
    wav_path = os.path.join(args.voxmm_dir,'wav',fn+'.wav')
    vid_path = os.path.join(args.voxmm_dir,'video',fn+'.mp4')
    face_track_path = os.path.join(args.voxmm_dir,'face_track',fn+'.json')

    if not args.no_wav:
        src_wav, sr = librosa.load(wav_path, sr=args.sample_rate)

    if not args.no_video:
        vid_gen = vu.Face_Track_Generator(vid_path, face_track_path, **vars(args))
        if args.audio_in_video:
            if args.no_wav:
                vid_gen.load_audio_from_path(wav_path, args.sample_rate)
            else:
                vid_gen.load_audio(src_wav, sr)

    if not args.no_script:
        script_gen = su.Script_Generator(**vars(args))

    for seg in metadata['segments']:
        if args.use_vid_index:
            wav_path, video_path, txt_path = cu.output_destination(str(fidx), seg, output_dir, args.dataset_style)
        else:
            wav_path, video_path, txt_path = cu.output_destination(fn, seg, output_dir, args.dataset_style)

        try:
            # wav generation
            if not args.no_wav:
                os.makedirs(os.path.dirname(wav_path),exist_ok=True)
                wu.crop_wav(src_wav, seg['start'], seg['end'], sr, args.volume, wav_path)

            # vid generation
            if not args.no_video:
                os.makedirs(os.path.dirname(video_path),exist_ok=True)
                if len(seg['face_track'])==1:
                    vid_gen(seg['face_track'][0]['index'], seg['start'], seg['end'], video_path, args.volume)
                else:
                    print("Zero or multiple face track detected. Skip this segment. file: {}, seg idx: {}".format(fn, seg['segment_index']))
                    continue


            # txt generation
            if not args.no_script:
                os.makedirs(os.path.dirname(txt_path),exist_ok=True)
                if args.dataset_style=='librispeech':
                    with open(txt_path, 'a') as f:
                        f.write('{} {}\n'.format(os.path.basename(wav_path).replace('flac',''),script_gen(seg['text'])))

                elif args.dataset_style=='lrs3':
                    with open(txt_path, 'w') as f:
                        f.write('Text:  {}\n'.format(script_gen(seg['text'])))

                else:
                    raise Exception(f"Invalid dataset style input: {style}")

        except Exception as e:
            print('Error occur while processing the segment. Skip this segment. file: {}, seg idx: {}'.format(fn,seg['segment_index']))
            print(f'{str(e)}')


    if 'vid_gen' in locals():
        del vid_gen
    if 'src_wav' in locals():
        del src_wav
    if 'script_gen' in locals():
        del script_gen



def asr_preprocessor(args):
    metadata_dir = os.path.join(args.voxmm_dir,'metadata')
    segment_list_paths = args.segment_list_paths.replace('[','').replace(']','').split(',')
    cnt = 0
    buff = []
    for segment_list_path in segment_list_paths:
        print(f'\nPreprocessing start for {segment_list_path}')
        output_dir = os.path.join(args.output_dir, os.path.splitext(os.path.basename(segment_list_path))[0])
        os.makedirs(output_dir,exist_ok=True)

        segment_dict = cu.load_segment_list(segment_list_path.strip())
        for fn in segment_dict.keys():
            with open(os.path.join(metadata_dir,fn+'.json'), 'r') as fm:
                metadata =  json.load(fm)
                cu.version_check(metadata['metadata_version'])

            metadata['segments'] = [seg for seg in metadata['segments'] if seg['segment_index'] in segment_dict[fn]]
            if len(metadata['segments'])>0:
                buff.append({"args": args, "metadata": metadata, "output_dir": output_dir})
                cnt += 1
                

    p = Pool(args.num_worker)
    with tqdm(total=len(buff)) as pbar:
        for _ in tqdm(p.imap_unordered(worker,buff)):
            pbar.update()
    p.close()
    p.join()

    print(f'{cnt} files processed')
    print('='*20)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description = "ASR Preprocessor")

    parser.add_argument("--config", type=str,   default=None,   help="config YAML file")
    parser.add_argument("--num_worker", type=int,   default=1,   help="number of process")
    
    parser.add_argument("--voxmm_dir", type=str,   default="./VoxMM",   help="VoxMM dataset")
    parser.add_argument("--segment_list_paths", type=str,   default="./results/A-ASR/segment_list/test.txt, ./results/A-ASR/segment_list/train.txt",   help="path list of segment list to preproces")
    parser.add_argument("--output_dir", type=str,   default="./results/A-ASR",   help="path for preprocessed results")
    
    # dataset style
    parser.add_argument("--dataset_style", type=str, choices=['librispeech','lrs3'], default="lrs3", help="resulting dataset format")
    parser.add_argument("--use_vid_index", "-vid", action="store_true", help="use video index instead of video name")

    # script setting
    parser.add_argument("--no_script", action="store_true", help="do not generate script file")
    parser.add_argument("--capitalize", action="store_true", help="capitalize script")
    parser.add_argument("--apostrophe", action="store_true", help="leave apostrophe")
    parser.add_argument("--hypen", action="store_true", help="leave hypen")
    parser.add_argument("--space_on_abbreviation", action="store_true", help="give a space between alphabet for abbreviation")
    parser.add_argument("--numeric_format", type=str, choices=['digit','pronunciation'],  default="pronunciation",   help="numeric format")
    parser.add_argument("--default_inaudible_process", type=str, choices=['drop','token'],  default="drop",   help="default process for inaudibles.")
    parser.add_argument("--default_uncertain_process", type=str, choices=['word','drop','token'],  default="drop",   help="default process for uncertain word")
    parser.add_argument("--default_diffluency_process", type=str, choices=['word','drop','token'],  default="drop",   help="default process for diffluencies")
    parser.add_argument("--default_interjection_process", type=str, choices=['word','drop','token'],  default="drop",   help="default process for interjections")
    parser.add_argument("--interjection_to_word", type=str,   default=",",   help="list of interjections to convert into normal word")
    parser.add_argument("--interjection_to_token", type=str,   default=",",   help="list of interjections to convert into token")
    parser.add_argument("--interjection_to_drop", type=str,   default=",",   help="list of interjections to drop")
    parser.add_argument("--inaudible_token", type=str,   default="<unknown>",   help="default token for inaudible words")
    parser.add_argument("--uncertain_token", type=str,   default="<unknown>",   help="default token of uncertain words")
    parser.add_argument("--diffluency_token", type=str,   default="<interjection>",   help="default token of diffluencies")
    parser.add_argument("--interjection_token", type=str,   default="<interjection>",   help="default token of interjection")

    # wav setting
    parser.add_argument("--no_wav", action="store_true", help="do not generate cropped wav file")
    parser.add_argument("--sample_rate", type=int,   default=16000,   help="sample rate")
    parser.add_argument("--volume", type=float,   default=-16,   help="target Audio RMS in dBFS scale")

    # video setting
    parser.add_argument("--no_video", action="store_true", help="do not generate face track video file")
    parser.add_argument("--track_size", type=int,   default=224,   help="size of resulting face track video")
    parser.add_argument("--track_framerate", type=int,   default=25,   help="frame rate of resulting face track video")
    parser.add_argument("--audio_in_video", action="store_true", help="audio in video")

    parser.set_defaults(use_vid_index=False)
    parser.set_defaults(no_wav=False)

    parser.set_defaults(no_script=False)
    parser.set_defaults(capitalize=True)
    parser.set_defaults(apostrophe=True)
    parser.set_defaults(hypen=True)
    parser.set_defaults(space_on_abbreviation=False)

    parser.set_defaults(no_video=True)
    parser.set_defaults(audio_in_video=True)

    args = cu.load_config(parser)

    # asr dataset preprocessor
    asr_preprocessor(args)
    

