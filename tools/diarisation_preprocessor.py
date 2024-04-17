import os, glob, shutil, argparse, yaml, re, json
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import common_utils as cu

_warning = False

def diarisation_preprocessor(args):
    rttms_dir = os.path.join(args.output_dir,'rttms')
    labs_dir = os.path.join(args.output_dir,'labs')
    split_dir = os.path.join(args.output_dir,'split')
    os.makedirs(rttms_dir, exist_ok=True)
    os.makedirs(labs_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)
    
    segment_list_paths = args.segment_list_paths.replace('[','').replace(']','').split(',')
    for segment_list_path in segment_list_paths:
        print(f'\nPreprocessing start for {segment_list_path}')
        segment_dict = cu.load_segment_list(segment_list_path.strip())
        with open(os.path.join(split_dir,os.path.basename(segment_list_path)),'w') as fs:
            cnt = 0
            for fn in tqdm(segment_dict.keys()):
                with open(os.path.join(args.voxmm_dir,'metadata',fn+'.json'), 'r') as fm:
                    metadata =  json.load(fm)
                    cu.version_check(metadata['metadata_version'])

                selected_segment = [seg for seg in metadata['segments'] if seg['segment_index'] in segment_dict[fn]]
                rttm_list = []
                spk_dict = {}
                for seg in selected_segment:
                    spk = seg['speaker_id']
                    start = seg['start']
                    end = seg['end']

                    if len(seg['face_track'])==1 and not args.no_track:
                        entity = seg['face_track']
                    else:
                        entity = None

                    if not spk in spk_dict:
                        spk_dict[spk] = f'spk{len(spk_dict):02d}'

                    rttm_list.append([start,end,spk_dict[spk],entity])

                if len(rttm_list)>0:
                    rttm_list = sorted(rttm_list, key=lambda k:k[0])
                    fs.write(f'{fn}\n')
                    cnt += 1

                    # rttm generation
                    if not args.no_rttm:
                        with open(os.path.join(rttms_dir,fn+'.rttm'), 'w') as fr:
                            for r in rttm_list:
                                fr.write("SPEAKER {} 1 {:6f} {:6f} <NA> <NA> {} <NA> <NA>\n".format(fn, r[0], r[1]-r[0], r[2]))

                    # lab generation (oracle vad)
                    if not args.no_lab:
                        lab_list = []
                        start = -1
                        end = -1
                        for seg in rttm_list:
                            if cu.check_overlap(seg[0],lab_list) and cu.check_overlap(seg[1],lab_list):
                                continue
                            if start==-1:
                                start = seg[0]
                            if end<seg[1]:
                                end = seg[1]
                            if not cu.check_overlap(end,rttm_list):
                                lab_list.append([start,end])
                                start = -1

                        with open(os.path.join(labs_dir,fn+'.lab'),'w') as fl:
                            for lab in lab_list:
                                fl.write('{:.6f} {:.6f} speech\n'.format(lab[0],lab[1]))

                    # track generation 
                    if not args.no_track:
                        tracks_dir = os.path.join(args.output_dir,'tracks')
                        os.makedirs(tracks_dir, exist_ok=True)
                        with open(os.path.join(args.voxmm_dir,'face_track',fn+'.json'),'r') as ftr:
                            face_tracks = json.load(ftr)

                        with open(os.path.join(tracks_dir,fn+'-activespeaker.csv'),'w') as ft:
                            for r in rttm_list:
                                if r[3]!=None:
                                    for track in r[3]:
                                        track_id = track['index']
                                        bbox = cu.extract_bbox([r[0],r[1]],face_tracks['face_tracks'][track_id])
                                        if len(bbox)==0 and _warning:
                                            print('Warning: face track cannot cover entire segment. filename: {}, segment: {:.2f}s ~ {:.2f}s {}, face track: {:.2f}s ~ {:.2f}s'.format(fn, 
                                                                                                                                                                                        r[0], 
                                                                                                                                                                                        r[1], 
                                                                                                                                                                                        r[2], 
                                                                                                                                                                                        face_tracks['face_tracks'][track_id]['time_stamp'][0],
                                                                                                                                                                                        face_tracks['face_tracks'][track_id]['time_stamp'][-1]
                                                                                                                                                                                        ))
                                        for bb in bbox:
                                            ft.write("{},{},{},{},{},{},SPEAKING_AUDIBLE,{}:{},{}\n".format(fn,
                                                                                                            bb[0],
                                                                                                            bb[1][0],
                                                                                                            bb[1][1],
                                                                                                            bb[1][2],
                                                                                                            bb[1][3],
                                                                                                            fn,
                                                                                                            track_id,
                                                                                                            r[2]
                                                                                                            ))
        print(f'{cnt} files processed')
        print('='*20)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description = "Diarisation Preprocessor")

    parser.add_argument("--config", type=str,   default=None,   help="config YAML file")
    
    parser.add_argument("--voxmm_dir", type=str,   default="./VoxMM/",   help="VoxMM dataset")
    parser.add_argument("--segment_list_paths", type=str,   default="./result/AV-Diar/segment_list/test.txt, ./result/AV-Diar/segment_list/train.txt",   help="path list of segment list file")
    parser.add_argument("--output_dir", type=str,   default="./result/A-Diar",   help="path for preprocessed results")
    parser.add_argument("--no_track", action="store_true", help="do not generate face track")
    parser.add_argument("--no_rttm", action="store_true", help="do not generate rttm file")
    parser.add_argument("--no_lab", action="store_true", help="do not generate oracle VAD file")
    parser.set_defaults(no_track=True)
    parser.set_defaults(no_rttm=False)
    parser.set_defaults(no_lab=False)

    args = cu.load_config(parser)

    # diarisation dataset preprocessor
    diarisation_preprocessor(args)
    

