import os, glob, shutil, argparse, yaml, re, json

VERSION = "1.0.x"

def version_check(version):
    if not version[:-1]==VERSION[:-1]:
        print(f"Warning: This preprocessor is optimized for dataset version {VERSION}, but your dataset version is {version}. Using this preprocessor with a different version could lead to compatibility issues or processing failures.")

def check_overlap(timeline, timestamp_list):
    for timestamp2 in timestamp_list:
        if timestamp2[0]<timeline and timestamp2[1]>timeline:
            return True
        
    return False

def bbox_only_in_screen(bbox):
    return [max(bbox[0],0.0), max(bbox[1],0.0), min(bbox[2],1.0), min(bbox[3],1.0)]



def extract_bbox(timestamp, entities, no_out_screen=True):
    idxs = [i for i in range(len(entities['time_stamp'])) if entities['time_stamp'][i]>=timestamp[0] and entities['time_stamp'][i]<=timestamp[1]]
    if no_out_screen:
        result = [[entities['time_stamp'][i],bbox_only_in_screen(entities['bbox'][i])] for i in idxs]
    else:
        result = [[entities['time_stamp'][i],entities['bbox'][i]] for i in idxs]

    return result

def load_segment_list(segment_list_path):
    segment_dict = {}
    
    with open(segment_list_path,'r') as f:
        for segment in f.readlines():
            fn = segment.split()[0].strip()
            seg_idx = int(segment.split()[1].strip())
            if not fn in segment_dict:
                segment_dict[fn] = []
            segment_dict[fn].append(seg_idx)

    return segment_dict
        

def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError


def load_config(parser):
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, "r") as f:
            yml_config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in yml_config.items():
            if k in args.__dict__:
                if isinstance(v,bool):
                    args.__dict__[k] = v

                elif isinstance(v,list):
                    args.__dict__[k] = ','.join(v)

                else:
                    typ = find_option_type(k, parser)
                    args.__dict__[k] = typ(v)
            else:
                #sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))
                print("Ignored unknown parameter {} in yaml.".format(k))

    return args


def output_destination(filename, segment, output_dir, style='librispeech'):
    if style=='librispeech':
        spk = str(int(segment['speaker_id'].replace("id","")))
        seg_idx = '{:04d}'.format(segment['segment_index'])
        wav_path = os.path.join(output_dir, spk, filename, f'{spk}-{filename}-{seg_idx}.flac')
        video_path = wav_path.replace(".flac",".mp4")
        txt_path = os.path.join(output_dir, spk, filename, f'{spk}-{filename}.trans.txt')

    elif style=='lrs3':
        seg_idx = '{:05d}'.format(segment['segment_index'])
        wav_path = os.path.join(output_dir, filename, f'{seg_idx}.wav')
        video_path = wav_path.replace(".wav",".mp4")
        txt_path = wav_path.replace(".wav",".txt")

    else:
        raise Exception(f"Invalid dataset style input: {style}")

    return wav_path, video_path, txt_path


def parse_face_track_json(face_track_json):
    parsed_face_track = {}
    for tr in face_track_json['face_tracks']:
        tidx = tr['track_index']
        parsed_face_track[tidx] = {}
        for i, f in enumerate(tr['frame']):
            parsed_face_track[tidx][f] = tr['bbox'][i]
    
    return parsed_face_track


def resample_face_track(face_track_json, trg_fps):
    parsed_face_track = parse_face_track_json(face_track_json)
    json_fps = face_track_json['FPS']
    face_track_json['FPS'] = trg_fps
    new_face_tracks = []
    for tr in face_track_json['face_tracks']:
        tidx = tr['track_index']
        start_f = tr['frame'][0]
        end_f = tr['frame'][-1]
        face_tracks = {'track_index':tidx, 'frame':[],'bbox':[],'time_stamp':[]}
        for f in range(int(start_f*trg_fps/json_fps),int(end_f*trg_fps/json_fps)):
            face_tracks['frame'].append(f)
            face_tracks['bbox'].append(parsed_face_track[tidx][min(max(int(f*json_fps/trg_fps),start_f),end_f)])
            face_tracks['time_stamp'].append(f/trg_fps)
        new_face_tracks.append(face_tracks)
    face_track_json['face_tracks'] = new_face_tracks
            
    return face_track_json
