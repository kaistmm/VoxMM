import os, glob, shutil, argparse, yaml, re, json
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import common_utils as cu

PATTERN = r'![^!]*!|\{[^}]*\}|\([^/][^)]*\)|\[[^\]]*\]|<[^>]*>|\w+'

def segment_filter(segment, args):
    global sing, overlap, onscreen, inaud, uncert, interj, diff, durr, wcnt, m_onscreen, p_onscreen

    txt = segment['text']
    duration = segment['end'] - segment['start']

    ###### common filter
    if duration>=args.max_duration or duration<=args.min_duration:
        durr += duration
        return False

    elif segment['singing'] and args.no_singing:
        sing += duration
        return False

    elif segment['overlapped_duration']>0 and args.no_overlap:
        overlap += duration
        return False


    # on-screen filter
    if segment['on-screen']:
        if len(segment['face_track'])!=1 and args.no_multiple_on_screen:
            m_onscreen += duration
            return False

        for track in segment['face_track']:
            if (track['timestamp'][0]-segment['start']>0.01 or segment['end']-track['timestamp'][1]>0.01) and args.no_partially_on_screen:
               p_onscreen += duration
               return False

    else:
        if args.only_on_screen:
            onscreen += duration
            return False

    # background noise 
    if segment['background_noise']!='N/A':
        filtered_noise_list = [n for n in segment['background_noise'].keys() if n in args.background_noise_list]
        if len(filtered_noise_list)>0 and args.no_overlap:
            return False

    ###### script filter
    if args.script_filter:
        splitted_txt = re.findall(PATTERN, txt)
        word_cnt = 0
        for inst in splitted_txt:
            if inst=='(inaudible)' and args.no_inaudible: # inaudible 
                #print("inaudible: ",txt)
                inaud += duration
                return False

            elif "[" in inst:  # uncertain
                if args.no_uncertain:
                    #print("uncertain: ",txt)
                    uncert += duration
                    return False
                if args.count_uncertain_as_word:
                    word_cnt += len(inst.replace("[","").replace("]","").split())

            elif "{" in inst:  # interjection
                if args.no_interjection:
                    #print("interj: ",txt)
                    interj += duration
                    return False
                if args.count_interjection_as_word:
                    word_cnt += len(inst.replace("{","").replace("}","").split())

            elif "<" in inst:  # diffluency
                if args.no_diffluency:
                    #print("diffluency: ",txt)
                    diff += duration
                    return False
                if args.count_diffluency_as_word:
                    word_cnt += len(inst.replace("<","").replace(">","").split())

            elif "(" in inst and "/" in inst:  # numeric
                word_cnt += len(inst.replace("(","").replace(")","").split("/")[-1].split())

            elif "*" in inst:
                return False

            else: # normal word or abbreviation
                word_cnt += 1

        if word_cnt>=args.max_word or word_cnt<=args.min_word:
            #print("word cnt: ",txt)
            wcnt += duration
            return False

    return True


def segment_selection(args):
    global sing, overlap, onscreen, inaud, uncert, interj, diff, durr, wcnt, m_onscreen, p_onscreen

    output_dir = os.path.join(args.output_dir,'segment_list')
    os.makedirs(output_dir, exist_ok=True)
    
    file_list_paths = args.file_list_paths.replace('[','').replace(']','').split(',')
    for file_list_path in file_list_paths:
        file_list_path = file_list_path.strip()
        file_list = []
        with open(file_list_path, 'r') as f:
            for l in f.readlines():
                file_list.append(l.strip())

        selected_cnt = 0
        selected_duration = 0
        selected_onscreen_duration = 0
        selected_spk = []

        total_duration = 0
        total_onscreen_duration = 0
        total_cnt = 0
        total_spk = []

        with open(os.path.join(output_dir,os.path.basename(file_list_path)),"w") as f:
            for fn in tqdm(file_list):
                metadata_path = os.path.join(args.voxmm_dir,'metadata',fn+'.json')

                local_cnt = 0
                local_dur = 0
                
                with open(metadata_path, 'r') as fm:
                    metadata = json.load(fm)
                    cu.version_check(metadata['metadata_version'])

                for seg in metadata['segments']:
                    if not seg['speaker_id'] in total_spk:
                        total_spk.append(seg['speaker_id'])
                    if segment_filter(seg,args):
                        f.write('{} {} \n'.format(fn, seg['segment_index']))
                        selected_cnt += 1
                        selected_duration += seg['end']-seg['start']
                        if seg['on-screen']:
                            selected_onscreen_duration += seg['end']-seg['start']

                        if not seg['speaker_id'] in selected_spk:
                            selected_spk.append(seg['speaker_id'])


                total_duration += metadata['statistics']['utterance_duration']
                total_onscreen_duration += metadata['statistics']['on-screen_duration']
                total_cnt += metadata['statistics']['segment_num']


        print(f'\nResults for {file_list_path}')
        print(f'\nTotal segments in file list ({file_list_path})') 
        print(f'- total segment num: {total_cnt}')
        print(f'- total segment duration: {total_duration/3600:.2f} hrs')
        print(f'- total speaker: {len(total_spk)}')

        print('\nSelected Segments') 
        print(f'- total segment num: {selected_cnt}')
        print(f'- total segment duration: {selected_duration/3600:.2f} hrs')
        print(f'- total speaker: {len(selected_spk)}')

        print('\nExcluded Segments') 
        print(f'- singing: {sing/60:.2f} mins')
        print(f'- overlap: {overlap/60:.2f} mins')
        print(f'- off-screen: {onscreen/60:.2f} mins')
        print(f'- partially on-screen: {p_onscreen/60:.2f} mins')
        print(f'- scene changed on-screen: {m_onscreen/60:.2f} mins')
        print(f'- inaudible: {inaud/60:.2f} mins')
        print(f'- uncertain: {uncert/60:.2f} mins')
        print(f'- interjection: {interj/60:.2f} mins')
        print(f'- diffluency: {diff/60:.2f} mins')
        print(f'- duration: {durr/60:.2f} mins')
        print(f'- word count: {wcnt/60:.2f} mins')
        print('\n','='*20)

    print("Segment Selection Done")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description = "Segment Selector")

    parser.add_argument("--config", type=str,   default=None,   help="config YAML file")
    
    parser.add_argument("--voxmm_dir", type=str,   default="./VoxMM",   help="VoxMM dataset")
    parser.add_argument("--file_list_paths", type=str,   default="VoxMM/split/test.txt, VoxMM/split/train.txt",   help="file list to preprocess")
    parser.add_argument("--output_dir", type=str,   default="./result",   help="path for preprocessed results")

    # common filter
    parser.add_argument("--min_duration", type=float,   default=1.5,    help="minimum duration of segment")
    parser.add_argument("--max_duration", type=float,   default=30,    help="maximum duration of segment")
    parser.add_argument("--background_noise_list", type=str,   default="",    help="list of background noise to exclude")
    parser.add_argument("--no_singing", action="store_true", help="exclude singing segment")
    parser.add_argument("--no_overlap", action="store_true", help="exclude overlapping speech segment")
    parser.set_defaults(no_singing=True)
    parser.set_defaults(no_overlap=True)

    parser.add_argument("--only_on_screen", action="store_true", help="exclude off-screen segment")
    parser.add_argument("--no_partially_on_screen", action="store_true", help="exclude segment that is partially off-screen")
    parser.add_argument("--no_multiple_on_screen", action="store_true", help="exclude segment that has scene changing face track")
    parser.set_defaults(only_on_screen=False)
    parser.set_defaults(no_partially_on_screen=False)
    parser.set_defaults(no_multiple_on_screen=False)


    # script filter
    parser.add_argument("--script_filter", dest="script_filter", action="store_true", help="enable script filter")
    parser.add_argument("--min_word", type=float,   default=3,    help="minimum word of segment")
    parser.add_argument("--max_word", type=float,   default=80,    help="maximum word of segment")
    parser.add_argument("--count_interjection_as_word", action="store_true", help="count interjection as a word")
    parser.add_argument("--count_uncertain_as_word", action="store_true", help="count uncertaion words as a word")
    parser.add_argument("--count_diffluency_as_word", action="store_true", help="cound diffluencies as a word")
    parser.add_argument("--no_inaudible", action="store_true", help="exclude segmnt that has inaudible part")
    parser.add_argument("--no_uncertain", action="store_true", help="exclude segment that has uncertain word part")
    parser.add_argument("--no_interjection", action="store_true", help="exclude segment that has interjection")
    parser.add_argument("--no_diffluency", action="store_true", help="exclude segment that has diffluency")
    parser.set_defaults(script_filter=True)
    parser.set_defaults(count_interjection_as_word=False)
    parser.set_defaults(count_diffluency_as_word=False)
    parser.set_defaults(count_uncertain_as_word=False)
    parser.set_defaults(no_inaudible=True)
    parser.set_defaults(no_uncertain=True)
    parser.set_defaults(no_interjection=False)
    parser.set_defaults(no_diffluency=True)


    args = cu.load_config(parser)

    sing = 0
    overlap = 0
    onscreen = 0
    inaud = 0
    uncert = 0
    interj = 0
    diff = 0
    durr = 0
    wcnt = 0
    m_onscreen = 0
    p_onscreen = 0

    # segment selection
    segment_selection(args)
    

