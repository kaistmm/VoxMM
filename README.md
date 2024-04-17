# VoxMM: Rich transcription of conversations in the wild

VoxMM is an audio-visual dataset containing rich transcriptions of spoken conversations from diverse domains of YouTube video clips. You can find updates and additional information about the dataset on our [website](https://mm.kaist.ac.kr/projects/voxmm).

One of the main features of the VoxMM dataset is its abundant metadata, which enables the creation of datasets for various tasks. This repository contains the basic preprocessing codes for the VoxMM dataset, suitable for Audio-only ASR/Diarisation and Audio-visual ASR/Diarisation. 

The preprocessor is optimized for dataset version v1.0.x. Using this preprocessor with a different version may result in compatibility issues or processing failures.

### Dependencies
```
pip install -r requirements.txt
```
In addition, `ffmpeg` is required for AV-ASR preprocessing.

### Dataset Download
The dataset is available for download on our [website](https://mm.kaist.ac.kr/projects/voxmm). The default dataset folder for this preprocessor is `VoxMM/`. If the dataset is stored in a different location, create symbolic links to `VoxMM/` or modify the `voxmm_dir` configuration in the config file.

### Data preparation
The preprocessing consists of two stages: selecting speech segments from metadata under specific conditions using `segment_selector.py`, and converting these selected segments into a dataset in the desired format using `asr_preprocessor.py` or `diar_preprocessor.py`. All code settings can be configured in the `config` file, with four default config files provided. These configuration were used to create the datasets used in the experiments described in our paper.

Below are examples of how to create four types of datasets.
#### Audio-only ASR
Use the following commands to create a LibriSpeech-style dataset. 
```
python ./tools/segment_selector.py --config='./configs/A-ASR.yaml'
python ./tools/asr_preprocessor.py --config='./configs/A-ASR.yaml'
```
#### Audio-only Diarisation
Use the following commands to create a VoxConverse-style dataset.
```
python ./tools/segment_selector.py --config='./configs/A-Diar.yaml'
python ./tools/diar_preprocessor.py --config='./configs/A-Diar.yaml'
```

#### Audio-visual ASR
Use the following commands to create an LRS3-style dataset.  
```
python ./tools/segment_selector.py --config='./configs/AV-ASR.yaml'
python ./tools/asr_preprocessor.py --config='./configs/AV-ASR.yaml'
```

#### Audio-visual Diarisation
Use the following commands to create an AVA-AVD-style dataset. Note that the generated `tracks/` might not be 100% compatible with AVA-AVD and AVA Spoken Activity Datasets. For more information and preprocessing methods for the AVA-AVD dataset, please refer to this [link](https://github.com/zcxu-eric/AVA-AVD).
```
python ./tools/segment_selector.py --config='./configs/AV-Diar.yaml'
python ./tools/diar_preprocessor.py --config='./configs/AV-Diar.yaml'
```

### Metadata Version Log

**v1.0.0:** Initial release with the same videos as v0.0.0 but minor changes in the split. Improved label quality through additional manual refinement. Interjections and disfluencies are now distinguished in the script labels. Metadata structure revised  for enhanced usability. Some attributes removed to address privacy concerns.

**v0.0.0:** Prototype version introduced in the ICASSP paper. Not released publicly.



### Citation
Please cite the following if you make use of the dataset.

```
@article{kwak2024voxmm,
title={VoxMM: Rich transcription of conversations in the wild},
author={Kwak, Doyeop and Jung, Jaemin and Nam, Kihyun and Jang, Youngjoon and Jung, Jee-won and Watanebe, Shinji and Chung, Joon Son},
booktitle={International Conference on Acoustics, Speech, and Signal Processing},
year={2024}
}
```

### License
The VoxMM dataset is available to download for research purposes under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0). The copyright remains with the original owners of the video.

In order to collect videos that simulate in-the-wild scenarios from as many diverse domains as possible, the dataset includes sensitive content such as political debates and news. The views and opinions expressed by the speakers in the dataset are those of the individual speakers and do not necessarily reflect the positions of the Korea Advanced Institute of Science and Technology (KAIST) or the authors of the paper.

We would also like to note that the distribution of identities in this dataset may not be representative the global human population. Please be careful of unintended societal, gender, racial, linguistic and other biases when training or deploying models trained on this data.

### Acknowledgement
This work was supported by Institute of Information \& communications Technology Planning \& Evaluation (IITP) grant funded by the Korea government (MSIT, 2022-0-00989).

