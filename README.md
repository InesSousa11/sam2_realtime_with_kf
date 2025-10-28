# SAM2 Real-Time Detection and Tracking

This repository adapts **[SAM2](https://github.com/facebookresearch/sam2)** to include real-time multi-object tracking. 
It allows users to specify and track a fixed number of objects in real time, integrating motion modeling 
from **[SAMURAI](https://github.com/yangchris11/samurai)** for improved tracking in complex scenarios.  

### About SAM2
**SAM2** (Segment Anything Model 2) is designed for object segmentation and tracking but lacks built-in capabilities 
for performing this in real time.

### About SAMURAI
**SAMURAI** enhances SAM2 by introducing motion modeling, leveraging temporal motion cues for better 
tracking accuracy without retraining or fine-tuning.  

#### Note on SAMURAI Features
While this repository integrates SAMURAI's motion modeling, it does not support the motion-aware memory selection mechanism, 
as conditional operations applied to memory would simultaneously affect all tracked objects.

The core implementation of SAMURAI's motion modeling, originally found in 
[sam2_base.py](https://github.com/yangchris11/samurai/blob/master/sam2/sam2/modeling/sam2_base.py) in SAMURAI's 
repository, has been relocated to 
[sam2_object_tracker.py](https://github.com/zdata-inc/sam2_realtime/blob/main/sam2/sam2_object_tracker.py) in this 
repository to maintain the original SAM2 codebase.

## Key Features

- **Real-Time Tracking**: Modified SAM2 to track a fixed number of objects in real time.
- **Motion-Aware Memory**: SAMURAI leverages temporal motion cues for robust object tracking without retraining or fine-tuning.

The core implementation resides in `sam2_object_tracker.py`, where the number of objects to track must be specified during instantiation.

---

## Setup Instructions

### 1. Create Conda Environment
```bash
conda create -n sam2 python=3.10.15
```

### 2. Clone SAM2 Realtime Repo
```
git clone https://github.com/zdata-inc/sam2_realtime
```

### 2. Install SAM2
```
cd sam2_realtime
pip install -e .
pip install -e ".[notebooks]"
```


### 4. Download SAM2 Checkpoints
```bash
cd checkpoints
./download_ckpts.sh
cd ..
```

---

## Usage
### Demo
Run the demo notebook to visualize real-time SAM2 object tracking in action:  
`notebooks/realtime_detect_and_track.ipynb`


### Acknowledgment
SAM2 Realtime extends [SAM 2](https://github.com/facebookresearch/sam2) by Meta FAIR, designed to enable real-time tracking. 
It integrates motion-aware segmentation techniques developed in [SAMURAI](https://github.com/yangchris11/samurai) by 
the Information Processing Lab at the University of Washington.


## Citation
```
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, 
  Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, 
  Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, 
  Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}

@misc{yang2024samurai,
      title={SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory}, 
      author={Cheng-Yen Yang and Hsiang-Wei Huang and Wenhao Chai and Zhongyu Jiang and Jenq-Neng Hwang},
      year={2024},
      eprint={2411.11922},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.11922}, 
}
```

---
