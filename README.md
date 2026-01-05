# Multi-task Adversarial Attacks for Autonomous Driving in Different Weather Conditions

This repository contains the source code of Multi-Task Learning with Min-max Optimization and baselines from the following papers:
1) [Robust Multi-task Adversarial Attacks Using Min-max Optimization](https://ieeexplore.ieee.org/abstract/document/10888541)) (Conference Version, In proceedings of ICASSP 2025);
2) Multi-task Adversarial Attacks for Autonomous Driving in Different Weather Conditions (Extended Journal Version, Accepted by Applied Intelligence, 2026).

All models were written in `PyTorch`. 

## Datasets
We implemented all weighting baselines presented in the paper for multiple computer vision tasks: Dense Prediction Tasks (for Cityscapes, Rainy Cityscapes, Foggy Cityscapes, and NYUv2).

### Cityscapes, Foggy Cityscapes, and Rainy Cityscapes

- `Cityscapes`, `Foggy Cityscapes`, and `Rainy Cityscapes` [3 Tasks]: 13 Class Semantic Segmentation + 4 Class Part Segmentation + Disparity Estimation. [512 x 256] Resolution.
- `NYUv2` [3 Tasks]: 13 Class Segmentation + Depth Estimation + Surface Normal Prediction. [288 x 384] Resolution.

The `Cityscapes` and `Foggy Cityscapes` datasets with ground truths could be downloaded from the official website [here](https://www.cityscapes-dataset.com/downloads/).

The `Rainy Cityscapes` dataset is generated following the instructions of [this paper](https://ieeexplore.ieee.org/abstract/document/10574400). Please refer to `Readme.md` --> `DATA` --> `1. Download the datasets`:

- Download the [Rainy Mask](https://github.com/tsingqguo/efficientderain) (rainmix/Streaks_Garg06.zip);
- Set your paths for rainy mask and Cityscape dataset in the code dataset/generate_rainy_cityscape.py, then to generate the `Rain Cityscapes` dataset.

All the images and labels are resized to [512 x 256] resolution.

### NYUv2
Please download the pre-processed `NYUv2` dataset [here](https://www.dropbox.com/scl/fo/p7n54hqfpfyc6fe6n62qk/AKVb28ZmgDiGdRMNkX5WJvo?rlkey=hcf31bdrezqjih36oi8usjait&e=1&dl=0) which is evaluated in the papers. (Moreover, if you are more interested, the raw 13-class NYUv2 dataset can be downloaded [here](https://github.com/ankurhanda/nyuv2-meta-data) with segmentation labels defined [here](https://github.com/ankurhanda/SceneNetv1.0/). )
All the images and labels are resized to [288 x 384] resolution.

## Experiments

### Basic Settings

- Please use `dataset/trainer_dense.py` to train and save the pretrain model for the above datasets.
- Please use `trainer_robust.py` to run the project for the above datasets in multiple weather conditions by loading the pretrained model above.

### Weighting-based Settings

- Equal: All task weights are 1. `--weight equal`
- Uncertainty: [Uncertainty](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf). `--weight uncert`
- Dynamic Weighting Average: [DWA](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf). `--weight dwa`
- Auto Lambda: [Auto-Lambda](https://openreview.net/pdf?id=KKeCMim5VN). `--weight autol`
- Min-max: Our proposed method. `--weight minmax`

### Parameter Settings

- The total epoch is set to 1 (i.e., `--epoch 1`) to attack the pretrained model.
- The Normalization method is implemented by using `--attack_weight normalize` and `--weight equal`. Otherwise, use `--attack_weight none`.
- The Min-max optimization is fully implemented by using `--attack_weight minmax`.

## Acknowledgements
We would sincerely thank Dr. Shikun Liu and his group for the Multi-task Attention Network (MTAN) design. The following links show their [MTAN Project Page](https://github.com/lorenmt/mtan) and [Auto-lambda Project Page](https://github.com/lorenmt/auto-lambda).

## Citations
If you find this code/work to be useful in your own research, please consider citing the following.
- Conference paper:
```bash
@inproceedings{guo2025robust,
  title={Robust Multi-task Adversarial Attacks Using Min-max Optimization},
  author={Guo, Jiacheng and Li, Lei and Yang, Haochen and Geng, Baocheng and Yu, Hongkai and Qin, Minghai and Zhang, Tianyun},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}

```
- Journal article: 
```bash
TBD
```
## Contact
If you have any questions, please contact Jiacheng Guo at `j.guo58@vikes.csuohio.edu`.
