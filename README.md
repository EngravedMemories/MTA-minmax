# Multi-task Adversarial Attacks for Autonomous Driving in Different Weather Conditions

This repository contains the source code of Multi-Task Learning with Min-max Optimization and baselines from the following papers:
1) [Robust Multi-task Adversarial Attacks Using Min-max Optimization](https://ieeexplore.ieee.org/abstract/document/10888541)) (Conference Version, In proceedings of ICASSP 2025);
2) Multi-task Adversarial Attacks for Autonomous Driving in Different Weather Conditions (Extended Journal Version, Accepted by Applied Intelligence, 2026).

All models were written in `PyTorch`. 

## Datasets
We implemented all weighting baselines presented in the paper for multiple computer vision tasks: Dense Prediction Tasks (for Cityscapes, Rainy Cityscapes, Foggy Cityscapes, and NYUv2).

- `Cityscapes`, `Foggy Cityscapes`, and `Rainy Cityscapes` [3 Tasks]: 13 Class Semantic Segmentation + 4 Class Part Segmentation + Disparity Estimation. [512 x 256] Resolution.
- `NYUv2` [3 Tasks]: 13 Class Segmentation + Depth Estimation + Surface Normal Prediction. [288 x 384] Resolution.

The `Cityscapes` and `Foggy Cityscapes` datasets with ground truths could be downloaded from the official website [here](https://www.cityscapes-dataset.com/downloads/).

The `Rainy Cityscapes` dataset is generated following the instructions of [this paper](https://ieeexplore.ieee.org/abstract/document/10574400). Please refer to `Readme.md` --> `DATA` --> `1. Download the datasets`:

- Download the [Rainy Mask](https://github.com/tsingqguo/efficientderain)(rainmix/Streaks_Garg06.zip);
- Set your paths for rainy mask and Cityscape dataset in the code dataset/generate_rainy_cityscape.py, then to generate the `Rain Cityscapes` dataset.

All the images and labels are resized to [512 x 256] resolution.

Moreover, Please download the pre-processed `NYUv2` dataset [here](https://www.dropbox.com/scl/fo/p7n54hqfpfyc6fe6n62qk/AKVb28ZmgDiGdRMNkX5WJvo?rlkey=hcf31bdrezqjih36oi8usjait&e=1&dl=0) which is evaluated in the papers. (Moreover, if you are more interested, the raw 13-class NYUv2 dataset can be downloaded [here](https://github.com/ankurhanda/nyuv2-meta-data) with segmentation labels defined [here](https://github.com/ankurhanda/SceneNetv1.0/). )
All the images and labels are resized to [288 x 384] resolution.

## Experiments

### Weight Pruning for Model Compression
The folder `prune_apgda` provides the code of our proposed network using weight pruning strategy to compress the model in 40x and 60x along with all the baselines on `NYUv2` dataset presented in paper 1. The basic network structure is established based on [MTAN](https://github.com/lorenmt/mtan). 
We propose a novel weight pruning method to compress the model, and a Min-Max optimization method including APGDA algorithm to further inprove the model performance.

### Dynamic Sparse Training and More Comparable Baselines
The root folder provides the code of our proposed network using dynamic sparse training together with weight pruning for comparison in 60x and 100x along with all the baselines on `NYUv2` and `CIFAR100` datasets presented in paper 2. The basic network structure is established based on [Auto-lambda](https://github.com/lorenmt/auto-lambda).

**Weighting-based settings:**
- Equal: All task weights are 1. `--weight equal`
- Uncertainty: [Uncertainty](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf). `--weight uncert`
- Dynamic Weighting Average: [DWA](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf). `--weight dwa`
- Min-max: Our proposed method. `--weight minmax`


### Parameter Settings

- The model compression project is located in `prune_apgda` folder. Please use `python trainer_apgda.py` to run the project.

  Please first use `--stage pretrain` to save the dense model. After using `--stage rew` to implement reweighting, finally use `--stage retrain` to retrain the compressed model.
  
  The pruning rate 40x is equivalent to `--prune-ratios 0.975`. The pruning rate 60x is equivalent to `--prune-ratios 0.983`.
  
  Min-max hyperparameter settings: beta = 50, gamma = 5 (pre-settled).

- The dynamic sparse training project is located in the root folder. Please use `python trainer_nyuv2.py` or `python trainer_cifar.py` to run the corresponding project.
  
  Please first use `--stage pretrain` to save the dense model. After using `--stage rew` to implement reweighting, finally use `--stage retrain` to retrain the compressed model.
  
  Directly use `--stage retrain` to implement Dynamic Sparse Training (without loading pretrained model).
  
  The pruning rate 60x is equivalent to `--prune-ratios 0.983`. In this case, the `layer_prune_ratios` and `layer_grow_ratios` should be set to 0.0051.
  
  The pruning rate 100x is equivalent to `--prune-ratios 0.99`. In this case, the `layer_prune_ratios` and `layer_grow_ratios` should be set to 0.003.
  
  Min-max hyperparameter settings: beta = 10, gamma = 5 (pre-settled).

## Acknowledgements
We would sincerely thank Dr. Shikun Liu and his group for the Multi-task Attention Network (MTAN) design. The following links show their [MTAN Project Page](https://github.com/lorenmt/mtan) and [Auto-lambda Project Page](https://github.com/lorenmt/auto-lambda).

## Citations
If you find this code/work to be useful in your own research, please consider citing the following.
- Conference paper:
```bash
@inproceedings{guo2024min,
  title={A Min-Max Optimization Framework for Multi-task Deep Neural Network Compression},
  author={Guo, Jiacheng and Sun, Huiming and Qin, Minghai and Yu, Hongkai and Zhang, Tianyun},
  booktitle={2024 IEEE International Symposium on Circuits and Systems (ISCAS)},
  pages={1--5},
  year={2024},
  organization={IEEE}
}
```
- Journal article: 
```bash
@article{guo2025min,
  title = {A Min–max Optimization Framework for Sparse Multi-task Deep Neural Network},
  author = {Guo, Jiacheng and Li, Lei and Sun, Huiming and Qin, Minghai and Yu, Hongkai and Zhang, Tianyun},
  journal = {Neurocomputing},
  volume = {650},
  pages = {130865},
  year = {2025},
}
```
## Contact
If you have any questions, please contact Jiacheng Guo at `j.guo58@vikes.csuohio.edu`.
