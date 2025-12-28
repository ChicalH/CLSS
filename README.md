# Causality-inspired Learning Semantic Segmentation in Unseen Domain

This repository contains the official implementation of our paper **"Causality-inspired Learning Semantic Segmentation in Unseen Domain"**, published in Pattern Recognition. 

## Overview
Semantic segmentation in unseen domain is a critical challenge. Previous approaches rely on neural networks that employ statistical models to learn the correlation patterns within the source domain. However, the generalization of source domain correlations is limited for domain shifts. In this paper, a novel causality-inspired learning method is proposed, which explores how to learn the causal properties to effectively improve generalization in semantic segmentation. Firstly, Consistent Embedding Representation (CER) is proposed to learn the causal completeness that ensures feature causal sufficiency and avoids overfitting the source domain. CER constructs a consistent embedding representation that is not inclined to fit the correlation and updates it to a sufficient prototype representation, which contains enough latent causal information for pixel classification. Secondly, Causal Prototype Learning (CPL) is proposed to learn causal independence. CPL improves the causal factors in confounding information through causal consistent regularization and adaptive learning, which encourages the model to classify according to the causal factors, thus improving the generalization of segmentation in unknown domain. Experiments on four domain generalized scene segmentation benchmarks demonstrate the effectiveness of the proposed approach. 

## Setup Environment
- Python 3.8.5
- CUDA 11.0
- In that environment, the requirements can be installed with:
```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7 
```

## Datasets
1. Download Cityscapes from [link](https://www.cityscapes-dataset.com/downloads/).
2. Download GTA dataset from [link](https://download.visinf.tu-darmstadt.de/data/from_games/).
3. Download Synthia dataset from [link](http://synthia-dataset.net/downloads/).
4. Download BDD-100K from [link](https://bdd-data.berkeley.edu/index.html).
5. Download Mapillary from [link](https://www.mapillary.com/dataset/vistas).
6. Download ACDC from [link](https://acdc.vision.ee.ethz.ch).

**Data Preprocessing:** 
please run the following scripts to convert the label IDs to the train IDs and to generate the class index for RCS:
 ```bash
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
python tools/convert_datasets/synthia.py data/synthia/ --nproc 8
python tools/convert_datasets/mapillary.py data/mapillary/ --nproc 8
 ```

## Training
A training job can be launched using:
```bash
CUDA_VISIBLE_DEVICES=0 python run_experiments.py --exp 500
```
The generated configs will be stored in configs/generated/.


## Evaluation
To evaluate the model, run:
```bash
CUDA_VISIBLE_DEVICES=0 python -m tools.test --test-set --eval mIoU --dataset BDD100K
CUDA_VISIBLE_DEVICES=0 python -m tools.test --test-set --eval mIoU --dataset Mapillary --eval-option efficient_test=True
```

## Checkpoints
We provide checkpoints trained on GTAV. The checkpoint model and test results can be downloaded from [link](https://pan.baidu.com/s/1aczGBl389BAGVpON3e0siw?pwd=7tvq).


## Citation
If you find our work helpful, please cite our paper:
```bash
@article{HE2025113006,
title = {Causality-inspired Learning Semantic Segmentation in Unseen Domain},
journal = {Pattern Recognition},
pages = {113006},
year = {2025},
issn = {0031-3203},
author = {Pei He and Lingling Li and Licheng Jiao and Xu Liu and Fang Liu and Ronghua Shang and Yuwei Guo and Puhua Chen and Shuyuan Yang}
}
```


## Acknowledgements
The code is based on the following open-source projects. We thank their authors for making the source code publicly available.

[MMSegmentation](https://github.com/open-mmlab/mmsegmentation)  
[HRDA](https://github.com/lhoyer/HRDA)  

## License
This project is released under the MIT License.
