# AWL_loss
This repository contains the framework for training speaker verification models described in the paper 'AWLloss: Speaker Verifcation Based on the
Quality and Diffculty of Speech'.
## Train Examples
The dataset path, batch size, embedding size, loss function, data augmentation, network structure, training schedule and optimizationer configured in `configs/*.yaml` file.<br>
Half-ResNet with AWLloss:
```
 python trainSpeakerNet.py --cfg configs/ResNet_emb512.yml --gpu 0,1
```
## Eval Examples
Loading weight files with `--resume`.<br>
In the `configs/*.yaml`, the `DATASET` section in `TEST`, `['O'], ['E'], ['H'], ['sitw']` represent the evaluation sets VoxCeleb-O, VoxCeleb-E, VoxCeleb-H, SITW respectively
```
 python trainSpeakerNet.py --cfg configs/HalfResNet_emb256.yml --resume weights/vox2/weights-HalfResNet-awlloss.pt --gpu 0,1
```

## Citation
Please cite [1] if you make use of the code. <br>
[1] AWLloss: Speaker Verifcation Based on the Quality and Diffculty of Speech
```
@ARTICLE{10246987,
  author={Liu, Qian and Zhang, Xia and Liang, Xinyan and Qian, Yuhua and Yao, Shanshan},
  journal={IEEE Signal Processing Letters}, 
  title={AWLloss: Speaker Verification Based on the Quality and Difficulty of Speech}, 
  year={2023},
  volume={30},
  number={},
  pages={1337-1341},
  doi={10.1109/LSP.2023.3314371}}
```
