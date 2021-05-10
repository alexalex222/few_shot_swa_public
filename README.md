# Meta-free representation learning

Requirements
-------------

1. Python >= 3.8
2. Numpy >= 1.19
3. Scikit-learn >= 0.23.2
4. [PyTorch](https://pytorch.org/) >= 1.7.1
5. [TorchVision](https://pytorch.org/) >= 0.8.2
6. [Tensorboard](https://www.tensorflow.org/tensorboard) >= 2.2.1

Few-shot Regression
-----------

**Sine Waves** The `sin_data_few_shot.pt` is in `data/regression`.
```
python nn_multitask_swa.py
```
By default, we use 10 support samples in meta-test.
```
python eval_nn_regression.py
```

**QMUL Head Pose Estimation.** Download [QMUL](http://www.eecs.qmul.ac.uk/~sgg/QMUL_FaceDataset/QMULFaceDataset.zip) to the folder `dataset`. 

```
python train_qmul_regression.py
```
By default, we use 10 support samples in meta-test.
```
python test_qmul_regression.py
```

Few-shot Classification
----------- 
Download and unzip [miniImageNet](https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0&preview=miniImageNet.tar.gz), [tieredImageNet](https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0&preview=tieredImageNet.tar.gz), [CIFAR-FS](https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0&preview=CIFAR-FS.tar.gz) and [FC100](https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0&preview=FC100.tar.gz) to `data` folder.

To learn the representation from training data:
```
python pretrain_swa_novel.py --dataset miniImageNet --model resnet12 --epochs 200 --swa_start 100 --num_workers 4 --swa
```

Meta-test:
```
python eval_fewshot_swa.py --dataset miniImageNet --model resnet12 --model_path "./pretrained_models/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_swa_trial_1/resnet12_db_swa_last.pth" --n_shots 5 --swa
```
