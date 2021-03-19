# Contents

- [Contents](#contents)
    - [Unet Description](#unet-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
            - [running on Ascend](#running-on-ascend)
            - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Evaluation Performance](#evaluation-performance)
        - [How to use](#how-to-use)
            - [Inference](#inference)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [Unet Description](#contents)

Unet3D Medical model for 3D image segmentation. This implementation is as described  in the original paper [Unet3D: Learning Dense VolumetricSegmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650). The network of Unet3D is similar to the Unet, it is use 3D operation instead of 2D operation.
While the Unet is anentirely 2D architecture, the Unet3D proposed in this paper takes 3D volumesas input and processes them with corresponding 3D operation.

## [Model Architecture](#contents)

Unet3D model is created based on the previous Unet(2D), which includes an encoder part and a decoder part. The encoder part is used to analyze the whole picture and extract and analyze features, while the decoder part is to generate a segmented block image.

## [Dataset](#contents)

Dataset used: [LUNA16](https://luna16.grand-challenge.org/)

- Description: The data is to automatically detect the location of nodules from volumetric CT images. 888 CT scans from LIDC-IDRI database are provided. The complete dataset is divided into 10 subsets that should be used for the 10-fold cross-validation. All subsets are available as compressed zip files.

## [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor. If you want to try Ascend, please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you can get the resources.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Select the network and dataset to use

Refer to `src/config.py`. We support some parameter configurations for quick start.

- Run on Ascend

```python
# run training example
python train.py --data_url=/path/to/data/ --seg_url=/path/to/data/ > train.log 2>&1 &
OR
bash scripts/run_standalone_train.sh [DATASET]

# run distributed training example
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET]

# run evaluation example
python eval.py --data_url=/path/to/data/ --seg_url=/path/to/data/ --ckpt_path=/path/to/checkpoint/ > eval.log 2>&1 &
OR
bash scripts/run_standalone_eval.sh [DATASET] [CHECKPOINT]

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```path
.
└─unet3D
  ├── README.md                       // descriptions about Unet3D
  ├── scripts
  │   ├──run_disribute_train.sh       // shell script for distributed on Ascend
  │   ├──run_standalone_train.sh      // shell script for standalone on Ascend
  │   ├──run_standalone_eval.sh       // shell script for evaluation on Ascend
  ├── src
  │   ├──config.py                    // parameter configuration
  │   ├──dataset.py                   // creating dataset
  │   ├──transform.py                 // handle dataset
  │   ├──convert_nifti.py             // convert dataset
  │   ├──loss.py                      // loss
  │   ├──conv.py                      // conv components
  │   ├──utils.py                     // General components (callback function)
  │   ├──unet3d_model.py              // Unet3D model
  │   ├──unet3d_parts.py              // Unet3D part
  ├── train.py                        // training script
  ├── eval.py                         // evaluation script

### [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py

- config for Unet, ISBI dataset

  ```python
  'model': 'Unet3d',                  # model name
  'lr': 0.0001,                       # learning rate
  'epochs': 50,                       # total training epochs when run 1p
  'distribute_epochs': 200,           # total training epochs when run 8p
  'batchsize': 1,                     # training batch size
  'num_classes': 4,                   # the number of classes in the dataset
  'num_channels': 1,                  # the number of channels
  'keep_checkpoint_max': 5,           # only keep the last keep_checkpoint_max checkpoint
  'loss_scale': 256.0,                # loss scale
  'roi_size': [224, 224, 96],         # random roi size
  'overlap': 0.25,                    # overlap rate
  'min_val': -500,                    # intersity original range min
  'max_val': 1000,                    # intersity original range max
  ```

## [Training Process](#contents)

### Training

#### running on Ascend

```shell
python train.py --data_url=/path/to/data/ -seg_url=/path/to/data/ > train.log 2>&1 &
OR
bash scripts/run_standalone_train.sh [DATASET]
```

The python command above will run in the background, you can view the results through the file `train.log`.

After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

```shell
# grep "loss is " train.log
step: 1, loss is 0.7011719, fps is 0.25025035060906264
step: 2, loss is 0.69433594, fps is 56.77693756377044
step: 3, loss is 0.69189453, fps is 57.3293877244179
step: 4, loss is 0.6894531, fps is 57.840651522059716
step: 5, loss is 0.6850586, fps is 57.89903776054361
step: 6, loss is 0.6777344, fps is 58.08073627299014
...  
step: 597, loss is 0.19030762, fps is 58.28088370287449
step: 598, loss is 0.19958496, fps is 57.95493929352674
step: 599, loss is 0.18371582, fps is 58.04039977720966
step: 600, loss is 0.22070312, fps is 56.99692546024671
```

The model checkpoint will be saved in the current directory.

#### Distributed Training

```shell
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET]
```

The above shell script will run distribute training in the background. You can view the results through the file `logs/device[X]/log.log`. The loss value will be achieved as follows:

```shell
# grep "loss is" logs/device0/log.log
step: 1, loss is 0.70524895, fps is 0.15914689861221412
step: 2, loss is 0.6925452, fps is 56.43668656967454
...
step: 299, loss is 0.20551169, fps is 58.4039329983891
step: 300, loss is 0.18949677, fps is 57.63118508760329
```

## [Evaluation Process](#contents)

### Evaluation

- evaluation on ISBI dataset when running on Ascend

Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/unet3D/Unet3d-50_855.ckpt".

```shell
python eval.py --data_url=/path/to/data/ --seg_url=/path/to/data/ --ckpt_path=/path/to/checkpoint/ > eval.log 2>&1 &
OR
bash scripts/run_standalone_eval.sh [DATASET] [CHECKPOINT]
```

The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

```shell
# grep "Cross valid dice coeff is:" eval.log
============== Cross valid dice coeff is: {'dice_coeff': 0.9085704886070473}
```

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

| Parameters          | Ascend                                                    |
| ------------------- | --------------------------------------------------------- | 
| Model Version       | Unet3D                                                    |
| Resource            | Ascend 910; CPU 2.60GHz，192cores；Memory，755G            |
| uploaded Date       | 03/18/2021 (month/day/year)                               |
| MindSpore Version   | 1.1.0                                                     |
| Dataset             | LUNA16                                                    |
| Training Parameters | epoch = 50,  batch_size = 1                               | 
| Optimizer           | Adam                                                      |
| Loss Function       | SoftmaxCrossEntropyWithLogits                             |
| Speed               | 8pcs: 90ms/step                                           |
| Total time          | 8pcs: 4.81hours                                           |
| Parameters (M)      | 34                                                        |
| Scripts             | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/unet3D> | <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/unet3D> |

#### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | Unet3D                      |
| Resource            | Ascend 910                  |
| Uploaded Date       | 03/18/2021 (month/day/year) |
| MindSpore Version   | 1.1.0                       |
| Dataset             | LUNA16                      |
| batch_size          | 1                           |
| Dice                | dice =                      |
| Model for inference | 34M(.ckpt file)             |

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
