
# EfficientDet-tensorflow2
A Tensorflow2.x implementation of EfficientDet as described in [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070), The project is based on official implementation of [google/automl/efficientdet](https://github.com/google/automl/tree/master/efficientdet).
## Update Log
[2021-03-03] 
* Package model's preprocess part into inference model:The key benefit to doing this is that it makes your model portable and it helps reduce the training/serving skew. 
* Add TTA(Test Time Augmentation) in inference mode. 

[2021-03-02] 
* Add Graph mode training with model.fit: this dramatically improve GPU utilization(over 95%). 

[2021-03-01] 
* Add support for: efficientdet d0-d7,huber loss,focal loss.
* Eager mode training with tf.GradientTape. 
* Add online coco evaluation callback.
* Add ssd_random_crop;mosaic.
* Support tensorboard.

## Installation
###  1. Clone project
  ``` 
  git clone https://github.com/wangermeng2021/EfficientDet-tensorflow2.git
  cd EfficientDet-tensorflow2
  ```
###   2. Install environment
* Install tesnorflow (skip this step if it's already installed,test environment:tensorflow 2.3.0)
* Install dependencies:  `pip install -r requirements.txt`

## Training:

* Pretrained EfficientDet Checkpoints(google automl's efficientdet):

|       Model    | AP<sup>test</sup>    |  AP<sub>50</sub> | AP<sub>75</sub> |AP<sub>S</sub>   |  AP<sub>M</sub>    |  AP<sub>L</sub>   |  AP<sup>val</sup> | | #params | #FLOPs |
|----------     |------ |------ |------ | -------- | ------| ------| ------ |------ |------ |  :------: |
|     EfficientDet-D0 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d0.tar.gz))    | 34.6 | 53.0 | 37.1 | 12.4 | 39.0 | 52.7 | 34.3 |  | 3.9M | 2.54B  |
|     EfficientDet-D1 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d1.tar.gz))    | 40.5 | 59.1 | 43.7 | 18.3 | 45.0 | 57.5 | 40.2 | | 6.6M | 6.10B |
|     EfficientDet-D2 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d2.tar.gz))    | 43.0 | 62.3 | 46.2 | 22.5 | 47.0 | 58.4 | 42.5 | | 8.1M | 11.0B |
|     EfficientDet-D3 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d3.tar.gz))    | 47.5 | 66.2 | 51.5 | 27.9 | 51.4 | 62.0 | 47.2 | | 12.0M | 24.9B |
|     EfficientDet-D4 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d4.tar.gz))    | 49.7 | 68.4 | 53.9 | 30.7 | 53.2 | 63.2 | 49.3 |  | 20.7M | 55.2B |
|     EfficientDet-D5 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d5.tar.gz))    | 51.5 | 70.5 | 56.1 | 33.9 | 54.7 | 64.1 | 51.2 |  | 33.7M | 130B |
|     EfficientDet-D6 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d6.tar.gz))    | 52.6 | 71.5 | 57.2 | 34.9 | 56.0 | 65.4 | 52.1 | | 51.9M  |  226B  |
|     EfficientDet-D7 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d7.tar.gz))    | 53.7 | 72.4 | 58.4 | 35.8 | 57.0 | 66.3 | 53.4 | | 51.9M  |  325B  |
|     EfficientDet-D7x ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d7x.tar.gz))    | 55.1 | 74.3 | 59.9 | 37.2 | 57.9 | 68.0 | 54.4 | | 77.0M  |  410B  |


* Download pretrain weight and place it under directory './pretrain'.

* For training on [pothole dataset](https://public.roboflow.com/object-detection/chess-full)(No need to download dataset,it's already included in project): <br>

    ```
  python train.py --model-type d0  --use-pretrain True --dataset-type voc --dataset dataset/pothole_voc --num-classes 1 --class-names dataset/pothole.names --voc-train-set dataset_1,train --voc-val-set dataset_1,val  --epochs 200 --batch-size 8 --augment ssd_random_crop 
  ```

## Tensorboard visualization:
  * Navigate to [http://0.0.0.0:6006](http://0.0.0.0:6006): you need to manually enable: "Setting"-->"Reload data" on tensorboard home page to automatically update data
## Evaluation results(GTX2080,mAP@0.5):

| model                                               | pothole |  VOC  | COCO |
|-----------------------------------------------------|---------|-------|------|
| efficientdet-d0(512)                                |  0.798  |       |      |
| Scaled-YoloV4-p5(416)                               |  0.826  |       |      |

* Evaluation on Pothole dataset: 
![pothole_d0_tensorboard_1.png](https://github.com/wangermeng2021/EfficientDet-tensorflow2/blob/main/images/results/pothole_d0_tensorboard_1.png)
![pothole_d0_coco_evaluation_1.png](https://github.com/wangermeng2021/EfficientDet-tensorflow2/blob/main/images/results/pothole_d0_coco_evaluation_1.png)
## Detection

* For detection on Pothole dataset:
  ```
  python3 detect.py --model-dir export/best_model_d0_189_0.798/1 --pic-dir images/pothole --class-names dataset/pothole.names --score-threshold 0.1
  ```
  detection result:

  ![pothole_d0_detection_1.png](https://github.com/wangermeng2021/EfficientDet-tensorflow2/blob/main/images/results/pothole_d0_detection_1.png)
  ![pothole_d0_detection_2.png](https://github.com/wangermeng2021/EfficientDet-tensorflow2/blob/main/images/results/pothole_d0_detection_2.png)
  ![pothole_d0_detection_3.png](https://github.com/wangermeng2021/EfficientDet-tensorflow2/blob/main/images/results/pothole_d0_detection_3.png)

## References
* [https://github.com/google/automl/tree/master/efficientdet](https://github.com/google/automl/tree/master/efficientdet)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/dmlc/gluon-cv](https://github.com/dmlc/gluon-cv)



