# Robust Training and Interpretability of Automated Diagnose Model against Diabetic Retinopathy
Diabetic Retinopathy is a very common eye disease in people having diabetes. This disease can lead to blindness if not taken care of in early stages. Machine learning helps in automated diagnose but its lack of interpretability prevents people from fully trusing it. This project aims at obtaining a more robust and more interpretable model through adversarial training.  


## **Prerequisites**

1. [Python 3.6](https://www.python.org/downloads)

2. [Pytorch](https://pytorch.org)

3. [Sacred](https://sacred.readthedocs.io/en/stable/)

4. [opencv-python](https://pypi.org/project/opencv-python/)

## **Dataset**

[Kaggle](https://www.kaggle.com) provides a very hefty and diverse dataset that contains round about 30,000 images. You can download it from [here](https://www.kaggle.com/c/diabetic-retinopathy-detection/data).


## **Project Structure**

- [Report_jiahuaWU_final.pdf](https://github.com/JiahuaWU/fundus-imaging/blob/master/Report_jiahuaWU_final.pdf):
Report summarizing research methods and results.

- [resize_data.py](https://github.com/JiahuaWU/fundus-imaging/blob/master/fundus_experiments/data/resize_image.py): 
Contains code for image preprocessing using [Graham's algorithm](https://storage.googleapis.com/kaggle-forum-message-attachments/88655/2795/competitionreport.pdf)

- [training_experiment.py](https://github.com/JiahuaWU/fundus-imaging/blob/master/fundus_experiments/scripts/training_experiment.py): 
Contains training script using sacred libarary for automatic metrics logging and parameters recording.

- [adversarial.py](https://github.com/JiahuaWU/fundus-imaging/blob/master/zeiss_umbrella/zeiss_umbrella/fundus/adversarial.py) 
Pytorch implementation of adversarial examples generation using [FGSM](http://arxiv.org/abs/1511.04508), [PGD-attack](http://arxiv.org/abs/1706.06083) and [Boundary attack](http://arxiv.org/abs/1712.04248)

- [data.py](https://github.com/JiahuaWU/fundus-imaging/blob/master/zeiss_umbrella/zeiss_umbrella/fundus/data.py):
Codes for data loading and dataset balancing using pytorch. 

- [train.py](https://github.com/JiahuaWU/fundus-imaging/blob/master/zeiss_umbrella/zeiss_umbrella/fundus/train.py):
Codes for model training scheme.

## **Author**

[Jiahua Wu](https://www.linkedin.com/in/gauvain-wu-jiahua-吴家桦-5835b4135/)
