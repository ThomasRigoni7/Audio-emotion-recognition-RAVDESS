# Audio-emotion-recognition-RAVDESS
 
## Emotion recognition from audio using python 3 (3.8), PyTorch and Librosa.
#### Mainly on the RAVDESS dataset, but with implementations for IEMOCAP, CREMA-D, CMU-MOSEI and others.

This repository is an implementation for Speech Emotion Recognition in the context of the [SPRING](https://spring-h2020.eu/) european project, with the objective of creating socially pertinent robots in gerontological healthcare.

To start using this code, first download one or more datasets, they are available at these links:

[RAVDESS](https://zenodo.org/record/1188976#.YPU9pkzOOUk)  
[IEMOCAP](https://sail.usc.edu/iemocap/) (must request access)  
[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)  
[CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/), the easiest way to download it is with the SDK [here](https://github.com/A2Zadeh/CMU-MultimodalSDK)  

The folder structure used is clearly visible in the config files under `config/`, some of the scripts used to generate these structures and the required csv files that contain the division of recordings in the train, validation and test sets are available in the files under `src/`, on the corresponding dataset file or in `genfiles.py` for RAVDESS, some pre-generated config files have been added in the `csv/` folder for ease of use.

Some config files are named using the `_SD` and `_SI` postfixes, these mean Speaker Dependent and Speaker Independent configurations, and are respectively equal to RANDOM and DIVIDED.

When you have downloaded the datasets in the right folders and generated the csv files, to start the training you must give three config files to the `src/main_train.py` file, as an example for a run involving the TCN and the IEMOCAP dataset you should type something like this:  
```
python3 ./src/main_train.py --dataset_config ./config/datasets/IEMOCAP/IEMOCAP_DIVIDED.yaml --model_config ./config/models/TCN.yaml --training_config ./config/TRAINING.yaml
```

The main_train.py file will take care of loading the datasets according to the division in the csv files specified in the config, train the specified model on these datasets and then test the performance of the last and the best model of the validation set. If not otherwise specified, the program creates a new run on W&B, to remove this feature you can add the parameters `--wandb f`.

When working on the CMU-MOSEI dataset it is necessary to change the training configuration file into `./config/TRAINING_MULTILABEL.yaml`, since this dataset is classified in a multi-label fashion. This creates slightly different metrics that the regular ones, adding F1 score.

The TCN reaches the current State-of-the-art for the RAVDESS dataset with an accuracy of 81.2% in the Speaker Dependent configuration, using both the speech and song files. Below is an image showing the most important changes applied to reach this result.

![Bar plot](/images/accuracies.png)


A rather detailed report of the work done and the results achieved with this repository is available on my Bachelor Thesis `thesis.pdf` (written in italian).

The base code for this project was taken from [here](https://github.com/VeroJulianaSchmalz/E2E-Sentence-Classification-on-Fluent-Speech-Commands) and adapted for the  SER task.
