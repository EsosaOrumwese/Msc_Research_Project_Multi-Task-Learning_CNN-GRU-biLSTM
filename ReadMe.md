# Multi-Task Learning for Driver Identification and Transport Mode Classification

This project is part of my Master's thesis for the **MSc in Data Science with Artificial Intelligence** program at the **University of Exeter**. It implements a multi-task deep learning model for simultaneous **driver identification** and **transport mode classification** using smartphone sensor data from the [SHL preview dataset](http://www.shl-dataset.org/download/#shldataset-preview). The project leverages multi-task learning (MTL) to improve performance across both tasks and optimize for real-world applications such as **usage-based insurance** and **transport analytics**.

## Project Structure

### Root Directory
The root directory contains the main notebooks used for training the models and testing different aspects of the pipeline:

- **`driver_identification_singletask.ipynb`**: Notebook for training the driver identification model using a ResNet50-GRU architecture.
- **`transport_classification_singletask.ipynb`**: Notebook for training the transport mode classification model using a BiLSTM architecture.
- **`multitask_MTL_model.ipynb`**: Notebook for training and evaluating the multi-task learning (MTL) model that combines both tasks.
- **`preprocess.ipynb`**: Notebook to preprocess the raw SHL dataset and generate the required feature maps and labels.
- **`metrics_result.ipynb`**: Notebook to evaluate and compare the performance of the single-task models and the MTL model.
- **`hyperparam_test/`**: Folder containing scripts and files used for hyperparameter tuning.

### `src/` Folder
The **`src/`** directory contains the core scripts for the project, including the data handling, model definitions, and utility functions:

- **`config.py`**: Configuration settings for training the models, including hyperparameters and file paths. [It is not updated though. Find config in training files]
- **`dataset.py`**: Functions for loading and processing the dataset.
- **`engine.py`**: Functions to train and evaluate the models.
- **`hyperparam.py`**: Functions to perform hyperparameter tuning.
- **`metrics.py`**: Functions to calculate metrics such as accuracy, precision, recall, F1-score, and ROC-AUC for the models.
- **`model_driverID.py`**: Driver identification model (ResNet50-GRU architecture).
- **`model_multitask.py`**: Multi-task learning model architecture that combines transport mode classification and driver identification.
- **`model_simpleCNN.py`**: Simple CNN model for testing.
- **`model_transportMode.py`**: Transport mode classification model (BiLSTM architecture).
- **`plot.py`**: Functions to generate learning curves, precision-recall curves, and other visualizations.
- **`preprocess.py`**: Script to preprocess the raw SHL dataset into feature maps and labels for training.
- **`utils.py`**: Utility functions for various tasks such as data augmentation, saving and loading models, etc.

### `prep_files/` Folder
This folder contains notebooks used for testing and debugging the functions on sample datasets before applying them to the full dataset.

### `seed_traintest/` Folder
This folder contains notebooks for training the models using different random seeds for reproducibility. These include:

- **`driver_identification_seeds.ipynb`**: Training the driver identification model with different seeds.
- **`transport_classification_seeds.ipynb`**: Training the transport mode classification model with different seeds.
- **`multitask_MTL_seeds.ipynb`**: Training the MTL model with different seeds.

### `model_checkpoint/` Folder
This folder stores model checkpoints during training.

### `data/` Folder
This folder is intended to hold the raw SHL dataset. **The dataset is not included in this repository due to its size.** It can be downloaded from the [official SHL dataset website](http://www.shl-dataset.org/download/#shldataset-preview). The preprocessing script in `preprocess.ipynb` will generate the required LSTM features, labels, and feature maps for the models.

## Getting Started

### Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- Torchvision
- Pandas

Install the required packages using:
```bash
pip install -r requirements.txt
```

### Data Preparation

1. Download the SHL dataset and place it in the `data/` folder.
2. Run the `preprocess.ipynb` notebook to generate the preprocessed data, including LSTM features, feature maps, and labels.

### Training the Models

- **Single-task models**:
  - To train the driver identification model, run the `driver_identification_singletask.ipynb`.
  - To train the transport mode classification model, run the `transport_classification_singletask.ipynb`.

- **Multi-task model**:
  - To train the multi-task learning model, run the `multitask_MTL_model.ipynb`.

- **Seed experiments**:
  - To evaluate the robustness of the models with different seeds, run the respective notebooks in the `seed_traintest/` folder.

### Evaluation and Metrics

Evaluation metrics include **accuracy**, **precision**, **recall**, and **F1-score**, calculated across multiple classes using the scripts in `metrics.py`. Performance is visualized using learning curves, precision-recall curves, and F1-score-recall curves, which are plotted using the `plot.py` script.

Class-wise performance is evaluated to understand how well the models generalize across different transport modes and drivers.

### Ensemble Comparisons

The performance of the multi-task learning model is compared against single-task models using an ensemble method. The ensemble model combines predictions from the transport mode classification and driver identification models to form composite labels, which are compared to the multi-task model's predictions.

## Conclusion

This project is part of a Master's thesis that successfully demonstrates the use of **multi-task learning** for **driver identification** and **transport mode classification**. The multi-task model shows competitive performance compared to single-task models, offering a scalable solution for **usage-based insurance** and other real-world applications.

## Future Work

Further exploration could involve testing additional sensor data, improving the augmentation techniques for the driver identification task, and extending the model to more complex multi-task scenarios.

---

