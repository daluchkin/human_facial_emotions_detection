[Portfolio](https://github.com/daluchkin/data-analyst-portfolio) |  [Projects](https://github.com/daluchkin/data-analyst-portfolio/blob/main/projects.md) | [Certificates](https://github.com/daluchkin/data-analyst-portfolio/blob/main/certificates.md) | [Contacts](https://github.com/daluchkin/data-analyst-portfolio#my_contacts)


## Human Facial Emotions Detection

This project focuses on detecting emotions from facial images using a Convolutional Neural Network (CNN). The model is trained on labeled facial expressions to classify emotions such as happiness, sadness, anger, surprise, and more. This can be applied in various domains, including human-computer interaction, sentiment analysis, and mental health monitoring.

### Objective

The goal is to develop a robust deep learning model capable of accurately classifying facial emotions. The models evaluated in this project include:

- Convolutional Neural Networks (CNN)
- Data Augmentation for better generalization

### Project Workflow

1. **Data Collection & Preprocessing**
   - Downloading and loading the dataset
   - Data augmentation techniques for better generalization
2. **Exploratory Data Analysis (EDA)**
   - Visualizing dataset distribution
   - Identifying class imbalances
3. **Model Training & Evaluation**
   - Splitting the dataset into training, validation, and test sets
   - Training CNN from scratch and evaluating performance
   - Performance evaluation using metrics:
     - Accuracy
     - F1-Score
     - ROC-AUC Score
4. **Model Selection & Optimization**
   - Selecting the best model based on evaluation metrics
   - Optimizing model performance using hyperparameter tuning

### Notebook/Code

- [`human-emotions-detector.ipynb`](./human-emotions-detector.ipynb)

### Runtime

Runtime of the model training per iteration:

+ **Kaggle GPU P100/Google Colab GPU:** ~30 min.
+ **MacBook Pro 2018 Intel CPU:** ~10 hours.

### Technologies Used

- Python (Jupyter Notebook)
- TensorFlow & Keras (Deep Learning Model Training)
- Matplotlib & Seaborn (Data Visualization)
- Scikit-Learn (Evaluation Metrics)

### Results & Findings

The project evaluates multiple deep learning architectures to identify the best-performing model. The final trained CNN achieves high accuracy in emotion classification, demonstrating its effectiveness in real-world scenarios.

### How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/daluchkin/human_emotions_detector.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open and run the Jupyter Notebook:

```bash
jupyter notebook
```

4. Merge and unzip model weights:

```bash
cd ./01_models/v1/
cat model.v1.weights.h5_part_* > model.v1.weights.h5.zip
unzip model.v1.weights.h5.zip
```

### Acknowledgements

The dataset used for training is sourced from publicly available facial expression datasets such as [FER2013](https://www.kaggle.com/msambare/fer2013).

### Future Improvements

- Training on a larger, more diverse dataset for improved accuracy
- Deploying the model as a web, mobile application or chat bot.

This project is part of my deep learning portfolio, showcasing my work in computer vision and emotion recognition.

Feel free to explore, contribute, or provide feedback!


[Portfolio](https://github.com/daluchkin/data-analyst-portfolio) |  [Projects](https://github.com/daluchkin/data-analyst-portfolio/blob/main/projects.md) | [Certificates](https://github.com/daluchkin/data-analyst-portfolio/blob/main/certificates.md) | [Contacts](https://github.com/daluchkin/data-analyst-portfolio#my_contacts)
