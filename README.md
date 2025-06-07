# Breast Cancer Detection Using Deep Learning

This project contains a complete pipeline for preprocessing, building, training, and testing a deep learning model for breast cancer classification. It includes exploratory data analysis, image preprocessing, model design, training, evaluation, and critical reflection. Three deep learning architectures are implemented and compared: a self-built CNN, EfficientNet-B0, and Vision Transformer (ViT).

## 📊 1. Exploratory Data Analysis and Preprocessing

- Visualized sample images and pixel intensity distributions  
- Converted `.npy` arrays to image format using PIL  
- Applied data augmentation techniques  
- Split dataset into training, validation, and test sets  
- Displayed visual summaries of the processed dataset  

### 📁 Dataset

- The dataset is publicly available on Kaggle:  
  🔗 [Lymphoma Subtype Classification (FL vs CLL)](https://www.kaggle.com/datasets/simjeg/lymphoma-subtype-classification-fl-vs-cll)

## 🧠 2. Model Design, Training, and Fine-tuning

### Models Implemented:
- **Custom CNN**: A modular architecture with tunable convolutional blocks  
- **EfficientNet-B0**: A pretrained model with a custom binary classification head  
- **ViT (Vision Transformer)**: A pretrained vision transformer model with a custom classification layer  

### Training Features:
- Device setup for CPU/GPU  
- Training loop with logging and checkpointing  
- Hyperparameter tuning for each model architecture  
- Fine-tuning and early stopping implemented for training optimization  

## 📈 3. Evaluation and Comparison

- Classification reports: precision, recall, F1-score  
- ROC curve and AUC comparison  
- Confusion matrix visualization  
- Misclassified image inspection  
- Learning curves for each model  
- Summary of best hyperparameter configurations  

### 🏆 Best Model:
The **ViT (Vision Transformer)** outperformed the other models, achieving **83.63% accuracy**. Despite its lightweight configuration, it demonstrated strong generalization and learning capacity for this binary classification task.

## 💾 Pretrained Weights

Trained model weights and logs are saved in the `{model_name}` directories:
- `.pth`: PyTorch model state dictionary  
- `.pkl`: Training logs and evaluation metadata  


## 💡 4. Critical Reflection

- **Strengths**: Flexible pipeline, comparative analysis across model families, and interpretable evaluation  
- **Limitations**: Small dataset size, long training time for ViT, no k-fold cross-validation  
- **Improvements**: Future work could include ensemble learning, additional datasets, and cross-validation  
- **Conclusion**: This study demonstrates that modern transformer-based models can outperform CNNs on histopathological image classification with appropriate tuning and training.

---

## 🔧 Requirements

- Python 3.8+  
- PyTorch  
- torchvision  
- scikit-learn  
- matplotlib  
- numpy  
- Pillow (PIL)  
