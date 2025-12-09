# Fashion MNIST Classifier

**Description:**  
A Convolutional Neural Network (CNN) to classify Fashion MNIST images into 10 categories. Includes data augmentation, batch normalization, dropout, and visualization of training performance.

**Technologies:**  
Python, TensorFlow, Keras, NumPy, Matplotlib

---

## Features
- CNN architecture with multiple convolutional and dense layers
- Data augmentation for better generalization
- Visualize training/validation loss and accuracy curves
- Confusion matrix for performance evaluation
- Sample predictions displayed

---

## Folder Structure
```

fashion-mnist-classifier/
├─ src/                  # Source code
│  ├─ train_model.py     # Train the CNN
│  └─ predict.py         # Run predictions on sample images
├─ models/               # Saved trained model (optional)
├─ README.md             # This file
└─ requirements.txt      # Python dependencies

````

---

## Installation
1. Clone the repo:
```bash
git clone https://github.com/your-username/fashion-mnist-classifier.git
cd fashion-mnist-classifier
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

Train the model:

```bash
python src/train_model.py
```

Predict on new images:

```bash
python src/predict.py
```

---

## Usage

1. Run training to fit the CNN on Fashion MNIST.
2. Visualize training metrics and confusion matrix.
3. Predict the category of new fashion images using the trained model.

---

