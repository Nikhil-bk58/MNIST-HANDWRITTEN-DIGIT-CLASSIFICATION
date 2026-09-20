
# MNIST Handwritten Digit Classification (TensorFlow / Keras)

End-to-end deep learning pipeline built with **TensorFlow/Keras** to classify handwritten digits from the classic **MNIST dataset**, complete with exploratory data analysis, baseline training, visualization, and a regularization experiment using **Dropout**.

---

## 📁 Repository Structure
```text
├── mnist_classification.py   # Main training, evaluation, and comparison script
└── README.md                 # Project documentation
```


## 🚀 Features & Pipeline Workflow

1. **Data Loading & Exploration**:
* Loads the 28x28 grayscale MNIST training (60,000) and test (10,000) images.
* Inspects tensor shapes and raw pixel value ranges (`0` to `255`).
* Renders a 2x5 grid of sample handwritten digits with their ground-truth labels using `matplotlib`.
* Verifies class distribution across digits `0–9` to ensure balanced representation.


2. **Data Preprocessing**:
* Normalizes pixel values from `[0, 255]` to floating-point range `[0.0, 1.0]` for numerical stability and faster gradient descent convergence.


3. **Baseline Neural Network Architecture**:
* **Input Layer**: `28 x 28` images.
* **Flatten Layer**: Converts 2D spatial representation into a 784-element vector.
* **Hidden Dense Layer**: 128 neurons with **ReLU** activation.
* **Output Layer**: 10 neurons with **Softmax** activation (multiclass probability distribution over digits 0–9).


4. **Training & Evaluation (Baseline)**:
* Optimizer: `adam`
* Loss Function: `sparse_categorical_crossentropy`
* Metrics: `accuracy`
* Split: 80% training, 20% validation (`validation_split=0.2`), trained for 10 epochs with batch size 32.
* Plots training vs. validation accuracy and loss curves.
* Evaluates test performance and visualizes predictions on the first 5 test samples.


5. **Regularization Experiment (Dropout)**:
* Introduces a `Dropout(0.2)` layer right after the first dense layer (`Flatten -> Dense(128) -> Dropout(0.2) -> Dense(10)`).
* Trains under identical hyperparameters to test generalization improvement.
* Directly compares validation accuracy and validation loss trajectories between baseline and dropout models.



---

## 🛠️ Requirements

Make sure you have the required dependencies installed:

```bash
pip install tensorflow numpy matplotlib

```

---

## 💻 Usage

Run the complete pipeline script:

```bash
python mnist_classification.py

```

---

## 📊 Sample Visualizations Generated

* Sample MNIST dataset grid (`2x5`)
* Training vs. Validation Accuracy (Baseline)
* Training vs. Validation Loss (Baseline)
* 5 Test predictions with Actual vs. Predicted comparison
* Dropout vs. Baseline Validation Accuracy overlay
* Dropout vs. Validation Loss overlay

---

## 🔬 Architecture Summary

| Layer | Type | Output Shape | Activation / Param Note |
| --- | --- | --- | --- |
| `input_layer` | Input | `(None, 28, 28)` | Raw shape |
| `flatten` | Flatten | `(None, 784)` | Vectorized |
| `dense_1` | Dense | `(None, 128)` | ReLU |
| `dropout` *(Experiment)* | Dropout | `(None, 128)` | 20% drop rate |
| `output` | Dense | `(None, 10)` | Softmax |

---

## 👤 Author

* GitHub: [@Nikhil-bk58](https://github.com/Nikhil-bk58/)
