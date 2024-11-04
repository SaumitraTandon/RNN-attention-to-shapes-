
# RNN with Attention to Shapes

This project provides a demonstration of a simple Recurrent Neural Network (RNN) implemented in TensorFlow. The notebook walks through the process of defining, training, and testing an RNN to model temporal sequences, with a focus on understanding the dimensional shapes of the input, hidden, and output layers. The notebook is ideal for those new to RNNs or for educational purposes related to sequence modeling.

## Project Structure

- **Model Definition**: Defines a simple RNN architecture using `SimpleRNN` layers and fully connected `Dense` layers.
- **Data Generation**: Generates synthetic data to simulate a sequence prediction task.
- **Model Prediction and Evaluation**: Makes predictions on the generated data and examines model output dimensions.

## Features

- **TensorFlow and Keras Integration**: Built using TensorFlow and Keras' high-level API for ease of model building and experimentation.
- **Configurable Parameters**: Parameters like sequence length, number of features, and hidden units can be adjusted to observe changes in model behavior.
- **Educational Focus**: Emphasis on understanding input-output shapes in RNNs and how they affect model performance.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install required dependencies:
   ```bash
   pip install tensorflow numpy pandas matplotlib
   ```

## Usage

1. **Run the Notebook**: Open the notebook in Jupyter or any compatible environment to explore the code and modify parameters.
   ```bash
   jupyter notebook RNN_(attention_to_shapes).ipynb
   ```
2. **Experiment with Parameters**: You can change the following parameters to see their effect on the model's performance:
   - `N`: Number of samples
   - `T`: Sequence length
   - `D`: Number of input features
   - `M`: Number of hidden units
   - `K`: Number of output units

3. **Model Summary**: The notebook provides a summary of the RNN model architecture, which can be useful for understanding the structure and dimensionality of each layer.

## Example Code

The following code snippet illustrates the core structure of the RNN model in the notebook:

```python
from tensorflow.keras.layers import Input, SimpleRNN, Dense
from tensorflow.keras.models import Model

# Parameters
N = 1    # Number of samples
T = 10   # Sequence length
D = 3    # Number of input features
M = 5    # Number of hidden units
K = 2    # Number of output units

# Define the model
i = Input(shape=(T, D))
x = SimpleRNN(M)(i)
x = Dense(K)(x)
model = Model(i, x)

# Model summary
model.summary()
```

## Output

After defining the model, the notebook generates predictions based on randomly initialized input data. The predictions' shapes and values are displayed to illustrate the RNN's functionality.

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit changes:
   ```bash
   git commit -m "Add feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a pull request.
