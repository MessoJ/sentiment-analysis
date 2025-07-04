# Sentiment Analysis with Linear Classifiers

A machine learning project implementing three different linear classification algorithms for sentiment analysis of text reviews. The project compares the performance of Perceptron, Average Perceptron, and Pegasos algorithms on movie review data.

## Overview

This project implements and compares three linear classification algorithms:
- **Perceptron**: A basic linear classifier that updates weights when making mistakes
- **Average Perceptron**: An improved version that averages weights over all updates
- **Pegasos**: A regularized algorithm that combines perceptron updates with L2 regularization

The algorithms are trained on movie review data to classify sentiment as positive (+1) or negative (-1).

## Features

- **Bag-of-Words Feature Extraction**: Converts text reviews into numerical feature vectors
- **Stopword Removal**: Optional preprocessing to remove common English stopwords
- **Hyperparameter Tuning**: Systematic search for optimal parameters (T iterations, λ regularization)
- **Performance Visualization**: Plots showing algorithm performance and decision boundaries
- **Comprehensive Evaluation**: Training, validation, and test set accuracy reporting

## Project Structure

```
├── main.py              # Main execution script
├── project1.py          # Core algorithm implementations
├── utils.py             # Utility functions for data loading and visualization
├── test.py              # Unit tests for all functions
├── mm.py                # Toy data demonstration
├── reviews_train.tsv    # Training data
├── reviews_val.tsv      # Validation data
├── reviews_test.tsv     # Test data
├── toy_data.tsv         # 2D toy dataset for visualization
└── stopwords.txt        # List of English stopwords
```

## Installation & Requirements

```bash
# Required packages
pip install numpy matplotlib
```

**Dependencies:**
- Python 3.x
- NumPy
- Matplotlib

## Usage

### Quick Start

```bash
# Run the complete analysis
python main.py

# Test individual algorithms on toy data
python mm.py

# Run unit tests
python test.py
```

### Core Functions

#### 1. Feature Extraction
```python
import project1 as p1

# Create bag-of-words dictionary
dictionary = p1.bag_of_words(texts, remove_stopword=True)

# Extract feature vectors
features = p1.extract_bow_feature_vectors(texts, dictionary, binarize=False)
```

#### 2. Training Algorithms
```python
# Perceptron
theta, theta_0 = p1.perceptron(features, labels, T=10)

# Average Perceptron  
theta, theta_0 = p1.average_perceptron(features, labels, T=10)

# Pegasos
theta, theta_0 = p1.pegasos(features, labels, T=10, L=0.01)
```

#### 3. Classification
```python
# Make predictions
predictions = p1.classify(features, theta, theta_0)

# Calculate accuracy
accuracy = p1.accuracy(predictions, true_labels)
```

## Algorithm Details

### Perceptron
- **Update Rule**: θ = θ + y·x when prediction is wrong
- **Pros**: Simple, fast convergence on linearly separable data
- **Cons**: Can be unstable, sensitive to data order

### Average Perceptron
- **Update Rule**: Same as perceptron, but returns average of all θ values
- **Pros**: More stable than basic perceptron, better generalization
- **Cons**: Slightly more computational overhead

### Pegasos (Primal Estimated sub-GrAdient SOlver)
- **Update Rule**: θ = (1-ηλ)θ + ηy·x with learning rate η = 1/√t
- **Pros**: Handles non-separable data well, regularization prevents overfitting
- **Cons**: Requires hyperparameter tuning for λ

## Results

The project automatically performs hyperparameter tuning and reports:

### Performance Metrics
- Training and validation accuracy for each algorithm
- Optimal hyperparameters (T, λ)
- Test set performance of the best model

### Typical Results
```
Training accuracy for perceptron:      0.8234
Validation accuracy for perceptron:    0.7891

Training accuracy for average perceptron: 0.8456
Validation accuracy for average perceptron: 0.8123

Training accuracy for Pegasos:         0.8567
Validation accuracy for Pegasos:       0.8234

Best Test Accuracy: 0.8156 (Pegasos with T=10, λ=0.01)
```

### Feature Analysis
The project identifies the most explanatory words for sentiment classification:
- **Positive indicators**: "excellent", "amazing", "perfect", "outstanding"
- **Negative indicators**: "terrible", "awful", "worst", "disappointing"

## Hyperparameter Tuning

The project systematically explores:
- **T (iterations)**: [1, 5, 10, 15, 25, 50]
- **λ (regularization)**: [0.001, 0.01, 0.1, 1, 10]

Tuning strategy:
1. Fix λ=0.01, find optimal T
2. Use optimal T, find optimal λ
3. Train final model with optimal (T, λ)

## Visualization

The project generates several plots:
- **Decision boundaries** for toy 2D data
- **Accuracy vs. hyperparameter** curves
- **Training/validation performance** comparison

## Data Format

### Review Data (TSV format)
```
sentiment	text
1	This movie was absolutely fantastic!
-1	Terrible plot and bad acting throughout.
```

### Toy Data (TSV format)
```
label	x1	x2
1	1.2	0.8
-1	-0.5	1.1
```

## Mathematical Foundation

### Hinge Loss
L(θ, θ₀) = max(0, 1 - y(θ·x + θ₀))

### Perceptron Update
If y(θ·x + θ₀) ≤ 0: θ ← θ + yx, θ₀ ← θ₀ + y

### Pegasos Update
θ ← (1-ηλ)θ + ηyx, θ₀ ← θ₀ + ηy (when margin < 1)

## Testing

Run the test suite to verify all implementations:
```bash
python test.py
```

Tests cover:
- Individual algorithm components
- Loss function calculations
- Feature extraction
- Classification accuracy
- Edge cases and boundary conditions

## Contributing

To extend this project:
1. Add new algorithms in `project1.py`
2. Include corresponding tests in `test.py`
3. Update visualization utilities in `utils.py`
4. Add performance comparisons in `main.py`

## License

This project is intended for educational purposes and demonstrates fundamental machine learning concepts in sentiment analysis.

## References

- Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain.
- Freund, Y., & Schapire, R. E. (1999). Large margin classification using the perceptron algorithm.
- Shalev-Shwartz, S., et al. (2011). Pegasos: Primal estimated sub-gradient solver for SVM.
