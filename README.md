# Predicting Nonprofit Success with Deep Learning

## Overview
The goal of this project is to develop a deep learning model to predict the success of nonprofit organizations in securing funding. Using historical data from various applications, we apply machine learning techniques to build a predictive model that identifies organizations with the highest likelihood of success. The analysis involves data preprocessing, feature engineering, neural network modeling, and evaluation.

## Instructions
For detailed instructions, refer to the [Instructions PDF](instructions.pdf).

## Data Preprocessing

### Dataset Overview
The dataset provided contains information about nonprofit organizations applying for funding, including categorical and numerical features. The target variable is `IS_SUCCESSFUL`, which indicates whether an organization received funding (1) or not (0).

### Data Cleaning and Transformation
1. **Dropped Irrelevant Columns**: The `EIN` and `NAME` columns were removed as they do not contribute to the predictive analysis.
2. **Categorical Feature Reduction**:
   - `APPLICATION_TYPE`: Several categories had low representation in the dataset. Categories with fewer than 500 instances were grouped into an "Other" category.
   - `CLASSIFICATION`: Similarly, classifications with fewer than 1,000 instances were grouped into "Other."
3. **Encoding Categorical Variables**: Categorical features were converted into numeric format using one-hot encoding with `pd.get_dummies()`.
4. **Feature Scaling**: Numerical features were standardized using `StandardScaler` to improve the model’s learning performance.

### Splitting Data
- Features (`X`) and target (`y`) variables were defined.
- Data was split into training (75%) and testing (25%) datasets using `train_test_split()` with a `random_state` of 78.
- Standardization was applied to the training and testing sets using `StandardScaler`.

## Deep Learning Model

### Model Architecture
The deep learning model was designed as a sequential neural network with the following architecture:
- **Input Layer**: The number of input features matches the transformed dataset.
- **Hidden Layers**:
  - **Layer 1**: 80 neurons, ReLU activation function.
  - **Layer 2**: 30 neurons, ReLU activation function.
- **Output Layer**: 1 neuron with a sigmoid activation function to output probabilities for binary classification.

### Compilation and Training
- **Loss Function**: `binary_crossentropy`
- **Optimizer**: `adam`
- **Metrics**: `accuracy`
- **Epochs**: 100

### Model Evaluation
After training the model, we evaluated its performance on the test set:
- **Loss**: 0.5583
- **Accuracy**: 72.76%

## Results and Analysis
- The model achieved **72.76% accuracy**, which is reasonable but below the desired 75% threshold.
- Further optimizations could include:
  - Adding more hidden layers and neurons.
  - Adjusting learning rates and optimizers.
  - Experimenting with dropout layers to reduce overfitting.
  - Performing hyperparameter tuning using `GridSearchCV` or `Keras Tuner`.

## Conclusion
This analysis successfully applied deep learning techniques to predict nonprofit funding success. While the initial model performed well, further optimization is necessary to improve accuracy beyond 75%. Alternative machine learning models, such as Random Forest or XGBoost, may also be considered for comparison.

The trained model was saved as `NNSkeleton.keras` for future use and deployment.

