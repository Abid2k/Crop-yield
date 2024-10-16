Crop Yield Prediction with MLP-GRU Model
This project aims to predict crop yield based on various agricultural features using a hybrid MLP-GRU model. The dataset used includes crop information, yield, fertilizer, rainfall, and pesticide usage data across various states in India.

Table of Contents
Project Overview
Installation
Data Overview
Exploratory Data Analysis (EDA)
Model Architecture
Training
Evaluation Metrics
Results
Dependencies
License
Project Overview
The project combines data preprocessing, feature engineering, and a hybrid MLP-GRU model to predict crop yield. The aim is to assess the impact of factors like rainfall, pesticide, fertilizer, and crop type on crop yield using deep learning techniques.

Installation
Clone the repository:

bash
Copy code
git clone <repository_url>
Install the required libraries:

bash
Copy code
pip install -r requirements.txt
Data Overview
The dataset is structured with crop attributes, meteorological data, and yearly yield values for various states in India.

Key Columns:
Crop: Type of crop cultivated.
Crop Year: Year of cultivation.
Season: Season of cultivation (e.g., Kharif, Rabi).
State: Indian state where the crop was cultivated.
Yield: Yield per unit area (target variable).
Data Loading
Data is loaded using pandas and summarized to check for missing values, data types, and basic statistics:

python
Copy code
df = pd.read_csv('Dataset/crop_yield.csv')
df.info()
df.describe()
Exploratory Data Analysis (EDA)
The EDA covers:

Count of records by crop, season, and state.
Summary statistics and distribution of key variables.
Correlation analysis of numerical variables.
Outlier detection using the IQR method.
Sample plots include:

Bar plots for crop and season counts.
Box plots for identifying outliers in Area, Production, and Yield.
Scatter plots to assess the impact of rainfall, fertilizer, and pesticide on yield.
Model Architecture
This model is a hybrid of MLP (Multilayer Perceptron) and GRU (Gated Recurrent Unit) layers:

MLP Input: Processes the features as dense layers with L2 regularization.
GRU Input: Handles temporal dependencies in a sequence format.
Concatenation Layer: Combines MLP and GRU outputs.
Output Layer: Dense layer to predict crop yield.
python
Copy code
# Define the MLP-GRU model
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

gru_input = Input(shape=(X_train.shape[1], 1))
gru_output = GRU(64)(gru_input)

mlp_input = Input(shape=(X_train.shape[1],))
mlp_output = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(mlp_input)

combined = Concatenate()([gru_output, mlp_output])
final_output = Dense(1)(combined)

model = Model(inputs=[gru_input, mlp_input], outputs=final_output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
Training
The model is trained with callbacks for early stopping, model checkpointing, and learning rate reduction.

Training Hyperparameters:
Learning Rate: 0.001
Epochs: 50
Batch Size: 32
Evaluation Metrics
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R-squared (R²)
Sample code:

python
Copy code
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
Results
The model provides high accuracy with R² values close to 0.99. Sample results and plots include:

Actual vs Predicted Values plot for random samples.
Model Loss and MAE during training and validation.
Dependencies
Python 3.8+
Pandas, Numpy, Scikit-Learn
TensorFlow, Keras
Matplotlib, Seaborn
License
This project is licensed under the MIT License.
