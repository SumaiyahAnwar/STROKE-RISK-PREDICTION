# STROKE-RISK-PREDICTION

## Problem Identification:
### Stroke has emerged as a major global cause of mortality and long-term impairment with no proven cure. Predicting stroke risk is based on the dataset features such as hypertension, heart disease, smoking status, and blood glucose level parameters. Machine and Deep Learning based models are broadly used to extract significant stroke features for prediction. This work will enhance the performance by using various algorithms such as RF, KNN, CNN, LSTM, and MLP for predicting stroke. The experimental results show the accuracy, precision, recall, and f1-score.

## Data Collection:
### Data collection is the process of selecting the data. In this study, the Stroke Risk Dataset from Kaggle is used as input for detecting stroke risk prediction. The dataset contains 5110 records and 12 features.

## Data Pre-processing:
### A dataset might be inadequate, have manual entry errors, have duplicate data, or use many names to relate to the same entity. Preprocessing data is the process of removing irrelevant information from a dataset. The removal of unwanted data by data cleansing during data pre-processing makes it possible to have a dataset that contains more important information after the pre-processing stage in the data mining process for further data manipulation.

### 1. Missing Data Removal - Missing values and Nan values are replaced by 0 during this process to remove missing data. Using the print (df.isnull().sum()) library, the null values, such as missing values, are eliminated in this procedure. Missing and duplicate values were removed and data was cleaned of any abnormalities.

### 2. Encoding Categorical Data - Categorical data are variables having a limited number of label values that encode categorical variables to numerical values.

## Proposed Method:
### A stroke disrupts the blood supply to a portion of the brain. Although the risk can be decreased, strokes can be fatal. The proposed model has a comparative analysis using various machine learning and deep learning algorithms for predicting stroke risk. This uses a Stroke Risk Dataset from Kaggle as input. After loading the dataset, the preprocessing techniques are implemented. The preprocessing steps include checking missing values by using isnull() method and encoding categorical values by replacing them with numerical values. After preprocessing, the feature selection technique is implemented by using the chi-square method. The data is split into 80% of the training set and 20% of the testing set.  Using machine learning algorithms such as the Random Forest (RF), and K-Nearest Neighbors (KNN), and deep learning algorithms such as the Convolutional Neural Network (CNN), Long Short Term Memory (LSTM), and Multilayer Perceptron (MLP), the implementation is carried out.

## Steps for implementation:
### Step 1: Load the stroke dataset containing a number of parameters.
### Step 2: Load the useful libraries and packages.
### Step 3: Data preprocessing has been performed.
### Step 4: To train the model, the data has been split into the training set and testing set.
### Step 5: The model has been constructed by applying the algorithms (RF, KNN, CNN, LSTM, MLP).

## Algorithms:
### • Random Forest (RF)
### • K-Nearest Neighbors (KNN)
### • Convolutional Neural Network (CNN)
### • Long Short Term Memory (LSTM)
### • Multi-Layer Perceptron (MLP)

## Random Forest (RF):
### A popular algorithm for classification and regression issues is the supervised machine learning technique known as random forest. It creates decision trees from various samples, relying on their majority for categorization and the average for regression.
### Step 1: The Random Forest model has been trained on the training data which ensembles a method by combining multiple decision trees to make predictions. 
### Step 2: The predictions have been made on the test data by using the trained model and its hyperparameters have been adjusted.
### Step 3: The predicted values have been compared to the actual values in the test data to assess the model's performance.

## K-Nearest Neighbors (KNN):
### The k-nearest neighbors algorithm, sometimes referred to as KNN or k-NN, is a supervised learning classifier that employs proximity to producing classifications or predictions about the grouping of a single data point.
### Step 1: The data has been scaled to ensure that all features contribute evenly to the distance calculation.
### Step 2: A suitable K value has been chosen that balances bias and variance.
### Step 3: The KNN model has been trained on the training data.
### Step 4: The predictions have been made on the test data by using the trained model. The KNN algorithm locates the K closest neighbors for every test instance in the training set and then labels each test instance with the majority class.
### Step 5: The model's effectiveness has been assessed by contrasting the projected values with the actual values from the test data.

## Convolution Neural Network (CNN):
### Convolutional Neural Network, sometimes known as CNN, is a deep learning method used for object and picture recognition. To recognize local features in an image, it employs convolutional layers, and to minimize the spatial dimensions of the input, it employs layer pooling. For categorization, the output from these layers is subsequently sent via fully connected layers.
### Step 1: The CNN model has been trained by using training data  which has typically consisted of convolutional, pooling, and fully connected layers. These layer’s dimensions are determined by the amount and complexity of the dataset as well as the problem's complexity.
### Step 2: The model has been trained on the training data by using adam optimizer and mae loss function as parameters.
### Step 3: The predictions have been made on the test data by using the trained model and its hyperparameters have been adjusted.
### Step 4: The predicted values have been compared to the actual values in the test data to assess the model's performance.

## Long Short-Term Memory (LSTM):
### LSTM, or Long Short-Term Memory, is a type of Recurrent Neural Network (RNN) that is widely used for processing speech, time series data, and other sequential data. The ability of LSTM to sustain long-term dependencies by selectively forgetting or remembering information over time is its important characteristic.
### Step 1: The LSTM model has been trained by using the training data which has typically consisted of LSTM layers, dense layers, and dropout layers. These layer’s dimensions were determined by the amount and complexity of the dataset as well as the problem's complexity.
### Step 2: The model has been trained on the training data by using adam optimizer and binary cross-entropy loss function as parameters.
### Step 3: The predictions have been made on the test data by using the trained model and its hyperparameters have been adjusted.
### Step 4: The predicted values have been compared to the actual values in the test data to assess the model's performance.

## Multilayer Perceptron (MLP):
### A sort of artificial neural network called a "Multilayer Perceptron" (MLP) that uses feedforward computing consists of numerous layers of interconnected nodes or neurons. Each neuron receives inputs from the layer below, performs an activation function, and then generates an output signal that is passed into the layer below.
### Step 1: The MLP model has been trained by using the training data which has typically consisted of input, hidden, and output layers. The complexity of this task determines the number and size of these levels.
### Step 2: The model has been trained on the training data by using adam optimizer  as a parameter.
### Step 3: The predictions have been made on the test data by using the trained model and its hyperparameters have been adjusted.
### Step 4: The predicted values have been compared to the actual values in the test data to assess the model's performance.

## Results and Discussion:
### The proposed model has been built using machine learning and deep learning algorithms. It has a comparative analysis of Random Forest, K-Nearest Neighbors, Convolutional Neural Network, Long Short Term Memory, and Multilayer Perceptron algorithms. It has been observed that the Convolutional Neural Network Algorithm performs better in terms of accuracy than the other algorithms used. The data was gathered through a series of iterations to determine distinct ranges of accuracy rates. In this study of stroke prediction, the Convolutional Neural Network has an accuracy of approximately 99%, which is higher than that of the other machine learning and deep learning algorithms. Convolutional Neural Network has a better performance.

## Conclusion:
### This study shows the efficiency of various ML and DL algorithms to find the best algorithm for stroke prediction. The results indicate that the proposed Convolutional Neural Network Algorithm can be used to classify stroke risk with an improved accuracy of 99%. In order to achieve better accuracy, more data would be required. The accuracy of stroke prediction is higher using a CNN algorithm. The test results show that CNN is more effective at improving performance metrics than others.

## Future Enhancements:
### Future research will examine the hybridization of two machine learning algorithms, two deep learning algorithms, or machine learning with deep learning algorithms that can be made for even better performance. It can also be implemented in web applications for further enhancement. Combining two or more datasets can also give better results for further work.
