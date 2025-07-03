# BUILD-A-MACHINE-LEARNING-MODEL-E.G.CLASSIFICATION-TO-PREDICT-OUTCOMES-BASED-ON-A-DATASET

COMPANY : CODTECH IT SOLUTIONS

NAME : SWAPNIL SOMNATH JAGDALE

INTERN ID : CT04DF1586

DURATION : 4 WEEKS

MENTOR : NEELA SANTOSH

# DESCRIPTION 

# Dataset Description
The dataset includes the following features:

order_id and user_id: Unique identifiers (non-predictive).

order_time and delivery_time: Timestamps to calculate delivery duration.

location: City where the order was placed (e.g., Delhi, Mumbai).

items_ordered: A comma-separated list of items like Milk, Eggs, Fruits, etc.

total_amount: The total monetary value of the order.

payment_method: Mode of payment (Cash, Card, UPI, Wallet).

delivery_duration_mins: Difference between order and delivery time.

delivered_on_time: Target label, binary (1 = Yes, 0 = No).

The project follows a standard machine learning pipeline built in a Jupyter Notebook using Python and scikit-learn:

Data Loading: The synthetic CSV is loaded using pandas and previewed.

Feature Engineering: Irrelevant columns like IDs and raw timestamps are dropped. Categorical features are prepared for encoding.

Preprocessing:

# Numerical features are scaled using StandardScaler.

Categorical features are one-hot encoded via OneHotEncoder.

Pipelines are created using ColumnTransformer and Pipeline for clean, reusable code.

Feature Selection: SelectKBest with ANOVA F-test is used to retain the top 10 most informative features.

Model Training: A RandomForestClassifier is trained on the transformed dataset.

Evaluation: The model is tested using a hold-out test set (20%) and evaluated using accuracy score and a full classification report.

# Results
The trained model achieved 100% accuracy on the test dataset, correctly classifying all 400 test samples. This perfect score is expected due to the clean, noise-free structure of the synthetic data and the clear rule (delivery within 45 minutes = on time). However, in real-world scenarios, this level of accuracy would not be realistic due to external variables like traffic, weather, inventory issues, etc.

# Conclusion
This project demonstrates how machine learning can be applied to optimize and predict outcomes in real-time logistics systems like Blinkit. By analyzing historical order attributes, the system can forecast whether an order will be delivered on time—enabling proactive intervention. The project also showcases best practices in feature preprocessing, pipeline design, and model evaluation, making it easily extensible to real-world datasets.

Future enhancements could include time series analysis, live API integration, or modeling using real operational data for production deployment.


## OUTPUT

Accuracy: 1.0

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       234
           1       1.00      1.00      1.00       166

    accuracy                           1.00       400
   macro avg       1.00      1.00      1.00       400
weighted avg       1.00      1.00      1.00       400








