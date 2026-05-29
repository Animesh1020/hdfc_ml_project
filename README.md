# HDFC Loan Analytics & Prediction System

## Project Overview

The HDFC Loan Analytics & Prediction System is a Machine Learning project designed to analyze loan applicant data and build predictive models for banking decision-making.

The project focuses on three key business use cases:

1. **Loan Approval Prediction** – Predict whether a loan application will be approved or rejected.
2. **Default Risk Analysis** – Identify applicants who are likely to default on loan repayments.
3. **Loan Amount Prediction** – Estimate the loan amount that can be sanctioned based on applicant profile and financial information.

The project follows a complete Machine Learning workflow including data cleaning, exploratory data analysis (EDA), feature engineering, preprocessing, model training, evaluation, hyperparameter tuning, and model deployment preparation.

---

## Objectives

* Analyze loan applicant data and uncover business insights.
* Build classification models for loan approval prediction.
* Build risk assessment models for default prediction.
* Build regression models for loan amount estimation.
* Compare multiple machine learning algorithms.
* Perform model evaluation and optimization.
* Maintain professional Git workflows using feature branches and pull requests.

---

## Dataset Features

The dataset contains applicant demographic, financial, and credit-related information, including:

* Gender
* Marital Status
* Education
* Employment Status
* Applicant Income
* Co-applicant Income
* Loan Amount
* Loan Term
* Credit History
* CIBIL Score
* Debt-to-Income Ratio
* Existing EMIs
* Asset Value
* Property Area
* Occupation
* State
* Customer Sentiment
* Loan Status

---

## Project Structure

```text
hdfc_ml_project/
│
├── dataset/
│   ├── hdfc_loan_dataset_full_enriched.csv
│   └── hdfc_loan_dataset_cleaned.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_loan_approval_prediction.ipynb
│   ├── 03_default_risk_analysis.ipynb
│   └── 04_loan_amount_prediction.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/
│   ├── loan_approval_model.pkl
│   ├── risk_model.pkl
│   └── loan_amount_model.pkl
│
├── visuals/
│   └── Generated visualizations
│
├── reports/
│   └── Final project report
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Machine Learning Workflow

### Phase 1: Data Cleaning

* Missing value handling
* Duplicate record removal
* Data type validation
* Removal of personal identifiers
* Feature selection

### Phase 2: Exploratory Data Analysis (EDA)

* Loan approval distribution
* Income analysis
* CIBIL score analysis
* Correlation analysis
* State-wise trends
* Occupation-wise trends
* Debt-to-income analysis

### Phase 3: Data Preprocessing

* Encoding categorical variables
* Feature scaling
* Train-test split
* Feature engineering

### Phase 4: Model Development

#### Loan Approval Prediction

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

#### Default Risk Analysis

* Logistic Regression
* Random Forest
* Gradient Boosting

#### Loan Amount Prediction

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

### Phase 5: Hyperparameter Tuning

* Grid Search CV
* Cross Validation
* Model Optimization

### Phase 6: Evaluation

#### Classification Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

#### Regression Metrics

* MAE
* MSE
* RMSE
* R² Score

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebook
* Git
* GitHub

---

## Git Workflow

This project follows a feature-branch development workflow.

Branches used:

```text
main
│
├── feature/data-cleaning
├── feature/eda
├── feature/loan-approval-model
├── feature/default-risk-model
├── feature/loan-amount-model
└── feature/documentation
```

Each feature is developed independently and merged into the main branch through Pull Requests.


---

## Author

**Animesh Garg**
B.Tech Computer Science & Engineering (Cloud Computing)
SRM Institute of Science and Technology

---

## License

This project is developed for educational and academic purposes.
