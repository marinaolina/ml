# Eleving Task

## Overview
This project focuses on identifying the most important factors influencing the target variable using **PySpark** on a Databricks Community ML cluster.

### Achieved Results
- **Best Accuracy:** 0.63
- **Top Identified Factors:**
    1. `Industry_indexed`
    2. `Job_Role_indexed`
    3. `Years_of_Experience`
    4. `Region_indexed`
    5. `Hours_Worked_Per_Week`
    6. `Age`

## Methodology
1. **Data Processing:**
    - Features were preprocessed using PySpark’s MLlib tools, including feature indexing      
    - One-hot encoding, and bucketization are appropriate, but not implemented
    - Both numerical and categorical features were scaled or transformed to ensure consistency.

2. **Modeling:**
    - Gradient-Boosted Trees (GBT) classifier was used for model training.
    - The model was wrapped in a One-vs-Rest (OvR) strategy to handle the multiclass nature of the problem.
    - Cross-validation and hyperparameter tuning were supposed to be employed to optimize performance (not implemented)

3. **Feature Importance Analysis:**
    - Post-training, the top factors impacting model predictions were extracted and analyzed to derive actionable insights.

## Tools and Environment
- **PySpark:** For distributed data processing and machine learning.
- **Databricks Community Edition:** For collaborative development and execution on a cloud-based cluster.
- **MLlib:** PySpark’s machine learning library for feature transformation, model training, and evaluation.

## How to Run
1. Copy the code into your Databricks workspace.
2. Create and configure a cluster with the **Databricks Runtime 13.3 ML** (which includes preinstalled machine learning tools).
3. Attach the notebook to the cluster and execute the cells sequentially.

## Notebook
The full implementation details, including data preprocessing, model training, and results analysis, are available in the published notebook:

[Databricks Notebook](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3818707172192602/490510621157755/3764009250637812/latest.html)

