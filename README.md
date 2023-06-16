# **Credit Risk Binary Classification**

A Case study analyzing the performance of different classification algorithms against loan risk data.

We created a predictive classification model using R to predict if a customer is likely to default their loan or not, helping make informed decisions about which loan applications to approve. The target variable `loan_status` is a binary class where **0** stands for eligible and **1** being not eligible for a loan.

Additionaly, we compared the difference between Hold-out and Cross-Validation method for splitting the testing and training partitions, as well as different split percentages.

---
<br>

## **Data Preparation**

The original dataset was retrieved from [Kaggle](https://www.kaggle.com/datasets/shadabhussain/credit-risk-loan-eliginility).

+ Handling Missing Data
    - Columns irrelevant to the classifier such as `member_id`,  `zip_code`, `addr_state` etc. were dropped.
    - Columns where the majority of data have missing values *(greater than 50%)* were also removed. These were the columns named `batch_enrolled`, `emp_title`, `mnths_since_last record`, `mnths_since_last_major_derog`, etc.<br><br>
+ Data Type
    - After removing the problematic features, we are left with a data frame with only 31 variables.
    -  We then transformed the data types of the columns to be more appropriate for model learning. Ordinal values such as `loan_status` *(the class variable)*, `grade`, `verification_status`, and others are turned into factors instead of characters.
    - The remaining missing values were handled by dropping the rows containing such instances.<br><br>
+ Balancing the Class Variable
    - The original distribution of the class variable is unbalanced, wherein 80% have the value **0** and 20% are **1**. <br>
    <img src="assets/Fig1.png" alt="" width="50%">
    - In order to remove the bias towards the negative value in training the model, the researchers tried using the Synthetic Minority Oversampling Technique (SMOTE) and the undersampling technique in balancing the class variable.<br><br>
+ Feature Preparation
    - Feature selection is the method of reducing input variables by using only relevant data and getting rid of noise in data. The researchers obtained the Univariance Feature Importance Score of each variable in order to select the only relevant features. Different methods for finding feature importance was used: **Chi-square test** and **Information Gain ratio**, **Correlation/Entropy with Best First Search**, and **Black-box feature selection**<br>
    <img src="assets/Fig2.png" alt="Chi-Square Test" width="50%"> 
    - In the chi-square test statistic, we first discretized the numeric and int variables to convert them as factors with 5 levels. After this, the algorithm outputs the top five variables, which are last_week_pay, initial_list_status, term, verification_status, and dti.<br>
    <img src="assets/Fig3.png" alt="Chi-Square Test" width="50%">
    - Computing the univariate feature importance score using the **information gain ratio**, the algorithm outputs the top five variables with the highest gain which are recoveries, collection_recovery_fee, last_week_pay, int_rate, and initial_list_status. 
    - The **cfs method** stated that the variables int_rate, initial_list_status, recoveries, and last_week_pay, respectively, are the most important features.
    - In the **black-box feature selection**, the five most important features are loan_amnt, term, int_rate, emp_length, and home_ownership as least important.
    - Additionally, the researchers tried greedy search strategies to find the most important features. Both **forward-search** and **best-first search** identified last_week_pay as the most important feature. **Hill-climbing search** identified recoveries as the most important attribute, followed by last_week_pay, then initial_list_status.
    - The researchers decided to pick the top 5 features using the information gain ratio method of the balanced dataset. The top 5 features are `recoveries`, `collection_recovery_fee`, `last_week_pay`, `int_rate`, and `initial_list_status`.


## **MODELING**

> 