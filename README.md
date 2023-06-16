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
    - Columns where the majority of data have missing values *(greater than 50%)* were also removed. These were the columns named `batch_enrolled`, `emp_title`, `mnths_since_last record`, `mnths_since_last_major_derog`, etc.

+ Data Type
    - After removing the problematic features, we are left with a data frame with only 31 variables.
    -  We then transformed the data types of the columns to be more appropriate for model learning. Ordinal values such as `loan_status` *(the class variable)*, `grade`, `verification_status`, and others are turned into factors instead of characters.
    - The remaining missing values were handled by dropping the rows containing such instances.

+ Balancing the Class Variable
    - The original distribution of the class variable is unbalanced, wherein 80% have the value **0** and 20% are **1**.

    - <img src="assets/fig1.png" alt="" width="60%">
    - In order to remove the bias towards the negative value in training the model, the researchers tried using the Synthetic Minority Oversampling Technique (SMOTE) and the undersampling technique in balancing the class variable.

    - Asa <br>
    <img src="assets/project3.1.png" alt="" width="70%">

+ Data Augmentation
    - asa


## **Test-Train Partition**
> 