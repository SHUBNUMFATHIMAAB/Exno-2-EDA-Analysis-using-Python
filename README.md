# Exno-2-EDA-Analysis-using-Python

# AIM:

  To perform Exploratory Data Analysis on the given data set.
  
# EXPLANATION:

  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.

# ALGORITHM:

STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

# CODING AND OUTPUT
#Step 1: Import
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load the Dataset
data = pd.read_csv(r"C:\Users\acer\Downloads\bank_marketing.csv")

# Drop index column if present
if 'Unnamed: 0' in data.columns:
    data.drop(columns=['Unnamed: 0'], inplace=True)

# Step 3: Data Cleansing - Replace Null Values
# Use mean for numeric and mode for categorical values
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = data[column].fillna(data[column].mode()[0])
    else:
        data[column] = data[column].fillna(data[column].mean())

print("Missing values handled.")

# Step 4: Boxplot to Analyze Outliers (Balance)
sns.boxplot(x=data['balance'])
plt.title("Boxplot - Balance")
plt.xlabel("Balance")
plt.show()

# Step 5: Remove Outliers Using IQR Method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Apply IQR method on balance
data = remove_outliers_iqr(data, 'balance')
print("Outliers removed using IQR method.")

# Step 6: Countplot for Categorical Data (Job)
sns.countplot(x='job', data=data)
plt.title("Countplot - Job Distribution")
plt.xticks(rotation=45)
plt.show()

# Step 7: Displot for Univariate Distribution (Age)
sns.displot(data['age'], kde=True)
plt.title("Displot - Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Step 8: Cross Tabulation (Marital vs Deposit)
crosstab_result = pd.crosstab(data['marital'], data['deposit'])
print("\nCross Tabulation Result:")
print(crosstab_result)

# Step 9: Heatmap to Show Relationships (Correlation Between Variables)
numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

<img width="687" height="573" alt="image" src="https://github.com/user-attachments/assets/ef47faa5-eb4c-4a3f-81d8-6b708584fa30" />
<img width="769" height="626" alt="image" src="https://github.com/user-attachments/assets/871da78c-40f5-4b67-b768-3817dbca5a03" />
<img width="661" height="647" alt="image" src="https://github.com/user-attachments/assets/e0a60e22-5a50-4f28-b869-9b8ee98d91fb" />
<img width="258" height="140" alt="image" src="https://github.com/user-attachments/assets/bbf8fc3f-b9f5-43be-95f1-2d6a9dd9f0cb" />
<img width="766" height="625" alt="image" src="https://github.com/user-attachments/assets/cdbce8b9-5d75-47b8-8ec5-763a07510dbe" />

# RESULT

   Thus the data analysis has been implemented succesfully.
