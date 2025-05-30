```python
pip uninstall imblearn 
```


```python
pip install scikit-learn==1.2.2
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
data = pd.read_csv('heart_2022_no_nans.csv')
#data.drop(columns=['State'], inplace=True)
data
#data.head(10)
```


```python
#DELETE

print(data['AgeCategory'])
```


```python
# Encode the AgeCategory. 
# We could assign a number to each age group, 
# but for now, we will apply a unique identifier for each group.


encode_AgeCategory = {
    'Age 18 to 24': 21,
    'Age 25 to 29': 27,
    'Age 30 to 34': 32,
    'Age 35 to 39': 37,
    'Age 40 to 44': 42,
    'Age 45 to 49': 47,
    'Age 50 to 54': 52,
    'Age 55 to 59': 57,
    'Age 60 to 64': 62,
    'Age 65 to 69': 67,
    'Age 70 to 74': 72,
    'Age 75 to 79': 77,
    'Age 80 or older': 80
}

data['Age_Category_Avg'] = data['AgeCategory'].map(encode_AgeCategory)
#data.to_csv('heart_2022_no_nans.csv', index=False)
    
#data_2.to_csv('modified_data.csv', index=False)
```


```python
print(data['AgeCategory'])
```


```python
print(data['Age_Category_Avg'])
```


```python
print(data['AgeCategory'].dtype)
```


```python
data.describe()
```


```python
# checking if we still have any null data (even though the author of the file says it is already cleaned)

data.isnull().sum()
```


```python
data.info()
```


```python
data.shape
```


```python
data.value_counts('HadHeartAttack')
```

## Exploratory Data Analysis

A few details from the first glance: 

The dataset contains imbalanced data, reflecting an uneven distribution across its categories. (Examples and plots will be provided further)

Features such as "HadStroke", "AgeCategory", "DifficultyWalking", and possibly "HadDiabetes" have a greater impact on predicting the higher target variable rate ("HadHeartAttack").

"Sex" and "Race" exhibit lower correlation values, indicating a weaker direct relationship with heart attack in this dataset. That means we can drop these features in future.


```python
# Encode 'Sex' column

data_check = data.copy()

data_check['Sex'] = data_check['Sex'].map({'Female': 0, 'Male': 1})

# Encode 'RaceEthnicityCategory' column
data_check['RaceEthnicityCategory'] = data_check['RaceEthnicityCategory'].map({
    'White only, Non-Hispanic': 0,
    'Black only, Non-Hispanic': 1,
    'Other race only, Non-Hispanic': 2,
    'Multiracial, Non-Hispanic': 3,
    'Hispanic': 4
})

# Encode 'HadHeartAttack' column
data_check['HadHeartAttack'] = data_check['HadHeartAttack'].map({'Yes': 1, 'No': 0})

# Create a correlation matrix
corr_matrix = data_check[['Sex', 'RaceEthnicityCategory', 'HadHeartAttack']].corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 12})
plt.title('Correlation Heatmap')
plt.show()
```


```python
sns.countplot(x="HadHeartAttack", data=data)
plt.title("Distribution of Heart Attack")
plt.xlabel("Had Heart Attack")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=["No", "Yes"])  # Rename the x-axis tick labels
plt.show()
```


```python
#MAYBE

plt.figure(figsize=(12,12))
sns.boxplot(data=data)
plt.title('Boxplots of Numerical Features')
plt.show()
```


```python
cat_data=data.select_dtypes(include='object')
num_data=data.select_dtypes(exclude='object')

#categorical features:  ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 
                        #'DiffWalking', 'Sex', 'Race', 'Diabetic', 'PhysicalActivity', 
                        #'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']
#numerical features:  ['BMI', 'PhysicalHealth', 'MentalHealth', 'AgeCategory', 'SleepTime']
    

print("categorical features: ", cat_data.columns.to_list())
print("numerical features: ", num_data.columns.to_list())


data.head()
```


```python
for c in cat_data:
    plt.rcParams['figure.figsize'] = (20, 8) 
    sns.countplot(x=c, hue='HadHeartAttack', data=data)
    plt.title(f'Heart Disease Count Grouped by {c} Status')
    plt.xlabel(c)
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()
```


```python
sns.heatmap(num_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
```


```python
data_check = data.copy()

# Encode 'HadHeartAttack' column
data_check['HadHeartAttack'] = data_check['HadHeartAttack'].map({'Yes': 1, 'No': 0})

# Select numerical and categorical columns
num_data = data.select_dtypes(exclude='object')
num_data['HadHeartAttack'] = data_check['HadHeartAttack']  # Include 'HadHeartAttack' in numerical data

# Save as data_num_cat_check
data_num_cat_check = num_data

# Build Correlation Heatmap
corr_matrix = data_num_cat_check.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Heatmap')
plt.show()
```


```python
sns.pairplot(data,height=2)
plt.show()
```


```python
disease=['HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD', 
          'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'HadDiabetes']
for d in disease:
    df_filtered = data[data[d] == 'Yes']
    
    if not df_filtered.empty:
        plt.figure(figsize=(20,10))
        sns.countplot(x='HadHeartAttack', data=df_filtered)
        plt.title(f'Heart Disease Count among Patients with {d}')
        plt.xlabel('HadHeartAttack')
        plt.ylabel('Count')
        plt.show()
```


```python
for feature in num_data:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x=feature, hue='HadHeartAttack', kde=True, element='step', stat='count')
    plt.title(f'Histogram of {feature} with Heart Attack Overlay')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.legend()
    plt.show()
```


```python
sorted_age_categories = sorted(data['AgeCategory'].unique())

sns.scatterplot(data=data, x='BMI', y='AgeCategory', hue='HadHeartAttack')

# Set the title, labels, and legend
plt.title('Scatter Plot of BMI vs Age by Heart Disease Status')
plt.xlabel('BMI')
plt.ylabel('Age')
plt.yticks(ticks=range(len(sorted_age_categories)), labels=sorted_age_categories)  # Set y-axis tick labels
plt.legend(title='Heart Attack')

# Show the plot
plt.show()
```


```python
num_data = num_data.drop(columns=['HadHeartAttack'])
num_data.hist(figsize=(16, 20), bins=40, xlabelsize=6, ylabelsize=6);
```


```python
age_heart_disease = data.groupby('AgeCategory')['HadHeartAttack'].value_counts().unstack().fillna(0)

age_heart_disease.plot(kind='bar', stacked=True)
plt.title('Number of People with Heart Disease by Age Category')
plt.xlabel('Age Category')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Heart Attack', labels=['No', 'Yes'])
plt.tight_layout() 
plt.show()
```


```python
gender_heart_attack = data.groupby('Sex')['HadHeartAttack'].value_counts().unstack().fillna(0)

gender_heart_attack.plot(kind='bar', stacked=True)
plt.title('Number of People with Heart Attack by Sex Category')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Heart Attack', labels=['Female', 'Male'])
plt.tight_layout() 
plt.show()
```

## Preprocessing

I will encoded categorical features using Label Encoder. 

I also will apply RobustScaler to minimize the skewness of outliers. 
Note: no outlier removal was performed as they were present in a significant amount, which is important for current data analysis. 

I will use oversampling with SMOTE to handle the imbalance of the majority class.


```python
from sklearn import preprocessing 
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
```


```python
import sklearn
print(sklearn.__version__)
```


```python
pip install scikit-learn==0.22.2
```


```python
print(sklearn.__version__)
```


```python
label_encoder = preprocessing.LabelEncoder() 
for c in cat_data:
    data[c]= label_encoder.fit_transform(data[c]) 
    data[c].unique()
    
data.head()
```


```python
scaler=RobustScaler()
scaled_data=scaler.fit_transform(data)
sns.boxplot(data=scaled_data)
plt.show()
```


```python
race_groups = data.groupby('RaceEthnicityCategory')['HadHeartAttack'].value_counts(normalize=True).unstack(fill_value=0)
race_groups['Ratio'] = race_groups['Yes'] / race_groups['No']

print(race_groups[['Ratio']])
```

Since there isn't any difference in the ratio of heart disease across different race/ethnicity categories, it suggests that race doesn't have any significant effect on heart attack in individuals.


```python
data.drop(columns=['RaceEthnicityCategory'], inplace=True)
data.head()
```


```python
data.info()
```

## Resampling and splitting


```python
y = data['HadHeartAttack'] 
X = data.drop(columns=['HadHeartAttack'], inplace=True) 

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```


```python
if 'HadHeartAttack' in data.columns:
    # Drop the 'HadHeartAttack' column
    X = data.drop(columns=['HadHeartAttack'])
    y = data['HadHeartAttack']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
else:
    print("Column 'HadHeartAttack' not found in the DataFrame.")
```
