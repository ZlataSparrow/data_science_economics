```python
pip uninstall imblearn 
```

    [33mWARNING: Skipping imblearn as it is not installed.[0m[33m
    [0mNote: you may need to restart the kernel to use updated packages.



```python
pip install scikit-learn==1.2.2
```

    Collecting scikit-learn==1.2.2
      Downloading scikit_learn-1.2.2-cp311-cp311-macosx_12_0_arm64.whl.metadata (11 kB)
    Requirement already satisfied: numpy>=1.17.3 in /opt/homebrew/anaconda3/lib/python3.11/site-packages (from scikit-learn==1.2.2) (1.24.3)
    Requirement already satisfied: scipy>=1.3.2 in /opt/homebrew/anaconda3/lib/python3.11/site-packages (from scikit-learn==1.2.2) (1.11.1)
    Requirement already satisfied: joblib>=1.1.1 in /opt/homebrew/anaconda3/lib/python3.11/site-packages (from scikit-learn==1.2.2) (1.2.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/homebrew/anaconda3/lib/python3.11/site-packages (from scikit-learn==1.2.2) (2.2.0)
    Downloading scikit_learn-1.2.2-cp311-cp311-macosx_12_0_arm64.whl (8.4 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m8.4/8.4 MB[0m [31m23.8 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hInstalling collected packages: scikit-learn
      Attempting uninstall: scikit-learn
        Found existing installation: scikit-learn 1.3.0
        Uninstalling scikit-learn-1.3.0:
          Successfully uninstalled scikit-learn-1.3.0
    Successfully installed scikit-learn-1.2.2
    Note: you may need to restart the kernel to use updated packages.



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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Sex</th>
      <th>GeneralHealth</th>
      <th>PhysicalHealthDays</th>
      <th>MentalHealthDays</th>
      <th>LastCheckupTime</th>
      <th>PhysicalActivities</th>
      <th>SleepHours</th>
      <th>RemovedTeeth</th>
      <th>HadHeartAttack</th>
      <th>...</th>
      <th>HeightInMeters</th>
      <th>WeightInKilograms</th>
      <th>BMI</th>
      <th>AlcoholDrinkers</th>
      <th>HIVTesting</th>
      <th>FluVaxLast12</th>
      <th>PneumoVaxEver</th>
      <th>TetanusLast10Tdap</th>
      <th>HighRiskLastYear</th>
      <th>CovidPos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>Female</td>
      <td>Very good</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>Within past year (anytime less than 12 months ...</td>
      <td>Yes</td>
      <td>9.0</td>
      <td>None of them</td>
      <td>No</td>
      <td>...</td>
      <td>1.60</td>
      <td>71.67</td>
      <td>27.99</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes, received Tdap</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alabama</td>
      <td>Male</td>
      <td>Very good</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Within past year (anytime less than 12 months ...</td>
      <td>Yes</td>
      <td>6.0</td>
      <td>None of them</td>
      <td>No</td>
      <td>...</td>
      <td>1.78</td>
      <td>95.25</td>
      <td>30.13</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes, received tetanus shot but not sure what type</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alabama</td>
      <td>Male</td>
      <td>Very good</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Within past year (anytime less than 12 months ...</td>
      <td>No</td>
      <td>8.0</td>
      <td>6 or more, but not all</td>
      <td>No</td>
      <td>...</td>
      <td>1.85</td>
      <td>108.86</td>
      <td>31.66</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No, did not receive any tetanus shot in the pa...</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alabama</td>
      <td>Female</td>
      <td>Fair</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>Within past year (anytime less than 12 months ...</td>
      <td>Yes</td>
      <td>9.0</td>
      <td>None of them</td>
      <td>No</td>
      <td>...</td>
      <td>1.70</td>
      <td>90.72</td>
      <td>31.32</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No, did not receive any tetanus shot in the pa...</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alabama</td>
      <td>Female</td>
      <td>Good</td>
      <td>3.0</td>
      <td>15.0</td>
      <td>Within past year (anytime less than 12 months ...</td>
      <td>Yes</td>
      <td>5.0</td>
      <td>1 to 5</td>
      <td>No</td>
      <td>...</td>
      <td>1.55</td>
      <td>79.38</td>
      <td>33.07</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No, did not receive any tetanus shot in the pa...</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>246017</th>
      <td>Virgin Islands</td>
      <td>Male</td>
      <td>Very good</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Within past 2 years (1 year but less than 2 ye...</td>
      <td>Yes</td>
      <td>6.0</td>
      <td>None of them</td>
      <td>No</td>
      <td>...</td>
      <td>1.78</td>
      <td>102.06</td>
      <td>32.28</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes, received tetanus shot but not sure what type</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>246018</th>
      <td>Virgin Islands</td>
      <td>Female</td>
      <td>Fair</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>Within past year (anytime less than 12 months ...</td>
      <td>Yes</td>
      <td>7.0</td>
      <td>None of them</td>
      <td>No</td>
      <td>...</td>
      <td>1.93</td>
      <td>90.72</td>
      <td>24.34</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No, did not receive any tetanus shot in the pa...</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>246019</th>
      <td>Virgin Islands</td>
      <td>Male</td>
      <td>Good</td>
      <td>0.0</td>
      <td>15.0</td>
      <td>Within past year (anytime less than 12 months ...</td>
      <td>Yes</td>
      <td>7.0</td>
      <td>1 to 5</td>
      <td>No</td>
      <td>...</td>
      <td>1.68</td>
      <td>83.91</td>
      <td>29.86</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes, received tetanus shot but not sure what type</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>246020</th>
      <td>Virgin Islands</td>
      <td>Female</td>
      <td>Excellent</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>Within past year (anytime less than 12 months ...</td>
      <td>Yes</td>
      <td>7.0</td>
      <td>None of them</td>
      <td>No</td>
      <td>...</td>
      <td>1.70</td>
      <td>83.01</td>
      <td>28.66</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes, received tetanus shot but not sure what type</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>246021</th>
      <td>Virgin Islands</td>
      <td>Male</td>
      <td>Very good</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Within past year (anytime less than 12 months ...</td>
      <td>No</td>
      <td>5.0</td>
      <td>None of them</td>
      <td>Yes</td>
      <td>...</td>
      <td>1.83</td>
      <td>108.86</td>
      <td>32.55</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No, did not receive any tetanus shot in the pa...</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>246022 rows × 40 columns</p>
</div>




```python
#DELETE

print(data['AgeCategory'])
```

    0            Age 65 to 69
    1            Age 70 to 74
    2            Age 75 to 79
    3         Age 80 or older
    4         Age 80 or older
                   ...       
    246017       Age 60 to 64
    246018       Age 25 to 29
    246019       Age 65 to 69
    246020       Age 50 to 54
    246021       Age 70 to 74
    Name: AgeCategory, Length: 246022, dtype: object



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

    0         67
    1         72
    2         77
    3         80
    4         80
              ..
    246017    62
    246018    27
    246019    67
    246020    52
    246021    72
    Name: AgeCategory, Length: 246022, dtype: int64



```python
print(data['Age_Category_Avg'])
```

    0         67
    1         72
    2         77
    3         80
    4         80
              ..
    246017    62
    246018    27
    246019    67
    246020    52
    246021    72
    Name: Age_Category_Avg, Length: 246022, dtype: int64



```python
print(data['AgeCategory'].dtype)
```

    int64



```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PhysicalHealthDays</th>
      <th>MentalHealthDays</th>
      <th>SleepHours</th>
      <th>AgeCategory</th>
      <th>HeightInMeters</th>
      <th>WeightInKilograms</th>
      <th>BMI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>246022.000000</td>
      <td>246022.000000</td>
      <td>246022.000000</td>
      <td>246022.000000</td>
      <td>246022.000000</td>
      <td>246022.000000</td>
      <td>246022.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.119026</td>
      <td>4.167140</td>
      <td>7.021331</td>
      <td>55.392262</td>
      <td>1.705150</td>
      <td>83.615179</td>
      <td>28.668136</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.405844</td>
      <td>8.102687</td>
      <td>1.440681</td>
      <td>17.218703</td>
      <td>0.106654</td>
      <td>21.323156</td>
      <td>6.513973</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>21.000000</td>
      <td>0.910000</td>
      <td>28.120000</td>
      <td>12.020000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>42.000000</td>
      <td>1.630000</td>
      <td>68.040000</td>
      <td>24.270000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>57.000000</td>
      <td>1.700000</td>
      <td>81.650000</td>
      <td>27.460000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>72.000000</td>
      <td>1.780000</td>
      <td>95.250000</td>
      <td>31.890000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>30.000000</td>
      <td>30.000000</td>
      <td>24.000000</td>
      <td>80.000000</td>
      <td>2.410000</td>
      <td>292.570000</td>
      <td>97.650000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# checking if we still have any null data (even though the author of the file says it is already cleaned)

data.isnull().sum()
```




    State                        0
    Sex                          0
    GeneralHealth                0
    PhysicalHealthDays           0
    MentalHealthDays             0
    LastCheckupTime              0
    PhysicalActivities           0
    SleepHours                   0
    RemovedTeeth                 0
    HadHeartAttack               0
    HadAngina                    0
    HadStroke                    0
    HadAsthma                    0
    HadSkinCancer                0
    HadCOPD                      0
    HadDepressiveDisorder        0
    HadKidneyDisease             0
    HadArthritis                 0
    HadDiabetes                  0
    DeafOrHardOfHearing          0
    BlindOrVisionDifficulty      0
    DifficultyConcentrating      0
    DifficultyWalking            0
    DifficultyDressingBathing    0
    DifficultyErrands            0
    SmokerStatus                 0
    ECigaretteUsage              0
    ChestScan                    0
    RaceEthnicityCategory        0
    AgeCategory                  0
    HeightInMeters               0
    WeightInKilograms            0
    BMI                          0
    AlcoholDrinkers              0
    HIVTesting                   0
    FluVaxLast12                 0
    PneumoVaxEver                0
    TetanusLast10Tdap            0
    HighRiskLastYear             0
    CovidPos                     0
    dtype: int64




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 246022 entries, 0 to 246021
    Data columns (total 40 columns):
     #   Column                     Non-Null Count   Dtype  
    ---  ------                     --------------   -----  
     0   State                      246022 non-null  object 
     1   Sex                        246022 non-null  object 
     2   GeneralHealth              246022 non-null  object 
     3   PhysicalHealthDays         246022 non-null  float64
     4   MentalHealthDays           246022 non-null  float64
     5   LastCheckupTime            246022 non-null  object 
     6   PhysicalActivities         246022 non-null  object 
     7   SleepHours                 246022 non-null  float64
     8   RemovedTeeth               246022 non-null  object 
     9   HadHeartAttack             246022 non-null  object 
     10  HadAngina                  246022 non-null  object 
     11  HadStroke                  246022 non-null  object 
     12  HadAsthma                  246022 non-null  object 
     13  HadSkinCancer              246022 non-null  object 
     14  HadCOPD                    246022 non-null  object 
     15  HadDepressiveDisorder      246022 non-null  object 
     16  HadKidneyDisease           246022 non-null  object 
     17  HadArthritis               246022 non-null  object 
     18  HadDiabetes                246022 non-null  object 
     19  DeafOrHardOfHearing        246022 non-null  object 
     20  BlindOrVisionDifficulty    246022 non-null  object 
     21  DifficultyConcentrating    246022 non-null  object 
     22  DifficultyWalking          246022 non-null  object 
     23  DifficultyDressingBathing  246022 non-null  object 
     24  DifficultyErrands          246022 non-null  object 
     25  SmokerStatus               246022 non-null  object 
     26  ECigaretteUsage            246022 non-null  object 
     27  ChestScan                  246022 non-null  object 
     28  RaceEthnicityCategory      246022 non-null  object 
     29  AgeCategory                246022 non-null  int64  
     30  HeightInMeters             246022 non-null  float64
     31  WeightInKilograms          246022 non-null  float64
     32  BMI                        246022 non-null  float64
     33  AlcoholDrinkers            246022 non-null  object 
     34  HIVTesting                 246022 non-null  object 
     35  FluVaxLast12               246022 non-null  object 
     36  PneumoVaxEver              246022 non-null  object 
     37  TetanusLast10Tdap          246022 non-null  object 
     38  HighRiskLastYear           246022 non-null  object 
     39  CovidPos                   246022 non-null  object 
    dtypes: float64(6), int64(1), object(33)
    memory usage: 75.1+ MB



```python
data.shape
```




    (246022, 40)




```python
data.value_counts('HadHeartAttack')
```




    HadHeartAttack
    No     232587
    Yes     13435
    Name: count, dtype: int64



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


    
![png](output_16_0.png)
    



```python
sns.countplot(x="HadHeartAttack", data=data)
plt.title("Distribution of Heart Attack")
plt.xlabel("Had Heart Attack")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=["No", "Yes"])  # Rename the x-axis tick labels
plt.show()
```


    
![png](output_17_0.png)
    



```python
#MAYBE

plt.figure(figsize=(12,12))
sns.boxplot(data=data)
plt.title('Boxplots of Numerical Features')
plt.show()
```


    
![png](output_18_0.png)
    



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

    categorical features:  ['State', 'Sex', 'GeneralHealth', 'LastCheckupTime', 'PhysicalActivities', 'RemovedTeeth', 'HadHeartAttack', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands', 'SmokerStatus', 'ECigaretteUsage', 'ChestScan', 'RaceEthnicityCategory', 'AgeCategory', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'TetanusLast10Tdap', 'HighRiskLastYear', 'CovidPos']
    numerical features:  ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'HeightInMeters', 'WeightInKilograms', 'BMI']





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Sex</th>
      <th>GeneralHealth</th>
      <th>PhysicalHealthDays</th>
      <th>MentalHealthDays</th>
      <th>LastCheckupTime</th>
      <th>PhysicalActivities</th>
      <th>SleepHours</th>
      <th>RemovedTeeth</th>
      <th>HadHeartAttack</th>
      <th>...</th>
      <th>HeightInMeters</th>
      <th>WeightInKilograms</th>
      <th>BMI</th>
      <th>AlcoholDrinkers</th>
      <th>HIVTesting</th>
      <th>FluVaxLast12</th>
      <th>PneumoVaxEver</th>
      <th>TetanusLast10Tdap</th>
      <th>HighRiskLastYear</th>
      <th>CovidPos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>Female</td>
      <td>Very good</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>Within past year (anytime less than 12 months ...</td>
      <td>Yes</td>
      <td>9.0</td>
      <td>None of them</td>
      <td>No</td>
      <td>...</td>
      <td>1.60</td>
      <td>71.67</td>
      <td>27.99</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes, received Tdap</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alabama</td>
      <td>Male</td>
      <td>Very good</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Within past year (anytime less than 12 months ...</td>
      <td>Yes</td>
      <td>6.0</td>
      <td>None of them</td>
      <td>No</td>
      <td>...</td>
      <td>1.78</td>
      <td>95.25</td>
      <td>30.13</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes, received tetanus shot but not sure what type</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alabama</td>
      <td>Male</td>
      <td>Very good</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Within past year (anytime less than 12 months ...</td>
      <td>No</td>
      <td>8.0</td>
      <td>6 or more, but not all</td>
      <td>No</td>
      <td>...</td>
      <td>1.85</td>
      <td>108.86</td>
      <td>31.66</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No, did not receive any tetanus shot in the pa...</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alabama</td>
      <td>Female</td>
      <td>Fair</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>Within past year (anytime less than 12 months ...</td>
      <td>Yes</td>
      <td>9.0</td>
      <td>None of them</td>
      <td>No</td>
      <td>...</td>
      <td>1.70</td>
      <td>90.72</td>
      <td>31.32</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No, did not receive any tetanus shot in the pa...</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alabama</td>
      <td>Female</td>
      <td>Good</td>
      <td>3.0</td>
      <td>15.0</td>
      <td>Within past year (anytime less than 12 months ...</td>
      <td>Yes</td>
      <td>5.0</td>
      <td>1 to 5</td>
      <td>No</td>
      <td>...</td>
      <td>1.55</td>
      <td>79.38</td>
      <td>33.07</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No, did not receive any tetanus shot in the pa...</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>




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


    
![png](output_20_0.png)
    



    
![png](output_20_1.png)
    



    
![png](output_20_2.png)
    



    
![png](output_20_3.png)
    



    
![png](output_20_4.png)
    



    
![png](output_20_5.png)
    



    
![png](output_20_6.png)
    



    
![png](output_20_7.png)
    



    
![png](output_20_8.png)
    



    
![png](output_20_9.png)
    



    
![png](output_20_10.png)
    



    
![png](output_20_11.png)
    



    
![png](output_20_12.png)
    



    
![png](output_20_13.png)
    



    
![png](output_20_14.png)
    



    
![png](output_20_15.png)
    



    
![png](output_20_16.png)
    



    
![png](output_20_17.png)
    



    
![png](output_20_18.png)
    



    
![png](output_20_19.png)
    



    
![png](output_20_20.png)
    



    
![png](output_20_21.png)
    



    
![png](output_20_22.png)
    



    
![png](output_20_23.png)
    



    
![png](output_20_24.png)
    



    
![png](output_20_25.png)
    



    
![png](output_20_26.png)
    



    
![png](output_20_27.png)
    



    
![png](output_20_28.png)
    



    
![png](output_20_29.png)
    



    
![png](output_20_30.png)
    



    
![png](output_20_31.png)
    



    
![png](output_20_32.png)
    



    
![png](output_20_33.png)
    



```python
sns.heatmap(num_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
```


    
![png](output_21_0.png)
    



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


    
![png](output_22_0.png)
    



```python
sns.pairplot(data,height=2)
plt.show()
```

    /opt/homebrew/anaconda3/lib/python3.11/site-packages/seaborn/axisgrid.py:118: UserWarning: The figure layout has changed to tight
      self._figure.tight_layout(*args, **kwargs)



    
![png](output_23_1.png)
    



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


    
![png](output_24_0.png)
    



    
![png](output_24_1.png)
    



    
![png](output_24_2.png)
    



    
![png](output_24_3.png)
    



    
![png](output_24_4.png)
    



    
![png](output_24_5.png)
    



    
![png](output_24_6.png)
    



    
![png](output_24_7.png)
    



    
![png](output_24_8.png)
    



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

    No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.



    
![png](output_25_1.png)
    


    No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.



    
![png](output_25_3.png)
    


    No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.



    
![png](output_25_5.png)
    


    No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.



    
![png](output_25_7.png)
    


    No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.



    
![png](output_25_9.png)
    


    No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.



    
![png](output_25_11.png)
    


    No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.



    
![png](output_25_13.png)
    



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


    
![png](output_26_0.png)
    



```python
num_data = num_data.drop(columns=['HadHeartAttack'])
num_data.hist(figsize=(16, 20), bins=40, xlabelsize=6, ylabelsize=6);
```


    
![png](output_27_0.png)
    



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


    
![png](output_28_0.png)
    



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


    
![png](output_29_0.png)
    


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


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    Cell In[229], line 3
          1 from sklearn import preprocessing 
          2 from sklearn.preprocessing import RobustScaler
    ----> 3 from imblearn.over_sampling import SMOTE


    File /opt/homebrew/anaconda3/lib/python3.11/site-packages/imblearn/__init__.py:52
         48     sys.stderr.write("Partial import of imblearn during the build process.\n")
         49     # We are not importing the rest of scikit-learn during the build
         50     # process, as it may not be compiled yet
         51 else:
    ---> 52     from . import (
         53         combine,
         54         ensemble,
         55         exceptions,
         56         metrics,
         57         over_sampling,
         58         pipeline,
         59         tensorflow,
         60         under_sampling,
         61         utils,
         62     )
         63     from ._version import __version__
         64     from .base import FunctionSampler


    File /opt/homebrew/anaconda3/lib/python3.11/site-packages/imblearn/combine/__init__.py:5
          1 """The :mod:`imblearn.combine` provides methods which combine
          2 over-sampling and under-sampling.
          3 """
    ----> 5 from ._smote_enn import SMOTEENN
          6 from ._smote_tomek import SMOTETomek
          8 __all__ = ["SMOTEENN", "SMOTETomek"]


    File /opt/homebrew/anaconda3/lib/python3.11/site-packages/imblearn/combine/_smote_enn.py:12
          9 from sklearn.base import clone
         10 from sklearn.utils import check_X_y
    ---> 12 from ..base import BaseSampler
         13 from ..over_sampling import SMOTE
         14 from ..over_sampling.base import BaseOverSampler


    File /opt/homebrew/anaconda3/lib/python3.11/site-packages/imblearn/base.py:21
         18 from sklearn.utils.multiclass import check_classification_targets
         20 from .utils import check_sampling_strategy, check_target_type
    ---> 21 from .utils._param_validation import validate_parameter_constraints
         22 from .utils._validation import ArraysTransformer
         25 class SamplerMixin(BaseEstimator, metaclass=ABCMeta):


    File /opt/homebrew/anaconda3/lib/python3.11/site-packages/imblearn/utils/_param_validation.py:908
        906 from sklearn.utils._param_validation import generate_valid_param  # noqa
        907 from sklearn.utils._param_validation import validate_parameter_constraints  # noqa
    --> 908 from sklearn.utils._param_validation import (
        909     HasMethods,
        910     Hidden,
        911     Interval,
        912     Options,
        913     StrOptions,
        914     _ArrayLikes,
        915     _Booleans,
        916     _Callables,
        917     _CVObjects,
        918     _InstancesOf,
        919     _IterablesNotString,
        920     _MissingValues,
        921     _NoneConstraint,
        922     _PandasNAConstraint,
        923     _RandomStates,
        924     _SparseMatrices,
        925     _VerboseHelper,
        926     make_constraint,
        927     validate_params,
        928 )


    ImportError: cannot import name '_MissingValues' from 'sklearn.utils._param_validation' (/opt/homebrew/anaconda3/lib/python3.11/site-packages/sklearn/utils/_param_validation.py)



```python
import sklearn
print(sklearn.__version__)
```

    1.3.0



```python
pip install scikit-learn==0.22.2
```

    [31mERROR: Could not find a version that satisfies the requirement scikit-learn==0.22.2 (from versions: 0.9, 0.10, 0.11, 0.12, 0.12.1, 0.13, 0.13.1, 0.14, 0.14.1, 0.15.0, 0.15.1, 0.15.2, 0.16.0, 0.16.1, 0.17, 0.17.1, 0.18, 0.18.1, 0.18.2, 0.19.0, 0.19.1, 0.19.2, 0.20.0, 0.20.1, 0.20.2, 0.20.3, 0.20.4, 0.21.1, 0.21.2, 0.21.3, 0.22, 0.22.1, 0.22.2.post1, 0.23.0, 0.23.1, 0.23.2, 0.24.0, 0.24.1, 0.24.2, 1.0, 1.0.1, 1.0.2, 1.1.0, 1.1.1, 1.1.2, 1.1.3, 1.2.0rc1, 1.2.0, 1.2.1, 1.2.2, 1.3.0rc1, 1.3.0, 1.3.1, 1.3.2, 1.4.0rc1, 1.4.0, 1.4.1.post1)[0m[31m
    [0m[31mERROR: No matching distribution found for scikit-learn==0.22.2[0m[31m
    [0mNote: you may need to restart the kernel to use updated packages.



```python
print(sklearn.__version__)
```

    1.3.0



```python
label_encoder = preprocessing.LabelEncoder() 
for c in cat_data:
    data[c]= label_encoder.fit_transform(data[c]) 
    data[c].unique()
    
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>GeneralHealth</th>
      <th>PhysicalHealthDays</th>
      <th>MentalHealthDays</th>
      <th>LastCheckupTime</th>
      <th>PhysicalActivities</th>
      <th>SleepHours</th>
      <th>RemovedTeeth</th>
      <th>HadHeartAttack</th>
      <th>HadAngina</th>
      <th>...</th>
      <th>WeightInKilograms</th>
      <th>BMI</th>
      <th>AlcoholDrinkers</th>
      <th>HIVTesting</th>
      <th>FluVaxLast12</th>
      <th>PneumoVaxEver</th>
      <th>TetanusLast10Tdap</th>
      <th>HighRiskLastYear</th>
      <th>CovidPos</th>
      <th>Age_Category_Avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>4</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1</td>
      <td>9.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>71.67</td>
      <td>27.99</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>67</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1</td>
      <td>6.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>95.25</td>
      <td>30.13</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>72</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0</td>
      <td>8.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>108.86</td>
      <td>31.66</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>77</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1</td>
      <td>9.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>90.72</td>
      <td>31.32</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2</td>
      <td>3.0</td>
      <td>15.0</td>
      <td>3</td>
      <td>1</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>79.38</td>
      <td>33.07</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>80</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>




```python
scaler=RobustScaler()
scaled_data=scaler.fit_transform(data)
sns.boxplot(data=scaled_data)
plt.show()
```


    
![png](output_37_0.png)
    



```python
race_groups = data.groupby('RaceEthnicityCategory')['HadHeartAttack'].value_counts(normalize=True).unstack(fill_value=0)
race_groups['Ratio'] = race_groups['Yes'] / race_groups['No']

print(race_groups[['Ratio']])
```

    HadHeartAttack                    Ratio
    RaceEthnicityCategory                  
    Black only, Non-Hispanic       0.048208
    Hispanic                       0.039565
    Multiracial, Non-Hispanic      0.064873
    Other race only, Non-Hispanic  0.050887
    White only, Non-Hispanic       0.061260


Since there isn't any difference in the ratio of heart disease across different race/ethnicity categories, it suggests that race doesn't have any significant effect on heart attack in individuals.


```python
data.drop(columns=['RaceEthnicityCategory'], inplace=True)
data.head()
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Cell In[215], line 1
    ----> 1 data.drop(columns=['RaceEthnicityCategory'], inplace=True)
          2 data.head()


    File /opt/homebrew/anaconda3/lib/python3.11/site-packages/pandas/core/frame.py:5258, in DataFrame.drop(self, labels, axis, index, columns, level, inplace, errors)
       5110 def drop(
       5111     self,
       5112     labels: IndexLabel = None,
       (...)
       5119     errors: IgnoreRaise = "raise",
       5120 ) -> DataFrame | None:
       5121     """
       5122     Drop specified labels from rows or columns.
       5123 
       (...)
       5256             weight  1.0     0.8
       5257     """
    -> 5258     return super().drop(
       5259         labels=labels,
       5260         axis=axis,
       5261         index=index,
       5262         columns=columns,
       5263         level=level,
       5264         inplace=inplace,
       5265         errors=errors,
       5266     )


    File /opt/homebrew/anaconda3/lib/python3.11/site-packages/pandas/core/generic.py:4549, in NDFrame.drop(self, labels, axis, index, columns, level, inplace, errors)
       4547 for axis, labels in axes.items():
       4548     if labels is not None:
    -> 4549         obj = obj._drop_axis(labels, axis, level=level, errors=errors)
       4551 if inplace:
       4552     self._update_inplace(obj)


    File /opt/homebrew/anaconda3/lib/python3.11/site-packages/pandas/core/generic.py:4591, in NDFrame._drop_axis(self, labels, axis, level, errors, only_slice)
       4589         new_axis = axis.drop(labels, level=level, errors=errors)
       4590     else:
    -> 4591         new_axis = axis.drop(labels, errors=errors)
       4592     indexer = axis.get_indexer(new_axis)
       4594 # Case for non-unique axis
       4595 else:


    File /opt/homebrew/anaconda3/lib/python3.11/site-packages/pandas/core/indexes/base.py:6699, in Index.drop(self, labels, errors)
       6697 if mask.any():
       6698     if errors != "ignore":
    -> 6699         raise KeyError(f"{list(labels[mask])} not found in axis")
       6700     indexer = indexer[~mask]
       6701 return self.delete(indexer)


    KeyError: "['RaceEthnicityCategory'] not found in axis"



```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 246022 entries, 0 to 246021
    Data columns (total 39 columns):
     #   Column                     Non-Null Count   Dtype  
    ---  ------                     --------------   -----  
     0   State                      246022 non-null  object 
     1   Sex                        246022 non-null  object 
     2   GeneralHealth              246022 non-null  object 
     3   PhysicalHealthDays         246022 non-null  float64
     4   MentalHealthDays           246022 non-null  float64
     5   LastCheckupTime            246022 non-null  object 
     6   PhysicalActivities         246022 non-null  object 
     7   SleepHours                 246022 non-null  float64
     8   RemovedTeeth               246022 non-null  object 
     9   HadHeartAttack             246022 non-null  object 
     10  HadAngina                  246022 non-null  object 
     11  HadStroke                  246022 non-null  object 
     12  HadAsthma                  246022 non-null  object 
     13  HadSkinCancer              246022 non-null  object 
     14  HadCOPD                    246022 non-null  object 
     15  HadDepressiveDisorder      246022 non-null  object 
     16  HadKidneyDisease           246022 non-null  object 
     17  HadArthritis               246022 non-null  object 
     18  HadDiabetes                246022 non-null  object 
     19  DeafOrHardOfHearing        246022 non-null  object 
     20  BlindOrVisionDifficulty    246022 non-null  object 
     21  DifficultyConcentrating    246022 non-null  object 
     22  DifficultyWalking          246022 non-null  object 
     23  DifficultyDressingBathing  246022 non-null  object 
     24  DifficultyErrands          246022 non-null  object 
     25  SmokerStatus               246022 non-null  object 
     26  ECigaretteUsage            246022 non-null  object 
     27  ChestScan                  246022 non-null  object 
     28  AgeCategory                246022 non-null  object 
     29  HeightInMeters             246022 non-null  float64
     30  WeightInKilograms          246022 non-null  float64
     31  BMI                        246022 non-null  float64
     32  AlcoholDrinkers            246022 non-null  object 
     33  HIVTesting                 246022 non-null  object 
     34  FluVaxLast12               246022 non-null  object 
     35  PneumoVaxEver              246022 non-null  object 
     36  TetanusLast10Tdap          246022 non-null  object 
     37  HighRiskLastYear           246022 non-null  object 
     38  CovidPos                   246022 non-null  object 
    dtypes: float64(6), object(33)
    memory usage: 73.2+ MB


## Resampling and splitting


```python
y = data['HadHeartAttack'] 
X = data.drop(columns=['HadHeartAttack'], inplace=True) 

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[217], line 4
          1 y = data['HadHeartAttack'] 
          2 X = data.drop(columns=['HadHeartAttack'], inplace=True) 
    ----> 4 smote = SMOTE(random_state=42)
          5 X_resampled, y_resampled = smote.fit_resample(X, y)


    NameError: name 'SMOTE' is not defined



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

    Column 'HadHeartAttack' not found in the DataFrame.

