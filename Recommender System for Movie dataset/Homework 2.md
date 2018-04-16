
# Compare ItemBased and UserBased knn algorithms on the "ml-100k" data set.
### Step 1 Import libraries
Import standard tools


```python
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import ttest_rel
import numpy as np
from matplotlib import pyplot as plt
import heapq
import os
from math import log
```

Import from Surprise


```python
from surprise import KNNWithMeans
from surprise import Reader, Dataset, accuracy
from surprise.model_selection import cross_validate, KFold, ShuffleSplit
from surprise.prediction_algorithms import PredictionImpossible
```

Import from local code


```python
%load_ext autoreload
%autoreload
from sigweight import KNNSigWeighting
from prec_recall import pr_eval, precision_recall_at_k
```

### Step 2 Load the "Filmtrust" data
First load using Surprise


```python
# path to dataset file
file_path = os.path.expanduser('/Users/ivy/Desktop/CSC577/Week 2/Homework 2/filmtrust.txt')

# As we're loading a custom dataset, we need to define a reader. In the
# filmtrust dataset, each line has the following format:
# 'user item rating unused', separated by ' ' characters.
reader = Reader(line_format='user item rating', sep=' ', skip_lines=0,rating_scale=(0.5, 4))

data = Dataset.load_from_file(file_path, reader=reader)
```

Convert to a data frame


```python
# DF creation
df = pd.DataFrame(data.raw_ratings, columns=["user","item","rating","unused"])
```

### Step 3 Get data set stats
Group by item and count to get item rating distribution


```python
# Group by item and count to get item rating
# Get x-axis: the number of times that an item has been rated
df_item = df.groupby('item').rating.count().to_frame('item_rating_count').reset_index()
x = df_item['item_rating_count']
```

Plot item rating distribution (use log y axis)


```python
plt.hist(x,log=True,bins=15)
#plt.yscale('log') #another way of log y-axis
plt.title("Item rating distribution")
plt.xlabel("Number of item rated times")
plt.ylabel("Log Number of items with that number of ratings")
plt.show()
```
<p align="center"><img width="50%" src="https://github.com/IvyYun/Recommender-System/blob/master/Recommender%20System%20for%20Movie%20dataset/Images/output_13_0.png" /></p>



Group by user and count to get user rating distribution


```python
df_user = df.groupby('user').rating.count().to_frame('user_rating_count').reset_index()
df_user.head()
x1 = df_user['user_rating_count']
```

Plot user rating distribution (also log y axis)


```python
plt.hist(x1,log=True,bins=30)
plt.title("User rating distribution")
plt.xlabel("Number of user rated times")
plt.ylabel("Log Number of users with that number of ratings")
plt.show()
```


![png](output_17_0.png)


Plot distribution of ratings


```python
# produce histogram showing the rating distribution at each rating level. 
# Legal rating levels in this data set start at 0.5 and go up to 4 in steps of 0.5.
#pd.DataFrame(df.rating.value_counts().reset_index(),columns=['rating','rating_count'])
d = (df.rating.value_counts().reset_index())
d.columns = ['rating','rating_count']
d
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
      <th>rating</th>
      <th>rating_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.0</td>
      <td>9171</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>7877</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.5</td>
      <td>7142</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.5</td>
      <td>4392</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>3113</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.5</td>
      <td>1601</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0</td>
      <td>1141</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.5</td>
      <td>1060</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.bar(d.rating,d.rating_count,width=0.35)
plt.xticks(np.arange(0.5,4.5,0.5))
plt.xlabel("Rating Level")
plt.ylabel("Rating Count")
plt.show()
```


![png](output_20_0.png)


### Step 4 Compare algorithms RMSE
Use three algorithms
- regular KNNWithMeans, user-based, Pearson correlation
- significance weighting with corate_threshold=100
- significance weighting with default corate_threshold of 50
Use 5-fold cross-validation
Calculate RMSE only


```python
%autoreload
```

Produce a boxplot of the RMSE values. (They are all pretty similar.)


```python
# distance metric pearson for user-based 
sim_options1 = {'name': 'pearson',
               'user_based': True  # compute  similarities between users
               }

sim_options2={'name': 'pearson', 'user_based': True,
              'corate_threshold': 100}


# Regular KNNWithMeans, user-based, Pearson correlation
algo1 = KNNWithMeans(sim_options=sim_options1)

# Significance weighting with corate_threshold=100
algo2 = KNNSigWeighting(sim_options=sim_options2)

# Significance weighting with default corate_threshold of 50 Use 5-fold cross-validation Calculate RMSE only
algo3 = KNNSigWeighting(sim_options=sim_options1)

# Define a cross-validation iterator
kf = KFold(n_splits=5,random_state=200)

RMSE_KNNWithMeans = []
RMSE_KNNSigWeighting1 =[] #corate_threshold=100
RMSE_KNNSigWeighting2 = [] # default corate_threshold of 50


# Run 5-fold cross-validation and print results
for trainset, testset in kf.split(data):
    algo1.fit(trainset)
    algo2.fit(trainset)
    algo3.fit(trainset)
    KNNWithMeans_predictions = algo1.test(testset)
    KNNSigWeighting_predictions1 = algo2.test(testset)
    KNNSigWeighting_predictions2 = algo3.test(testset) # user-based
    # Then compute RMSE
    RMSE1 = accuracy.rmse(KNNWithMeans_predictions)
    RMSE2 = accuracy.rmse(KNNSigWeighting_predictions1)
    RMSE3 = accuracy.rmse(KNNSigWeighting_predictions2)

    RMSE_KNNWithMeans.append(RMSE1)
    RMSE_KNNSigWeighting1.append(RMSE2)
    RMSE_KNNSigWeighting2.append(RMSE3)
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    RMSE: 0.8078
    RMSE: 0.7962
    RMSE: 0.7963
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    RMSE: 0.8381
    RMSE: 0.8332
    RMSE: 0.8331
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    RMSE: 0.8360
    RMSE: 0.8319
    RMSE: 0.8319
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    RMSE: 0.8245
    RMSE: 0.8213
    RMSE: 0.8214
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    RMSE: 0.8209
    RMSE: 0.8151
    RMSE: 0.8151



```python
lists = [RMSE_KNNWithMeans,RMSE_KNNSigWeighting1,RMSE_KNNSigWeighting2]

plt.boxplot(lists,vert=False,labels=['RMSE_User_based', 'RMSE_KNNSigWeighting100','RMSE_KNNSigWeightingDefault'])

plt.title("RMSE Distribution")
plt.xlabel("RMSE Scores")
plt.show()
```


![png](output_25_0.png)


### Step 5 Compare algorithms precall, precision
Use the same algorithms but compute precision and recall.
Use `pr_eval` from the `prec_recall.py` file. This is more or less the same code as in the Surprise example code, but it returns a Pandas data frame with the results.


```python
v1 = pr_eval(algo1,data,kf,n=10,threshold=3.3)
v2 = pr_eval(algo2,data,kf,n=10,threshold=3.3)
v3 = pr_eval(algo3,data,kf,n=10,threshold=3.3)
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Precision-Recall
    {'Fold 1': Precision    0.347471
    Recall       0.320916
    dtype: float64, 'Fold 2': Precision    0.344964
    Recall       0.331622
    dtype: float64, 'Fold 3': Precision    0.369606
    Recall       0.354073
    dtype: float64, 'Fold 4': Precision    0.372673
    Recall       0.346919
    dtype: float64, 'Fold 5': Precision    0.369171
    Recall       0.338038
    dtype: float64}
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Precision-Recall
    {'Fold 1': Precision    0.375428
    Recall       0.351080
    dtype: float64, 'Fold 2': Precision    0.382591
    Recall       0.362705
    dtype: float64, 'Fold 3': Precision    0.389973
    Recall       0.377616
    dtype: float64, 'Fold 4': Precision    0.398027
    Recall       0.375663
    dtype: float64, 'Fold 5': Precision    0.383674
    Recall       0.361733
    dtype: float64}
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    Precision-Recall
    {'Fold 1': Precision    0.375428
    Recall       0.351080
    dtype: float64, 'Fold 2': Precision    0.382723
    Recall       0.362777
    dtype: float64, 'Fold 3': Precision    0.389973
    Recall       0.377616
    dtype: float64, 'Fold 4': Precision    0.397947
    Recall       0.375597
    dtype: float64, 'Fold 5': Precision    0.383674
    Recall       0.361733
    dtype: float64}


Create a box plot of the precision values


```python
precision1 = v1.values[0]
precision2 = v2.values[0]
precision3 = v3.values[0]

precisions = [precision1,precision2,precision3]
plt.boxplot(precisions,vert=False,labels=['KNNWithMeans','KNNSigWeighting100','KNNSigWeightingDefault'])
plt.title("Precision Value Distribution")
plt.xlabel("Precision Value")

```




    Text(0.5,0,'Precision Value')




![png](output_29_1.png)


Boxplot of the recall values


```python
recall1 = v1.values[1]
recall2 = v2.values[1]
recall3 = v3.values[1]

recalls = [recall1,recall2,recall3]
plt.boxplot(recalls,vert=False,labels=['KNNWithMeans','KNNSigWeighting100','KNNSigWeightingDefault'])
plt.title("Recall Score Distribution")
plt.xlabel("Recall Score")
```




    Text(0.5,0,'Recall Score')




![png](output_31_1.png)


Scatter plot of precision (y) vs recall (x). Include legend. See example below.


```python
fig, ax = plt.subplots()
plt.scatter(recall1,precision1,c='blue',label='KNN')
plt.scatter(recall2,precision1,c='green',label='Sig100')
plt.scatter(recall3,precision1,c='red',label='Sig50')
plt.legend(loc='lower right')
plt.title("Recall-Precision Scatter Plot")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()
```


![png](output_33_0.png)


![pr-scatter.png](attachment:pr-scatter.png)

# Test Code
Do not modify. Use this to test your code.

Imports


```python
from surprise.model_selection import PredefinedKFold
from surprise.accuracy import rmse
```

Read test data


```python
TEST_reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
TEST_train_file = '/Users/ivy/Desktop/CSC577/Week 2/Homework 2/test-data-train.csv'
TEST_test_file = '/Users/ivy/Desktop/CSC577/Week 2/Homework 2/test-data-test.csv'
TEST_folds_files = [(TEST_train_file, TEST_test_file)]

TEST_data = Dataset.load_from_folds(TEST_folds_files, reader=TEST_reader)
```

Create single split


```python
TEST_pkf = PredefinedKFold()
TEST_trainset, TEST_testset = next(TEST_pkf.split(TEST_data))
```

Run algorithm


```python
%autoreload
TEST_algo = KNNSigWeighting(k=3, sim_options={'name': 'pearson', 'user_based': True, 
                                              'corate_threshold': 4.0})
TEST_algo.fit(TEST_trainset)
```

Test similarity matrix


```python
TEST_correct_values = [(0,1, 0.375), (4, 3, -0.75), (2, 1, 0.7280)]
TEST_epsilon = 0.0001

for u1, u2, sim in TEST_correct_values:
    if abs(sim - TEST_algo.sim[u1, u2]) < TEST_epsilon:
        print("Your implementation is correct.")
    else:
        print("Your implementation is not correct. Keep working.")
```
