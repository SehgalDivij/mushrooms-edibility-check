
# coding: utf-8

# In[1]:


import os
os.chdir('/home/divij/Desktop/Divij/ML/Mushrooms/')
import csv, pandas as pd, numpy as np
import sklearn


# In[2]:


mushrooms = pd.read_csv('mushrooms.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 0)
X = mushrooms.loc[:, 'cap-shape':'habitat']
y = pd.DataFrame(mushrooms['class'])


# In[3]:


from sklearn.preprocessing import LabelEncoder
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# In[4]:


from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(categorical_features=[0])
X_encoded = MultiColumnLabelEncoder(columns=list(X.columns)).fit_transform(X)
X_1h_encoder = OneHotEncoder(categorical_features='all')
X_processed = X_1h_encoder.fit_transform(X_encoded).toarray()


# In[5]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.25, random_state=0)


# In[6]:


from sklearn.linear_model import LogisticRegression
logisticRegressor = LogisticRegression()
logisticRegressor.fit(x_train, y_train)


# In[7]:


predictions = logisticRegressor.predict(x_test)


# In[8]:


print(predictions)


# In[9]:


score = logisticRegressor.score(x_test, y_test)


# In[10]:


print(score)


# In[11]:


# A score of 1. Yay! or is it a mistake?
from sklearn.metrics import classification_report
regression_coefficients = logisticRegressor.coef_[0][0]


# In[12]:


# Now a confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


# In[13]:


# Constructing the confusion matrix
cm = metrics.confusion_matrix(y_test, predictions)


# In[14]:


# Plotting a confusion matrix
plt.figure(figsize=(2,2))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual category')
plt.xlabel('Predicted category')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)


# In[15]:


# Individual Weights of each feature on the class of the mushroom
categorical_features = (X_encoded.dtypes.values != np.dtype('float64'))
categorical = X_encoded.columns[categorical_features]
uniq_vals = X[categorical].apply(lambda x: x.value_counts()).unstack()
uniq_vals = uniq_vals[~uniq_vals.isnull()]
enc_cols = list(uniq_vals.index.map('{0[0]}_{0[1]}'.format)) # https://stackoverflow.com/questions/41987743/merge-two-multiindex-levels-into-one-in-pandas
enc_df = pd.DataFrame(X, columns=enc_cols, index=X_encoded.index, dtype='bool')
coefs = logisticRegressor.coef_[0]
# weights dictionary to hold actual weights values
weights = {}
for name, value in zip(enc_cols, coefs):
    weights[name] = value
#     print(name, ':', value)


# In[16]:


weights


# In[17]:


def get_original_feature_weights(weights):
    """
    Reconstruct original feature weights from encoded categorical features.
    """
    original_feature_weights = {}
    base_weight = 0
    for item in weights:
        feature_name = '_'.join(item.split('_')[0:-1])
        if feature_name not in original_feature_weights:
            original_feature_weights[feature_name] = weights[item]
        else:
            original_feature_weights[feature_name] = original_feature_weights[feature_name] + weights[item]
    return original_feature_weights


# In[23]:


original_weights = get_original_feature_weights(weights)
print(original_weights)


# In[22]:


# An attempt to reconstruct the feature weights for the original feature set.
# All seem to have more or less equal effect
# Which means that this last part of resonstructing the final weights for all features is more or less not logically correct.
plot = plt.figure(figsize=(40,10))
plt.bar(range(len(original_weights)), original_weights.values(), align='center')
plt.xticks(range(len(original_weights)), original_weights.keys())
plt.show()

