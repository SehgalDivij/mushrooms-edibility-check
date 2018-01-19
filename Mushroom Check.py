
# coding: utf-8

# In[1]:


import os
os.chdir('/home/divij/Desktop/Divij/ML/Mushrooms/')


# In[2]:


import csv, pandas as pd, numpy as np


# In[3]:


import sklearn


# In[4]:


mushrooms = pd.read_csv('mushrooms.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 0)
X = mushrooms.loc[:, 'cap-shape':'habitat']
y = pd.DataFrame(mushrooms['class'])


# In[5]:


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


# In[6]:


from sklearn.preprocessing import OneHotEncoder


# In[7]:


onehot_encoder = OneHotEncoder(categorical_features=[0])


# In[8]:


X_encoded = MultiColumnLabelEncoder(columns=list(X.columns)).fit_transform(X)


# In[9]:


X_1h_encoder = OneHotEncoder(categorical_features='all')


# In[10]:


X_processed = X_1h_encoder.fit_transform(X_encoded).toarray()


# In[11]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.25, random_state=0)


# In[12]:


from sklearn.linear_model import LogisticRegression


# In[14]:


logisticRegressor = LogisticRegression()


# In[15]:


y_train.shape


# In[16]:


logisticRegressor.fit(x_train, y_train)


# In[17]:


predictions = logisticRegressor.predict(x_test)


# In[18]:


score = logisticRegressor.score(x_test, y_test)


# In[19]:


from sklearn.metrics import classification_report


# In[20]:


score


# In[21]:


regression_coefficients = logisticRegressor.coef_[0][0]


# In[22]:


print(classification_report(predictions, y_test))


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


# In[24]:


cm = metrics.confusion_matrix(y_test, predictions)


# In[25]:


mushrooms


# In[26]:


# Plotting the confusion matrix into a heatmap
plt.figure(figsize=(2,2))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual category')
plt.xlabel('Predicted category')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)


# In[27]:


# Individual Weights of each feature on mushroom's class after encoded.
categorical_features = (X_encoded.dtypes.values != np.dtype('float64'))
categorical = X_encoded.columns[categorical_features]
uniq_vals = X[categorical].apply(lambda x: x.value_counts()).unstack()
uniq_vals = uniq_vals[~uniq_vals.isnull()]
enc_cols = list(uniq_vals.index.map('{0[0]}_{0[1]}'.format)) # https://stackoverflow.com/questions/41987743/merge-two-multiindex-levels-into-one-in-pandas
enc_df = pd.DataFrame(X, columns=enc_cols, index=X_encoded.index, dtype='bool')
coefs = logisticRegressor.coef_[0]
# weights dictionary to hold actual weights values for all encoded features
weights = {}
for name, value in zip(enc_cols, coefs):
    weights[name] = value
#     print(name, ':', value)


# In[32]:


# Print the total number of features that the model was worked on.
len(weights.keys())

