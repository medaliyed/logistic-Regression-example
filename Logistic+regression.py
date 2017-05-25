
# coding: utf-8

# In[1]:

# Data Imports
import numpy as np
import pandas as pd
from pandas import Series,DataFrame

# Math
import math

# Plot imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic('matplotlib inline')

# Machine Learning Imports
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

# For evaluating our ML results
from sklearn import metrics

# Dataset Import
import statsmodels.api as sm


# In[2]:

# Logistic Function
def logistic(t):
    return 1.0 / (1 + math.exp((-1.0)*t) )

# Set t from -6 to 6 ( 500 elements, linearly spaced)
t = np.linspace(-6,6,500)

# Set up y values (using list comprehension)
y = np.array([logistic(ele) for ele in t])

# Plot
plt.plot(t,y)
plt.title(' Logistic Function ')


# In[3]:

#The dataset is packaged within Statsmodels. It is a data set from a 1974 survey of women by Redbook magazine. Married women were asked if they have had extramarital affairs.
#The published work on the data set can be found in:
# https://fairmodel.econ.yale.edu/rayfair/pdf/1978a200.pdf


# In[4]:

df = sm.datasets.fair.load_pandas().data


# In[5]:

df.head()


# In[6]:

# Create check loyalty function 
def affair_check(x):
    if x != 0:
        return 1
    else:
        return 0

# Apply to DataFrame
df['Had_Affair'] = df['affairs'].apply(affair_check)


# In[7]:

df


# In[8]:

# Groupby Had Affair column
df.groupby('Had_Affair').mean()


# In[12]:

sns.factorplot('age',data=df,kind='count',hue='Had_Affair',palette='coolwarm')


# In[13]:

sns.factorplot('children',data=df,kind='count',hue='Had_Affair',palette='coolwarm')


# In[14]:

sns.factorplot('yrs_married',data=df,kind='count',hue='Had_Affair',palette='coolwarm')


# In[15]:

sns.factorplot('educ',data=df,kind='count',hue='Had_Affair',palette='coolwarm')


# In[18]:

#these h's ain't layal


# In[20]:

#data preparation
#we have 2 columns called Occupation and Husband's Occupation. These columns are in a format know as Categorical Variables.


# In[21]:

# Create new DataFrames for the Categorical Variables
occ_dummies = pd.get_dummies(df['occupation'])
hus_occ_dummies = pd.get_dummies(df['occupation_husb'])

# Let's take a quick look at the results
occ_dummies.head()


# In[22]:

# Create column names for the new DataFrames
occ_dummies.columns = ['occ1','occ2','occ3','occ4','occ5','occ6']
hus_occ_dummies.columns = ['hocc1','hocc2','hocc3','hocc4','hocc5','hocc6']


# In[23]:

# Set X as new DataFrame without the occupation columns or the Y target
X = df.drop(['occupation','occupation_husb','Had_Affair'],axis=1)


# In[24]:

# Concat the dummy DataFrames Together
dummies = pd.concat([occ_dummies,hus_occ_dummies],axis=1)
# Now Concat the X DataFrame with the dummy variables
X = pd.concat([X,dummies],axis=1)

# Preview of Result
X.head()


# In[25]:

# Set Y as Target class, Had Affair
Y = df.Had_Affair

# Preview
Y.head()


# In[26]:

df.head()


# In[27]:

#Multicollinearity Consideration.


# In[28]:

# Dropping one column of each dummy variable set to avoid multicollinearity
X = X.drop('occ1',axis=1)
X = X.drop('hocc1',axis=1)

# Drop affairs column so Y target makes sense
X = X.drop('affairs',axis=1)

# PReview
X.head()


# In[29]:

# Flatten array
Y = np.ravel(Y)

# Check result
Y


# In[30]:

#Logistic Regression with SciKit Learn


# In[31]:

# Create LogisticRegression model
log_model = LogisticRegression()

# Fit our data
log_model.fit(X,Y)

# Check our accuracy
log_model.score(X,Y)


# In[32]:

# Check percentage of women that had affairs
Y.mean()


# In[40]:

# Use zip to bring the column names and the np.transpose function to bring together the coefficients from the model
coeff_df = DataFrame(list(zip(X.columns, np.transpose(log_model.coef_))))


# In[41]:

coeff_df


# In[42]:

#negative values: loyalty ....positive: cheating


# In[43]:

#Testing and Training Data Sets


# In[44]:

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# Make a new log_model
log_model2 = LogisticRegression()

# Now fit the new model
log_model2.fit(X_train, Y_train)


# In[46]:

# Predict the classes of the testing data set
class_predict = log_model2.predict(X_test)

# Compare the predicted classes to the actual test classes
print (metrics.accuracy_score(Y_test,class_predict))


# In[47]:

#Now we have a 73.35% accuracy score, which is basically the same as our previous accuracy score, 72.58%.


# In[ ]:




# In[ ]:



