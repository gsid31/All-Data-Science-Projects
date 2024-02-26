#!/usr/bin/env python
# coding: utf-8

# In[159]:


#Importing the necessary librarires
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix


# In[3]:


#Reading the csv file
insurance_data = pd.read_csv("insurance.csv")


# In[8]:


#Data Description
insurance_data.describe()


# In[11]:


#Number of missing values in the dataset
insurance_data.isnull().sum()


# In[93]:


#Total number of male and female smokers
ins_data = insurance_data.groupby('smoker').aggregate(sum)
ins_age_smoker = ins_data['age']
print(ins_data)
smoker_labels = ['Non-Smoker','Smoker']
plt.bar(smoker_labels, ins_age_smoker)


# In[120]:


#Finding the people who are underweight/obese/risk of obese
need_body_Attention = insurance_data[ (insurance_data.bmi>29) | (insurance_data.bmi<15)]


# In[122]:


#The age group which has the most count of underweight/overweight
plt.figure(figsize = (12,8))
g = sb.countplot(x="age",data=need_body_Attention,palette='deep')
g.set_title("different age groups", fontsize=20)
g.set_xlabel("age", fontsize=15)
g.set_ylabel("count", fontsize=20)


# In[123]:


#Region wise concentration on underweight/overweight
plt.figure(figsize = (12,8))
g = sb.countplot(x="region",data=need_body_Attention,palette='deep')
g.set_title("different age groups", fontsize=20)
g.set_xlabel("age", fontsize=15)
g.set_ylabel("count", fontsize=20)

#The insurance company should concentrate more on southeast region in which more people 
#would be requiring health insurance


# In[126]:


for i in range(0,len(insurance_data)):
    if(insurance_data.sex[i] == 'female'):
        insurance_data.sex[i] = 0
    elif(insurance_data.sex[i] == 'male'):
        insurance_data.sex[i] = 1


# In[131]:


for i in range(0,len(insurance_data)):
    if(insurance_data.smoker[i] == 'no'):
        insurance_data.smoker[i] = 0
    elif(insurance_data.smoker[i] == 'yes'):
        insurance_data.smoker[i] = 1


# In[132]:


insurance_data


# In[134]:


for i in range(0,len(insurance_data)):
    if(insurance_data.region[i] == 'southeast'):
        insurance_data.region[i] = 0
    elif(insurance_data.region[i] == 'southwest'):
        insurance_data.region[i] = 1
    elif(insurance_data.region[i] == 'northeast'):
        insurance_data.region[i] = 2
    elif(insurance_data.region[i] == 'northwest'):
        insurance_data.region[i] = 3


# In[147]:





# In[149]:


for i in range(0,len(insurance_data)):
    if(insurance_data.expenses[i]>9382):
        insurance_data.expenses[i] = 0
    else:
        insurance_data.expenses[i] = 1


# In[154]:


X = insurance_data[['age','sex','bmi','children','smoker','region']]
Y = insurance_data[['expenses']]


# In[155]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# In[175]:


#Decision tree
classifier = tree.DecisionTreeClassifier(random_state=1,max_depth=2)
trained_model = classifier.fit(X_train, y_train)
tree.plot_tree(trained_model)


# In[179]:


y_pred = classifier.predict(X_test)


# In[181]:


conf_matrix_result = confusion_matrix(y_test, y_pred)


# In[182]:


(conf_matrix_result[0][0] + conf_matrix_result[1][1]) / sum(conf_matrix_result)


# In[188]:


(conf_matrix_result[0][0] + conf_matrix_result[1][1]) / (conf_matrix_result[0][0] + conf_matrix_result[1][1] + conf_matrix_result[0][1] + conf_matrix_result[1][0])


# In[184]:


conf_matrix_result


# # With 91.79% confidence we can say that the people above age 49.5 spend more than median and so need a good insurance
