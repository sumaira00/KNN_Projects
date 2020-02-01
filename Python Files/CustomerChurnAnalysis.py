import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#df=pd.read_csv(r'C:\Users\saim-zain\Desktop\DATA SCIENCE\Python Certification Training Course imp\10 Project\Project 6-Capstone Project\Dataset.zip\Dataset\customer_churn.csv')
df=pd.read_csv(r"C:\Users\saim-zain\Desktop\DATA SCIENCE\WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.info())

df.describe()

df.head()

#a.	Extract the 5th column & store it in ‘customer_5’
customer_5=df.iloc[:,4]
customer_5.head()

#b.	Extract the 15th column & store it in ‘customer_15’
customer_15=df[df.columns[14]]
print(customer_15.head())

#c.	Extract all the male senior citizens whose Payment Method is Electronic check & store the result in 
#.  ‘senior_male_electronic’
senior_male_electronic=df[(df['PaymentMethod']=='Electronic check')&(df['gender']== 'Male')&(df['SeniorCitizen']==1)]
senior_male_electronic.head()

#d.	Extract all those customers whose tenure is greater than 70 months or their Monthly charges 
df_tenure_more_70=df[(df['tenure']>70)|(df['MonthlyCharges']>70)]
df_tenure_more_70.head()

#e.	Extract all the customers whose Contract is of two years
#.  payment method is Mailed check & the value of Churn is ‘Yes’ & store the result in ‘two_mail_yes’
two_mail_yes=df[(df['Contract']=='Two year')&(df['PaymentMethod']=='Mailed check')&(df['Churn']=='Yes')]
two_mail_yes.head()

#f.	Extract 333 random records from the customer_churn dataframe & store the result in ‘customer_333’
customer_333=df.sample(n=333)
len(customer_333)

#g.	Get the count of different levels from the ‘Churn’ column
df['Churn'].value_counts()


#a.	Build a bar-plot for the ’InternetService’ column:
#   i.	Set x-axis label to ‘Categories of Internet Service’
#   ii.	Set y-axis label to ‘Count of Categories’
#   iii	Set the title of plot to be ‘Distribution of Internet Service’
#   iv.	Set the color of the bars to be ‘orange’

plt.figure(figsize=(9,8))
sns.set_style('whitegrid')
sns.countplot(x='InternetService',data=df,palette='viridis',color='orange');
plt.xlabel("Categories of Internet Service")
plt.ylabel('Count of Categories')
plt.title('Distribution of Internet Service');

#b.	Build a histogram for the ‘tenure’ column:
#. i.	Set the number of bins to be 30
#. ii.	Set the color of the bins  to be ‘green’
#. iii.	Assign the title ‘Distribution of tenure’

plt.figure(figsize=(6,6))
df['tenure'].hist(bins=30,color='green')
plt.title('Distribution of tenure');

#c.	Build a scatter-plot between ‘MonthlyCharges’ & ‘tenure’. Map ‘MonthlyCharges’ to the y-axis & ‘tenure’ to the ‘x-axis’:
# i.	Assign the points a color of ‘brown’
# ii.	Set the x-axis label to ‘Tenure of customer’
# iii.	Set the y-axis label to ‘Monthly Charges of customer’
# iv.	Set the title to ‘Tenure vs Monthly Charges’
sns.jointplot(x='tenure',y='MonthlyCharges',data=df,color='brown')
plt.xlabel("Tenure of customer")
plt.ylabel('Monthly Charges of customer')
plt.title('Tenure vs Monthly Charges');


#d.	Build a box-plot between ‘tenure’ & ‘Contract’. Map ‘tenure’ on the y-axis & ‘Contract’ on the x-axis. 
sns.boxplot(x='Contract',y='tenure',data=df)

#C)	Linear Regression:
#. a.	Build a simple linear model where dependent variable is ‘MonthlyCharges’ and independent variable is ‘tenure’
#. i.	Divide the dataset into train and test sets in 70:30 ratio. 
#. ii.	Build the model on train set and predict the values on test set
#. iii.	After predicting the values, find the root mean square error
#. iv.	Find out the error in prediction & store the result in ‘error’
#  v.	Find the root mean square error

X=pd.DataFrame(df['tenure'])
y=df['MonthlyCharges']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=42)


from sklearn.linear_model import LinearRegression
model=LinearRegression()

model.fit(X_train,y_train)

predictions=model.predict(X_test)
model.coef_

model.fit
predictions=model.predict(X_test)
from sklearn.metrics import mean_squared_error

print('Mean Squared Error     ',mean_squared_error(y_test,predictions))
print('Root Mean Squared Error',np.sqrt(mean_squared_error(y_test,predictions)))

#D)	Logistic Regression:
# a.	Build a simple logistic regression model where dependent variable is ‘Churn’ & independent variable is ‘MonthlyCharges’
# i.	Divide the dataset in 65:35 ratio
# ii.	Build the model on train set and predict the values on test set
# iii.	Build the confusion matrix and get the accuracy score
X=pd.DataFrame(df['MonthlyCharges'])
y=df['Churn']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.65, random_state=42)


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(X_train,y_train)

predictions=model.predict(X_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,predictions))

churn_data=pd.get_dummies(df,columns=['Churn'],drop_first=True)

#Multiple Linear Regression
#b.	Build a multiple logistic regression model where dependent variable is ‘Churn’ & independent variables are ‘tenure’ & ‘MonthlyCharges’
#i.	Divide the dataset in 80:20 ratio
#ii.	Build the model on train set and predict the values on test set
#iii.	Build the confusion matrix and get the accuracy score


X=pd.DataFrame(df[['tenure','MonthlyCharges']])
y=df['Churn']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=42)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)

predictions=model.predict(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))

predictions


#E)	Decision Tree:
#a.	Build a decision tree model where dependent variable is ‘Churn’ & independent variable is ‘tenure’
#i.	Divide the dataset in 80:20 ratio
#ii.	Build the model on train set and predict the values on test set
#iii.	Build the confusion matrix and calculate the accuracy
X=pd.DataFrame(df['tenure'])
y=df['Churn']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=42)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))


X=pd.DataFrame(df['tenure'])
y=df['Churn']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=42)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))

#F)	Random Forest:
##a.	Build a Random Forest model where dependent variable is ‘Churn’ & independent variables are ‘tenure’ and ‘MonthlyCharges’
#i.	Divide the dataset in 70:30 ratio
#ii.	Build the model on train set and predict the values on test set
#iii.	Build the confusion matrix and calculate the accuracy
X=pd.DataFrame(df[['tenure','MonthlyCharges']])
y=df['Churn']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=42)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

