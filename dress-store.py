#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#READING AND CLEANING DATA:

df=pd.read_csv("shop_data.csv")
print(df)
#we dont need these field cause they are not int/float (linear regression trains int/float values)
df.drop(["Email","Address","Avatar"],axis=1,inplace=True) #axis 1 because we are removing column, inplace to make it reflect in .csv file also
print(df.info()) #To view all the available columns

#VISUALIZING DATA:

#visualazing data as a scatterplot to understand the data
#(X AXIS content, Y AXIS content, Data Reference, Opacity of Common Dots)
'''sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=df,joint_kws={"alpha": 0.5})  
plt.show()
#NOT SATISFACTORY, SO WE TEST ANOTHER ONE.
sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=df,joint_kws={"alpha": 0.5})  
plt.show()
#NOT SATISFACTORY, SO WE TEST ANOTHER ONE.
sns.jointplot(x="Length of Membership", y="Yearly Amount Spent", data=df,joint_kws={"alpha": 0.5})  
plt.show()'''
#BETTER

#Another way of analyzing is my plotting a PAIRPLOT: it shows scatterplot of every element all at once
sns.pairplot(df, kind="scatter",plot_kws={"alpha": 0.5})
plt.show()  #We found that Length of Membership is best


#MODEL TRAINING AND TESTING  -X is capital,y is small 
X=df[["Avg. Session Length","Time on App","Time on Website","Length of Membership"]] #data that will be used to train
y=df["Yearly Amount Spent"] #data that will be used to check predictions\test

#to create train and test variables for model to learn - (traindata,testdata,portion of data used for test i.e, 70% is for train and 30% is for test,

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)