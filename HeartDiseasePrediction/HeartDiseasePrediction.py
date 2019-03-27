# Heart attack for dummies
# https://www.kaggle.com/ronitf/heart-disease-uci
'''
Creators of the Dataset: 
1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D. 
2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D. 
3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D. 
4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

Donor: David W. Aha (aha '@' ics.uci.edu) (714) 856-8779
'''

# Well Here we go

# Reading the data
import pandas as pd
data = pd.read_csv("heart.csv")
# Well, it looks like the heart disease guys are grouped together and the fit guys are grouped together
# I need to see if train_test_split separates them

data.info
data.info()
# Oh wow, info() and info are not the same thing, my childhood was a lie

data.describe()
# No missing values - heart attack averted

data.dtypes.value_counts()
# 13 integer features (includes the target)
# 1 float feature called oldpeak

# Whats in the head??
data.head()

# Checking for missing values
data.isnull().sum() # Glad to see all zeroes
data.isnull().any()

'''
# Doing Exploratory (Obligatory) Data Analysis
# Univariate
import seaborn as sns
import matplotlib.pyplot as plt
# Target variable
# sns.countplot(data['target']) # This will also give the same graph, 
sns.countplot(x=data['target'].value_counts()) # but value_counts() will just count the value first - easy on memory

# Age
# plt.hist(data['age'])
sns.countplot(data['age']) # I'm starting to like seaborn more - me likey
# Most of the people are in their 60s, I dont intend to cross 50 anyway
# Age in bins
#data['age_bin'] = pd.cut(data.age,[29,30,35,40,45,50,55,60],labels=[30,35,40,45,50,55,60])

# Sex
plt.hist(data['sex'], bins=2) # 0 is female, 1 is male
sns.countplot(data['sex']) # This is better, 
# More data from males


'''

# No null values in any columns
allColumns = data.columns.values.tolist()
numColumns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
catColumns = [col for col in allColumns if col not in numColumns]
# In notebook, print both

# Checking for duplicate values
data[data.duplicated() == True]
# patient no 164 is duplicate
# Removing the duplicate record
data.drop_duplicates(inplace=True)
# Now checking
data[data.duplicated() == True]
# Now it returned none


# Exploratory Data Analysis
# Univariate analysis
import seaborn as sns
import matplotlib.pyplot as plt
# Target variable
# sns.countplot(data['target']) # This will also give the same graph, 
plt.style.use('ggplot')
sns.set_style('whitegrid')

# Target Variable
sns.countplot(x=data['target']) # but value_counts() will just count the value first - easy on memory

# Sex
sns.countplot(data['sex'])

# Age distribution
pd.DataFrame(data['age'].describe())
data['ageBin'] = pd.cut(x=data.age, bins=[0, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80], labels=[30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])
'''
(0, 30] -- 1
(30, 35] -- 6
etc
'''
# Here, a new col is created
# If the age is <30, the col value for that record will be 30
# Personal Note: This will be a good function to use in monte carlo simulation
print(pd.DataFrame(data['ageBin'].value_counts()))
# Visualizing the above result in sns
sns.countplot(data['ageBin'])

# Sex distribution
pd.DataFrame(data['sex'].value_counts())
sns.countplot(data['sex'])

# CP: Chest Pain type
pd.DataFrame(data['cp'].value_counts())
sns.countplot(data['cp'])
# I'm no doctor, but I guess it is nominal 4 valued categorical from the dataset documentation
# Value 1: typical angina -- Value 2: atypical angina -- Value 3: non-anginal pain -- Value 4: asymptomatic
# In dataset, value starts from 0

# Cholestrol
pd.DataFrame(data['chol'].describe())
# Min is 126(Vegan for sure), max is 564(Holy Crap, I would like his diet)
# Lets sort these into bins
'''
(125, 150],
(150, 200],
...,
(550, 600]
'''
print(range(125, 601, 50))
mylist = list(range(150, 601, 50))
mylist.append(125)
mylist.sort()
mylist
data['cholBin'] = pd.cut(data.chol, bins=mylist, labels=list(range(150, 601, 50)))
pd.DataFrame(data['cholBin'].value_counts())
sns.countplot(data['cholBin'])

# trestbps
# It is the resting blood pressure on admission at the hospital
# Numerical Value
data['trestbps'].describe() # This also works
# data.trestbps.describe() # This also works
# Min is 84 mm Hg, max is 200 mm Hg (The nurse who took his BP might be hot)
data['trestbpsBin'] = pd.cut(data.trestbps, bins=[93, 110, 120, 130, 140, 150, 160, 205], labels=[110, 120, 130, 140, 150, 160, 205])
data['trestbpsBin'].value_counts()
sns.countplot(data.trestbpsBin) # This also works
# sns.countplot(data['trestbpsBin'])

# FBS
# Will be 1 if the fasting blood sugar is higher than the normal 120 mg/dl
data.fbs.unique()
# Two values = 1 and 0
sns.countplot(data.fbs)

# restecg
# Resting ECG results
data.restecg.unique()
# We get three values
# restecg: resting electrocardiographic results -- Value 0: normal -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria 20
# What I understood is 0 is normal, 1 is pretty bad and 2 is fucked up
# Therefore ordinal categorical
sns.countplot(data.restecg)

# thalach
# Maximum heart rate achieved
data.thalach.unique()
# Integer numerical value
data.thalach.describe()
# min is 71
# max is 202
data['thalachBin'] = pd.cut(data.thalach, bins=[70, 90, 110, 130, 150, 170, 180, 200, 203], labels=[90, 110, 130, 150, 170, 180, 200, 203])
data.thalachBin.value_counts()
sns.countplot(data.thalachBin) # Is that a normal distributioin I see?

# exang
# Exercise included?
# 1 is Yes, 0 is No
sns.countplot(data.exang)

# oldpeak
# ST depression induced due to exercise relative to rest
# ST means something Thoracic - some chest measure as I remember from data's doc and paper
data.oldpeak.describe()
# min is 0, max is 6.2
# float value
# sns.countplot(data.oldpeak) # isnt working well, need to discretize this
data['oldpeakBin'] = pd.cut(data.oldpeak, 
    bins=[-0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 6.5],
    labels=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 6.5])
sns.countplot(data.oldpeakBin)

# Slope
# the slope of the peak exercise ST segment -- Value 1: upsloping -- Value 2: flat -- Value 3: downsloping
# I'm guessing ordinal categorical
sns.countplot(data.slope)

# ca
# number of major vessels (0-3) colored by flourosopy
data.ca.unique()
sns.countplot(data.ca)

# thal
# thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
data.thal.unique()
sns.countplot(data.thal)
# I cant find this in the data's doc, I dont know whether to take it as ordinal or nominal
# And thus the univariate analysis is complete

# Multivariate Analysis

# Age with respect to heart disease
target1 = data[data['target']==1]['ageBin'].value_counts()
target0 = data[data['target']==0]['ageBin'].value_counts()
temp = pd.DataFrame([target0, target1])
temp.index = ['Healthy', 'Disease']
temp.plot(kind='bar', stacked=True)

# Sex with respect to heart disease
target1 = data[data['target']==1]['sex'].value_counts()
target0 = data[data['target']==0]['sex'].value_counts()
tempDf = pd.DataFrame([target0, target1])
tempDf.index = ['Healthy', 'Disease']
tempDf.plot(kind='bar', stacked=True)

# Relationship between age and trestbps
data.plot(kind='scatter', x='age', y='trestbps', color='green', alpha=0.5)
# More people will have higher blood pressure as they age

# Relationship between age and maximum heartrate acheived
data.plot(kind='scatter', x='age', y='thalach', color='blue', alpha=0.5)
# As you age, the maximum heart rate you can achieve will gradually reduce

# Relationships between age, cholestrol, ca and oldpeak
sns.pairplot(data.loc[:, ['age', 'chol', 'ca', 'oldpeak']])

# Correlation Matrix
dataCorr = data.corr()['target'][:-1] # Last row is the target
# Now take the most correlated features
goldFeaturesList = dataCorr[abs(dataCorr) > 0.1].sort_values()
# So the strongly correlated features with the target are
goldFeaturesList

# Drawing the correlation matrix
corr = data.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(data=corr[abs(corr) > 0.1], vmin=-1, vmax=1, cmap='summer', annot=True, cbar=True, square=True)

# Modeling 
# Again importing data
data = pd.read_csv('heart.csv')
y = data['target']
X = data.drop(['target'], axis=1)
# Train-Test Split
from sklearn.model_selection import train_test_split
Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=0)
# Yep, train test split will randomize (obviously, what was I thinking)


# Evaluating the results
# First, lets make a function to evaluate the results of prediction
from sklearn.metrics import confusion_matrix
# Precision = tp/(tp+fp) ie out of all predicted positive, how many actually have heartDisease
from sklearn.metrics import precision_score
# Recall = tp/(tp+fn) ie out of all heart disease patients, how many are detected by our MLalgo
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc

def evaluateModel(yTrue, yPredicted, modelName):
    print("=====================================================")
    print("Result of prediction for the model - ", modelName)
    confMatrix = confusion_matrix(yTrue, yPredicted)
    print("Confusion Matrix")
    print(confMatrix)
    precision = round(precision_score(yTrue, yPredicted), 4)
    print("Precision is ", precision)
    print("Out of all predicted as Heart Patients, {} percent actually have Heart Disease".format(precision*100))
    recall = round(recall_score(yTrue, yPredicted), 4)
    print("Recall is ", recall)
    print("Out of all actual heart patients, {0} is able to detect {1} percent of them".format(modelName, recall*100))
    print("Drawing the ROC")
    fpr, tpr, thresholds = roc_curve(yPredicted, yTrue)
    roc_auc = round(auc(fpr, tpr), 3) # I only need three decimal places
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=1, label="{0}, area={1}".format(modelName, roc_auc))
    plt.plot([0, 1], [0, 1], color='blue', lw=1, linestyle='--') # Apparently this is line between (0,0) and (1,1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Flase Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic for {}".format(modelName))
    plt.legend(loc="lower right")
    plt.show()
    print("=====================================================")
    
# Modeling    
# Traditional Models

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logisticRegression = LogisticRegression(random_state=0)
logisticRegression.fit(Xtrain, yTrain)
yPredLogReg = logisticRegression.predict(Xtest)
# Evaluating Logistic Regression
evaluateModel(yTest, yPredLogReg, "Logistic Regression")

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
naiveBayes = GaussianNB(priors=None)
naiveBayes.fit(Xtrain, yTrain)
yPredNaiveBayes = naiveBayes.predict(Xtest)
# Evaluating Naive Bayes
evaluateModel(yTest, yPredNaiveBayes, "Naive Bayes")

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
decTree = DecisionTreeClassifier(criterion='entropy', random_state=0)
decTree.fit(Xtrain, yTrain)
yPredDecTree = decTree.predict(Xtest)
# Evaluating Decision Tree
evaluateModel(yTest, yPredDecTree, "Decision Tree")


# Using Ensemble Methods
# Now let us use Bagging Methods
from sklearn.ensemble import RandomForestClassifier
randForest = RandomForestClassifier(criterion='entropy', random_state=0)
randForest.fit(Xtrain, yTrain)
yPredRandForest = randForest.predict(Xtest)
# Evaluating Random Forest
evaluateModel(yTest, yPredRandForest, "Random Forest")






