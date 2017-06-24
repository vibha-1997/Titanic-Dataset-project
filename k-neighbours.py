import numpy as np
from sklearn import preprocessing, cross_validation, neighbors,svm
from sklearn.linear_model import LogisticRegression
import pandas as pd

df = pd.read_csv('train (1).csv')
df2=pd.read_csv('test.csv')
df.drop(['Name'],1,inplace=True)
df2.drop(['Name'],1,inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)
df2.fillna(0,inplace=True)

def handle_non_numerical_data(df):
    columns=df.columns.values
    for column in columns:
        text_digit_vals={}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype!= np.int64 and df[column].dtype!= np.float64:
            column_contents=df[column].values.tolist()
            unique_elements=set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique]=x
                    x+=1

            df[column]=list(map(convert_to_int, df[column]))
    return df

df = handle_non_numerical_data(df)
#df.drop(['boat'],1,inplace=True)
print(df.head(5))

#df.drop(['id'], 1, inplace=True)
X = np.array(df.drop(['Survived'], 1).astype(float))
X=preprocessing.scale(X)

y = np.array(df['Survived'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#clf = svm.SVC()
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

df2=handle_non_numerical_data(df2)
prediction=clf.predict(np.array(df2))
print(prediction)

