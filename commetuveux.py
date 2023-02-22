import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import doctest

from sklearn import datasets

# Load the iris dataset
data = datasets.load_iris()
print(data.keys())
# make pd.DataFrame with the Iris data
df = pd.DataFrame(data.data, columns = data.feature_names)
df['class'] = data.target
print(df)

#visual 2D display of the petal lenght according to the sepal width
plt.figure()
plt.scatter(df['sepal width (cm)'], df['petal length (cm)'], c = data.target)
plt.xlabel('sepal width (cm)')
plt.ylabel('petal length (cm)')
plt.title('Iris dataset')
plt.show()

# creation of the ginni impurity for the decision tree
def split(key, value, df):
    """ 
    key: column name
    value: value to split the data
    df: dataframe
    return: left and right dataframe
    >>> len(split('sepal width (cm)', 3.0, df)[0])
    57
 
    """
    left = df[df[key] < value]
    right = df[df[key] >= value]
    return left, right


doctest.testmod()