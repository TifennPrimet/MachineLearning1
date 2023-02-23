import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import doctest

from sklearn import datasets

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

def gini_coef(left, right):
    """
    left: left dataframe
    right: right dataframe
    return: gini coefficient
    >>> gini_coef(split('sepal width (cm)', 3.0, df)[0], split('sepal width (cm)', 3.0, df)[1])
    0.47119999999999995
    """
    return 1-((len(left)/len(left+right))**2 + (len(right)/len(left+right))**2)

def gini_impurity(df, nb):
    """
    df: dataframe
    nb: number of threshold
    return: gini impurity for each feature
    """
    resultGini = np.zeros((len(df.keys()), nb))
    numKey = 0
    for key in df.keys():
        valeurs = np.linspace(min(df[key]), max(df[key]), nb)
        numSeuil = 0
        for seuil in valeurs:
            
            left, right = split(key, seuil, df)
            resultGini[numKey, numSeuil] =  gini_coef(left, right)
            numSeuil += 1
        numKey += 1
    return resultGini

def best_split(df, nb = 10):
    """
    df: dataframe
    nb: number of threshold
    return: best split for each feature
    """
    resultGini = gini_impurity(df, nb)
    result = np.zeros(len(df.keys()))
    numKey = 0
    for key in df.keys():
        minIndex = np.argmin(resultGini[numKey, :])
        result[numKey] = np.linspace(min(df[key]), max(df[key]), nb)[minIndex]
        numKey += 1
    return result


if __name__ == "__main__":
    doctest.testmod()
    # Load the iris dataset
    data = datasets.load_iris()
    print(data.keys())
    # make pd.DataFrame with the Iris data
    df = pd.DataFrame(data.data, columns = data.feature_names)
    # we create a copy of the dataframe to add the class
    dfClasse = df.copy()
    dfClasse['class'] = data.target
    print(dfClasse)

    # #visual 2D display of the petal lenght according to the sepal width
    # plt.figure()
    # plt.scatter(df['sepal width (cm)'], df['petal length (cm)'], c = data.target)
    # plt.xlabel('sepal width (cm)')
    # plt.ylabel('petal length (cm)')
    # plt.title('Iris dataset')
    # plt.show()
    print(gini_impurity(df, 10))
    print(best_split(df, 10))