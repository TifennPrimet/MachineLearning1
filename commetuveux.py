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
    left = df[df[key] <= value]
    right = df[df[key] > value]
    return left, right


def gini_group(y):
    """
    Function which calculates the gini impurity of a leaf
    y: dataframe
    return: gini coefficient 
    """
    # Computation of the frequency of the different class
    dfFrequency = y['class'].value_counts(normalize = True)
    # Formula for Gini
    gini = 1 - sum(dfFrequency ** 2)
    return gini

def gini_index(left, right):
    """
    Function which calculates the gini impurity of a split
    left: left dataframe
    right: right dataframe
    return: weighted average of gini 
    """
    total = len(left) + len(right)
    frLeft = len(left)/total
    frRight = len(right)/total
    gini = frLeft * gini_group(left) + frRight * gini_group(right)
    return gini


def gini_impurity(df, nb):
    """
    df: dataframe
    nb: number of threshold
    return: gini impurity for each feature
    """
    data = df.drop(['class'], axis=1)
    resultGini = np.zeros((len(data.keys()), nb))
    numKey = 0
    for key in data.keys():
        valeurs = np.linspace(min(data[key]), max(data[key]), nb)
        numSeuil = 0
        for seuil in valeurs:
            left, right = split(key, seuil, df)
            resultGini[numKey, numSeuil] =  gini_index(left, right)
            numSeuil += 1
        numKey += 1
    return resultGini

def best_split(df, nb = 50):
    """
    df: dataframe
    nb: number of threshold
    return: best split for each feature
    """
    data = df.drop(['class'], axis=1)
    resultGini = gini_impurity(df, nb)
    result = np.zeros((len(data.keys()),2))
    numKey = 0
    for key in data.keys():
        minIndex = np.argmin(resultGini[numKey, :])
        result[numKey,0] = np.linspace(min(df[key]), max(df[key]), nb)[minIndex]
        result[numKey,1] = resultGini[numKey,minIndex]
        numKey += 1
    return result


if __name__ == "__main__":
    # doctest.testmod()
    # Load the iris dataset
    data = datasets.load_iris()
    # print(data.keys())
    # make pd.DataFrame with the Iris data
    df = pd.DataFrame(data.data, columns = data.feature_names)
    # we create a copy of the dataframe to add the class
    dfClasse = df.copy()
    dfClasse['class'] = data.target
    # print(dfClasse)

    #visual 2D display of the petal lenght according to the sepal width
    plt.figure()
    plt.scatter(df['sepal width (cm)'], df['petal length (cm)'], c = data.target)
    plt.xlabel('sepal width (cm)')
    plt.ylabel('petal length (cm)')
    plt.title('Iris dataset')
    # plt.show()


    print('---Martin---')
    left, right = split("petal width (cm)", 2, dfClasse)
    print(f"Left: {len(left)}")
    print(f"Right: {len(right)}")
    print('coeff',gini_index(left,right))
    # print(gini_impurity(df, 10))
    print(best_split(dfClasse))
