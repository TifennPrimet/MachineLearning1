import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import doctest

from sklearn import datasets


def split_value(df, key):
    """"
    function wich create a vector of threshold possible for a feature
    df: dataframe
    key: column name
    return: vector of threshold
    """

    # we extract the column
    data = df[key]
    # we sort the unique value of the column
    data = np.sort(data.unique())
    # we create a vector of threshold equal to the mean of two consecutive values
    result = np.zeros(len(data)-1)
    for i in range(len(data)-1):
        result[i] = (data[i] + data[i+1])/2
    return result



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


def gini_impurity(df):
    """
    df: dataframe
    nb: number of threshold
    return: gini impurity for each feature
    """
    data = df.drop(['class'], axis=1)
    resultGini = []
    numKey = 0
    for key in data.keys():
        valeurs = split_value(df, key)
        numSeuil = 0
        resultGiniInter = []
        for seuil in valeurs:
            left, right = split(key, seuil, df)
            resultGiniInter.append(gini_index(left, right))
            numSeuil += 1
        resultGini.append(resultGiniInter)
        numKey += 1
    return resultGini

def best_split_for_all(df):
    """
    df: dataframe
    nb: number of threshold
    return: best split for each feature
    """
    data = df.drop(['class'], axis=1)
    resultGini = gini_impurity(df)
    result = np.zeros((len(data.keys()),2))
    numKey = 0
    for key in data.keys():
        minIndex = np.argmin(resultGini[numKey])
        result[numKey,0] = split_value(df, key)[minIndex]
        result[numKey,1] = resultGini[numKey][minIndex]
        numKey += 1
    return result

def best_split(df):
    result = best_split_for_all(df)
    scoreGini = result[:,1]
    value = result[:,0]
    minIndex = np.argmin(scoreGini)
    return np.array([df.keys()[minIndex],value[minIndex],scoreGini[minIndex]])


class TreeNode:
    def __init__(self):
        """
        proba (list or array or dataframe): list of all the frequencies encounter
        main_class: class the most represent with the frequencies
        depth: depth in the tree
        is_leaf: if the node is terminal or not
        split_col: name of the column used to split
        split_value: value of the split

        left: child node
        right: child node
        """
        self.proba = None
        self.main_class = None
        self.depth = None
        self.is_leaf = False
        self.split_col = None
        self.split_value = None
        
        self.left = None
        self.right = None


class DecisionTree:
    def __init__(self, max_depth=None, min_samples=1):
        self.tree = TreeNode()
        self.max_depth = max_depth
        self.min_samples = min_samples 
    
    def extend_node(self, node, df, y_col):
        pass

    def fit(self, df, y_col):
        pass


    def predict(self, new_df):
        pass
      


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
    print(split_value(dfClasse, 'petal width (cm)'))
    print(best_split_for_all(dfClasse))
    print(best_split(dfClasse))

