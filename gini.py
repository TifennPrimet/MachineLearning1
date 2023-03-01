import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import doctest
from sklearn import datasets

# =============================================================================
# Functions to split the data
# =============================================================================
def split_value(df, key):
    """"
    function wich create a vector of threshold possible for a feature
    :param df: dataframe
    :type df: pandas.DataFrame
    :param key: column name
    :type key: str
    :return: vector of threshold
    """
    # we extract the column
    data = df[key]
    # we sort the unique value of the column
    data = np.sort(data.unique())
    if data.shape[0] == 1:
        return [data[0]]
    else:
        # we create a vector of threshold equal to the mean of two consecutive values
        result = np.zeros(len(data)-1)
        for i in range(len(data)-1):
            result[i] = (data[i] + data[i+1])/2
        return result


def split(key, value, df):
    """ 
    :param key: column name
    :type key: str
    :param value: value to split the data
    :type value: float
    :param df: dataframe
    :type df: pandas.DataFrame
    :return: left and right dataframe
    >>> len(split('sepal width (cm)', 3.0, df)[0])
    57
    """
    left = df[df[key] <= value]
    right = df[df[key] > value]
    return left, right


# =============================================================================
# Functions to compute the gini coefficient
# =============================================================================
def gini_group(y):
    """
    Function which calculates the gini impurity of a leaf
    :param y: dataframe
    :type y: pandas.DataFrame
    :return: gini coefficient 
    """
    # Computation of the frequency of the different class
    df_frequency = y['class'].value_counts(normalize = True)
    # Formula for Gini
    gini = 1 - sum(df_frequency ** 2)
    return gini


def gini_index(left, right):
    """
    Function which calculates the gini impurity of a split
    :param left: left dataframe
    :type left: pandas.DataFrame
    :param right: right dataframe
    :type right: pandas.DataFrame
    :return: weighted average of gini 
    """
    total = len(left) + len(right)
    frequency_left = len(left)/total
    frequency_right = len(right)/total
    gini = frequency_left * gini_group(left) + frequency_right * gini_group(right)
    return gini


def gini_impurity(df):
    """
    :param df: dataframe
    :type df: pandas.DataFrame
    :param nb: number of threshold
    :type nb: int
    :return: gini impurity for each feature
    """
    data = df.drop(['class'], axis=1)
    result_gini = []
    for key in data.keys():
        valeurs = split_value(df, key)
        result_gini_inter = []
        for seuil in valeurs:
            left, right = split(key, seuil, df)
            result_gini_inter.append(gini_index(left, right))
        result_gini.append(result_gini_inter)
    return result_gini


# =============================================================================
# Functions to find the best split
# =============================================================================
def best_split_for_all(df):
    """
    :param df: dataframe
    :type df: pandas.DataFrame
    :param nb: number of threshold
    :type nb: int
    :return: best split for each feature
    """
    data = df.drop(['class'], axis=1)
    result_gini = gini_impurity(df)
    result = np.zeros((len(data.keys()),2))
    number_key = 0
    for key in data.keys():
        minimum_index = np.argmin(result_gini[number_key])
        result[number_key,0] = split_value(df, key)[minimum_index]
        result[number_key,1] = result_gini[number_key][minimum_index]
        number_key += 1
    return result


def best_split(df):
    """
    :param df: dataframe
    :type df: pandas.DataFrame
    :return: best split for the dataframe
    """
    result = best_split_for_all(df)
    score_gini = result[:,1]
    value = result[:,0]
    minimum_index = np.argmin(score_gini)
    return (df.keys()[minimum_index],value[minimum_index],score_gini[minimum_index])

if __name__ == "__main__":
    # doctest.testmod()
    # Load the iris dataset
    data = datasets.load_iris()
    # make pd.DataFrame with the Iris data
    df = pd.DataFrame(data.data, columns = data.feature_names)
    # we create a copy of the dataframe to add the class
    df_classe = df.copy()
    df_classe['class'] = data.target
    print(df_classe.head(5))
    
    #visual 2D display of the petal lenght according to the sepal width
    plt.figure()
    x = df['petal width (cm)']
    y = df['petal length (cm)']
    plt.scatter(x, y, c = data.target)
    plt.xlabel('petal width (cm)')
    plt.ylabel('petal length (cm)')
    plt.title('Iris dataset')
    plt.show()

    # We split the data
    left, right = split("petal width (cm)", 2, df_classe)
    print(f"Length Left: {len(left)}")
    print(f"Length Right: {len(right)}")
    print('coeff',gini_index(left,right))
    # Findng the best split
    print('Best split according to each column \n', best_split_for_all(df_classe))
    res_best_split = best_split(df_classe)
    print('Best split overall', res_best_split)
    left, right = split(res_best_split[0], res_best_split[1], df_classe)
    print(f"Length Left: {len(left)}")
    print(f"Length Right: {len(right)}")
    print('coeff',gini_index(left,right))