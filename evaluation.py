import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import doctest
import seaborn as sns
from sklearn import datasets
from  matplotlib.colors import LinearSegmentedColormap
color=LinearSegmentedColormap.from_list('rg',["lightcoral", "white", "palegreen"], N=256) 

from tree import DecisionTree

def accuracy(df,y_class,y_pred):
    res = df[y_class] == df[y_pred]
    return sum(res)/res.shape[0]*100

def cross_validation(df,y_col,k):
    """
    :param df : dataframe to split
    :type df: pandas.DataFrame
    :param y_col :  name of the column to predict (column class)
    :type y_col: str
    :param k : number of folds
    :type k: int
    :return: accuracy of the model
    :rtype: float
    """
    # We shuffle the dataframe
    df_shuffle = df.sample(frac = 1)
    # We split the dataframe in k folds
    df_split = np.array_split(df_shuffle,k)
    # We create a list to stock the accuracy of each fold
    accuracy_list = []
    # We create a DecisionTree
    Tree = DecisionTree()
    # We iterate on the folds
    for i in range(k):
        # We take the fold i to test the model
        df_new = df_split[i].copy()
        # We take the other folds to train the model
        df_a_classer = pd.concat(df_split[:i]+df_split[i+1:])
        # We fit the model
        Tree.fit(df_a_classer,y_col)
        # We predict the class of the fold i
        df_new['prediction'] = Tree.predict(df_new)
        # We compute the accuracy of the model
        accuracy_list.append(accuracy(df_new,y_col,'prediction'))
    # We return the mean of the accuracy of each fold
    return accuracy_list

def confusion_matrix(df,y_class,y_pred):
    """
    :param df : dataframe to split
    :type df: pandas.DataFrame
    :param y_class : name of the column to predict (column class)
    :type y_class: str
    :param y_pred : name of the column predicted (column prediction)
    :type y_pred: str
    :return: confusion matrix
    :rtype: pandas.DataFrame
    """
    classe = df[y_class]
    pred = df[y_pred]
    nb_class = 3
    matrix = np.zeros((nb_class,nb_class))
    for i in range(len(classe)):
        matrix[classe.iloc[i]][int(pred.iloc[i])] += 1
    return matrix


if __name__ == "__main__":
    # doctest.testmod()
    # Load the iris dataset
    data = datasets.load_iris()
    # print(data.keys())
    # make pd.DataFrame with the Iris data
    df = pd.DataFrame(data.data, columns = data.feature_names)
    # we create a copy of the dataframe to add the class
    df_classe = df.copy()
    df_classe['class'] = data.target
    # print(df_classe)

    # We shuffle the dataframe
    df_shuffle = df_classe.sample(frac = 1)
    # We take the first 100 rows to train the model
    nb_to_train = 100
    df_a_classer = df_shuffle.head(nb_to_train)
    # We take the last 50 rows to test the model
    df_new = df_shuffle.tail(len(df_classe)-nb_to_train)
    # Tree creation
    Tree = DecisionTree()
    Tree.fit(df_a_classer,'class')
    # Prediction
    result = Tree.predict(df_new)
    # We add the prediction to the dataframe
    df_new.insert(df_new.shape[1],'Prediction',result)
    # We print the accuracy of the prediction
    print('Accuracy of the prediction : ',accuracy(df_new, 'class', 'Prediction'))

    # We compute the accuracy of the model with cross validation
    cross = cross_validation(df_classe,'class',10)
    print('Cross validation : ', cross)
    print('Moyenne cross validation : ', np.mean(cross))
    print('Variance type cross validation : ', np.sqrt(np.std(cross)))

    # We compute the confusion matrix
    matrix = confusion_matrix(df_new,'class','Prediction')
    print('The confusion matrix is given by \n',matrix)
    print(np.trace(matrix), ' elements over ', np.sum(matrix), 'were predicted correctly')
    
    sns.heatmap(matrix, annot=True)
    