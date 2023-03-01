import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import doctest
from sklearn import datasets

from tree import DecisionTree, accuracy

# Load the iris dataset
data = datasets.load_iris()
# print(data.keys())
# make pd.DataFrame with the Iris data
df = pd.DataFrame(data.data, columns = data.feature_names)
# we create a copy of the dataframe to add the class
df_classe = df.copy()
df_classe['class'] = data.target
# print(df_classe)

def random_selection(df, nb, y_col):
    # We shuffle the dataframe
    df_shuffle = df.sample(frac = 1)
    # We take the first 100 rows to train the model
    df_a_classer = df_shuffle.head(nb)
    # We take the last 50 rows to test the model
    df_new = df_shuffle.tail(df.shape[0]-nb)
    # Tree creation
    Tree = DecisionTree()
    Tree.fit(df_a_classer,y_col)
    Tree.display_tree(y_col)
    # Prediction
    result = Tree.predict(df_new)
    # We add the prediction to the dataframe
    df_new.insert(df_new.shape[1],'Prediction',result)
    # We print the accuracy of the dataframe
    print('Pourcentage of accuracy',accuracy(df_new, y_col, 'Prediction'))
    return df_new

def stratified_split(df, y_col, nb):
    df_class0 = df[df[y_col] == 0]
    df_class1 = df[df[y_col] == 1]
    df_class2 = df[df[y_col] == 2]
    # We shuffle the dataframe
    df_class0_s = (df_class0.sample(frac = 1))
    df_class1_s = (df_class1.sample(frac = 1))
    df_class2_s = (df_class2.sample(frac = 1))
    # Dataframes to train the model
    df_class0_a_classer = df_class0_s.head(nb)
    df_class1_a_classer = df_class1_s.head(nb)
    df_class2_a_classer = df_class2_s.head(nb)
    # Concatenation of the dataframes to train the model
    df_a_classer = pd.concat([df_class0_a_classer, df_class1_a_classer, df_class2_a_classer])
    # Dataframes to test the model
    df_class0_new = df_class0_s.tail(df_class0.shape[0]-nb)
    df_class1_new = df_class1_s.tail(df_class1.shape[0]-nb)
    df_class2_new = df_class2_s.tail(df_class2.shape[0]-nb)
    # Concatenation of the dataframes to test the model
    df_new = pd.concat([df_class0_new, df_class1_new, df_class2_new])
    # Tree creation
    Tree = DecisionTree()
    Tree.fit(df_a_classer,y_col)
    Tree.display_tree(y_col)
    # Prediction
    result = Tree.predict(df_new)
    # We add the prediction to the dataframe
    df_new.insert(df_new.shape[1],'Prediction',result)
    # We print the accuracy of the dataframe
    print('Pourcentage of accuracy',accuracy(df_new, y_col, 'Prediction'))
    return df_new


print('###################### Random selection among all observations ###################### ')
print('\n We train a tree with 100 observations and test it with the 50 others')
random_selection(df_classe, 100, 'class')
print('\n We train a tree with 50 observations and test it with the 100 others')
random_selection(df_classe, 50, 'class')
print('\n We train a tree with 25 observations and test it with the 125 others')
random_selection(df_classe, 25, 'class')


print('\n ######################  Stratified split among all observations ###################### ')
print('\n We train a tree with 30 elements of each class and test it with the others')
stratified_split(df_classe, 'class', 30)
print('\n We train a tree with 15 elements of each class and test it with the others')
stratified_split(df_classe, 'class', 15)
print('\n We train a tree with 5 elements of each class and test it with the others')
stratified_split(df_classe, 'class', 5)
