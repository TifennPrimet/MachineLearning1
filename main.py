import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import doctest
import seaborn as sns
from sklearn import datasets

from tree import DecisionTree
from evaluation import accuracy, cross_validation, confusion_matrix

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

def rate(df,seuil):
    proportion = int(np.floor(df.shape[0]*seuil))
    df_a_classer = df.head(proportion)
    df_new = df.tail(df.shape[0]-proportion)
    Tree = DecisionTree()
    Tree.fit(df_a_classer,'class')
    # Prediction
    result = Tree.predict(df_new)
    # We add the prediction to the dataframe
    df_new.insert(df_new.shape[1],'Prediction',result)
    # Matrix confusion
    matrix = confusion_matrix(df_new, 'class', 'Prediction')
    true = np.trace(matrix)
    false = np.sum(matrix) - true
    return [true,false,matrix]


# print('###################### Random selection among all observations ###################### ')
# print('\n We train a tree with 100 observations and test it with the 50 others')
# random_selection(df_classe, 100, 'class')
# print('\n We train a tree with 50 observations and test it with the 100 others')
# random_selection(df_classe, 50, 'class')
# print('\n We train a tree with 25 observations and test it with the 125 others')
# random_selection(df_classe, 25, 'class')

# print('\n ######################  Stratified split among all observations ###################### ')
# print('\n We train a tree with 30 elements of each class and test it with the others')
# stratified_split(df_classe, 'class', 30)
# print('\n We train a tree with 15 elements of each class and test it with the others')
# stratified_split(df_classe, 'class', 15)
# print('\n We train a tree with 5 elements of each class and test it with the others')
# stratified_split(df_classe, 'class', 5)

print('\n ###################### Confusion matrix ###################### ')
v_rate = np.linspace(0.1,0.9,9)
v_true = np.zeros(len(v_rate))
v_false = np.zeros(len(v_rate))
lst_matrix = [0]*len(v_rate)
for i in range(len(v_rate)):
    v_true[i], v_false[i], lst_matrix[i] = rate(df_shuffle, v_rate[i])

fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle('Evolution of the rate of true and false predictions')
ax1.plot(v_rate, v_true)
ax1.set_title('True predictions')
ax1.set_xlabel('Percentage of data trained')
ax1.set_ylabel('Number of elements')
ax2.plot(v_rate, v_false)
ax2.set_title('False predictions')
ax2.set_xlabel('Percentage of data trained')
ax2.set_ylabel('Number of elements')

figs, axs = plt.subplots(2,int(np.ceil(len(v_rate)/2)))
j = 0
l = 0
for k in range(len(v_rate)):
    if k == np.ceil(len(v_rate)/2):
        j += 1
        l = 0
    sns.heatmap(lst_matrix[k], ax=axs[j,l]).set(title= str(v_rate[k]*100)[:3] + '% of data trained')
    l+=1
figs.suptitle('Confusion matrices for different fraction of data trained')
