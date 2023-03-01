import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import doctest
from sklearn import datasets


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
        # print('values',valeurs)
        result_gini_inter = []
        for seuil in valeurs:
            left, right = split(key, seuil, df)
            result_gini_inter.append(gini_index(left, right))
            # print('gini',gini_index(left, right))
        result_gini.append(result_gini_inter)
    return result_gini


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
    def __init__(self, max_depth = None, min_samples = 1):
        """
        tree : root of the tree
        :type tree: TreeNode
        max_depth : maximum depth of the tree
        :type max_depth: int
        min_samples : minimum number of samples in a leaf
        :type min_samples: int
        """
        self.tree = TreeNode()
        self.max_depth = max_depth
        self.min_samples = min_samples 
    
    def extend_node(self, node, df, y_col):
        """
        :param node : node to extend
        :type node: TreeNode
        :param df : dataframe to split
        :type df: pandas.DataFrame
        :param y_col : name of the column to predict (column class)
        :type y_col: str
        """
    def extend_node(self, node, df, y_col):
        """
        Recursive function to create the tree by extending one node
        node (TreeNode): node to extend
        df (dataframe): dataframe to split
        y_col (string): name of the column to predict (column class)
        """
        # Base case 
        # We stop the algorithm when all nodes are leaves
        if node.is_leaf: 
            return # we reached the end of the branch
        else:
            # We compute the frequency of all class in the node
            node.proba = df['class'].value_counts(normalize = True)
            # We determine the main class
            node.main_class = node.proba.idxmax()
            # We have several case where the node is a leaf
            # 1) If there is only one class is the node (purety is reached)
            if node.proba.shape[0] == 1:
                print('Le noeud est pur')
                node.is_leaf = True
                return
            # 2) If the maximum depth of the tree is reached
            elif self.max_depth is not None and node.depth >= self.max_depth:
                print('Profondeur max atteinte')
                node.is_leaf = True
                return 
            # 3) If the number of sample in the node is less than the minimum number of sample
            elif len(df) <= self.min_samples:
                print('Echantillon insuffisant')
                node.is_leaf = True
                return
            # When the node is not a leaf
            else:
                # We split the node (column, value and minimal value of gini)
                col, val, gini = best_split(df)
                print('resultat du split',col,val,gini)
                node.split_col = col
                node.split_value = val
                left, right = split(node.split_col, node.split_value, df)
                # If the split is not possible
                if len(left) == 0 or len(right) == 0:
                    print('Le split est impossible')
                    node.is_leaf = True
                    return
                # If the split is possible
                else:
                    # We create the left child
                    node.left = TreeNode()
                    node.left.depth = node.depth + 1
                    # We create the right child
                    node.right = TreeNode()
                    node.right.depth = node.depth + 1
                    # We extend the left child
                    self.extend_node(node.left, left, y_col)
                    # We extend the right child
                    self.extend_node(node.right, right, y_col)
                    return

    def fit(self, df, y_col):
        """
        :param df : dataframe to split
        :type df: pandas.DataFrame§
        :param y_col :  name of the column to predict (column class)
        :type y_col: str
        """
        self.tree.depth = 0
        self.extend_node(self.tree, df, y_col)

    def predict(self, new_df):
        """
        :param new_df : dataframe to predict
        :type new_df: pandas.DataFrame
        :return: prediction for the dataframe
        :rtype: array
        """
        result = np.zeros(len(new_df))
        for i in range(len(new_df)):
            node = self.tree
            while not node.is_leaf:
                if new_df[node.split_col].iloc[i] < node.split_value:
                    node = node.left
                else:
                    node = node.right
            result[i] = node.main_class
        return result

    def display_tree(self, y_col) :
        """
        Function which prints the tree
        : param y_col : name of the column to predict (column class)
        : type y_col: str
        """
        def display_node(node, y_col) :
            """
            Recursive function to print the tree
            : param node : node to print
            : type node: TreeNode
            : param y_col : name of the column to predict (column class)
            : type y_col: str
            """
            if node.is_leaf:
                return [f"  {node.main_class}  "]
            
            cut1 = f"{node.split_col}"
            len_cut_1 = len(cut1)
            cut2 = f"{node.split_value}"
            len_cut_2 = len(cut2)
            left = display_node(node.left, y_col)
            len_left = len(left[0])
            right = display_node(node.right, y_col)
            len_right = len(right[0])
            max_len_left = max(len_cut_1//2, len_cut_2//2, len_left)
            max_len_right = max((len_cut_1-1)//2, (len_cut_2-1)//2, len_right)
            tab1 = ["-"*(max_len_left-len_cut_1//2)+cut1+"-"*(max_len_right-(len_cut_1-1)//2), " "*(max_len_left-len_cut_2//2)+cut2+" "*(max_len_right-(len_cut_2-1)//2)]
            tab2 = [" "*(max_len_left-len_left)+(left[i] if i < len(left) else " "*len_left)+"|"+(right[i] if i < len(right) else " "*len_right)+" "*(max_len_right-len_right) for i in range(max(len(left),len(right)))]
            return tab1 + tab2 
        
        print("\n".join([f"|{line}|" for line in display_node(self.tree, y_col)]))     

def accuracy(df,y_class,y_pred):
    res = df_new[y_class] == df_new[y_pred]
    return sum(res)/res.shape[0]*100

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
    
    df_shuffle = df_classe.sample(frac = 1)

    #visual 2D display of the petal lenght according to the sepal width
    plt.figure()
    x = df['petal width (cm)']
    y = df['petal length (cm)']
    plt.scatter(x, y, c = data.target)
    # plt.plot(x,2.45*np.ones(len(x))) # Premier split
    # plt.plot(1.75*np.ones(len(y)),y) # Deuxième split
    # plt.plot(x,4.85*np.ones(len(x))) # Troisième split
    # plt.plot(x,4.95*np.ones(len(x))) # Quatrième split
    # plt.plot(1.55*np.ones(len(y)),y) # Cinquième split
    # plt.plot()
    plt.xlabel('petal width (cm)')
    plt.ylabel('petal length (cm)')
    plt.title('Iris dataset')
    plt.show()


    print('---Martin---')
    left, right = split("petal width (cm)", 2, df_classe)
    print(f"Left: {len(left)}")
    print(f"Right: {len(right)}")
    print('coeff',gini_index(left,right))
    # print(gini_impurity(df, 10))
    # print(split_value(df_classe, 'petal width (cm)'))
    print(best_split_for_all(df_classe))
    print(best_split(df_classe))
    print('---------------------------------------------------------')
    # Tree1 = DecisionTree()
    # Tree1.fit(df_classe, 'class')
    
    print('------------- Le temps du melange --------------')
    df_a_classer = df_shuffle.head(100)
    df_new = df_shuffle.tail(50)
    Tree = DecisionTree()
    Tree.fit(df_a_classer,'class')
    result = Tree.predict(df_new)
    df_new.insert(df_new.shape[1],'Prediction',result)
    Tree.display_tree('class')
    
    print(accuracy(df_new, 'class', 'Prediction'))

