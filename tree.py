import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import doctest
from sklearn import datasets

from gini import best_split, split

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
                # print('Le noeud est pur')
                node.is_leaf = True
                return
            # 2) If the maximum depth of the tree is reached
            elif self.max_depth is not None and node.depth >= self.max_depth:
                # print('Profondeur max atteinte')
                node.is_leaf = True
                return 
            # 3) If the number of sample in the node is less than the minimum number of sample
            elif len(df) <= self.min_samples:
                # print('Echantillon insuffisant')
                node.is_leaf = True
                return
            # When the node is not a leaf
            else:
                # We split the node (column, value and minimal value of gini)
                col, val, gini = best_split(df)
                # print('resultat du split',col,val,gini)
                node.split_col = col
                node.split_value = val
                left, right = split(node.split_col, node.split_value, df)
                # If the split is not possible
                if len(left) == 0 or len(right) == 0:
                    # print('Le split est impossible')
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
    df_a_classer = df_shuffle.head(100)
    # We take the last 50 rows to test the model
    df_new = df_shuffle.tail(50)
    # Tree creation
    Tree = DecisionTree()
    Tree.fit(df_a_classer,'class')
    Tree.display_tree('class')
    # Prediction
    result = Tree.predict(df_new)
    # We add the prediction to the dataframe
    df_new.insert(df_new.shape[1],'Prediction',result)
    # We print the accuracy of the dataframe
    print(accuracy(df_new, 'class', 'Prediction'))

    # We compute the accuracy of the model with cross validation
    cross = cross_validation(df_classe,'class',10)
    print('cross validation', cross)
    print('Moyenne cross validation', np.mean(cross))
    print('Variance type cross validation', np.sqrt(np.std(cross)))