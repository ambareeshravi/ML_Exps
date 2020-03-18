'''
Author: Ambareesh Ravi
Date: 08-03-2020
Title: decision tree

Links and References:
    Datasets:
        -
        
    Papers:
        -
        
    Lectures, tutorials, articles and posts:
        - 
        
    Additional comments / notes:
        - 
'''

# Imports
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from time import time
from collections import OrderedDict
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)

class Node:
    '''
    Creates nodes for the ID3 tree
    '''
    def __init__(self, feature, isNum = False, threshold = None):
        '''
        Declares the class variables
        '''
        self.isNum = isNum # left lesser than threshold & right greater than threshold
        if self.isNum: self.threshold = threshold
        self.feature = feature
        self.children = dict()
        
class DecisionTreeClassifier:
    '''
    Creates the ID3, C4.5 decison tree classifiers
    '''
    def __init__(self, eval_type = 'GR', max_depth = 10, debug = False):
        '''
        Declare the class variables
        
        eval_type - type of metric used for decisions - Information Gain (IG), Gain Ratio (GR)
        max_depth - maximum depth the tree can have
        debug - print debug statements
        '''
        self.max_depth = max_depth
        if eval_type in ['IG', 'EN', 'Information Gain']:
            self.decider_function = self.calc_InformationGain
            self.decider_basis = np.argmax
        elif eval_type in ['GR', 'Gain Ratio']:
            self.decider_function = self.calc_GainRatio
            self.decider_basis = np.argmax
        else:
            print("INVALID evaluation type")
        self.debug = debug
    
    def check_DataNature(self):
        '''
        Automatically check data nature
        
        Args:
            -
        Returns:
            -
        Exception:
            -
        '''
        self.data_nature = "categorical"
        for column in self.data.columns:
            if len(np.unique(self.data[column])) > int(0.2 * len(self.data)):
                self.data_nature = "numerical"
                break
        if self.data_nature == "categorical":
            self.add_child = self.add_child_categorical
            self.get_class = self.get_class_categorical
        elif self.data_nature == "numerical":
            self.add_child = self.add_child_numerical
            self.get_class = self.get_class_numerical
            
    def __depth__(self, d):
        '''
        Checks the maximum depth of the tree
        
        Args:
            d - tree as type <Node>
        Returns:
            max_depth as <int>
        Exception:
            -
        '''
        if isinstance(d, Node): return (max(map(self.__depth__, d.children.values())) if d else 0) + 1
        return 0
    
    def __display_numerical_tree__(self, node, spacing=""):
        '''
        Displays the numerical tree on the screen
        
        Args:
            node - the tree as type <Node>
            spacing - separation type as <str>
        Returns:
            -
        Exception:
            -
        '''
        try:
            print(spacing + node.feature, "<", str(node.threshold))
            print (spacing + '>> True:')
            self.__display_numerical_tree__(node.children["left"], spacing + "  ")
            print (spacing + '>> False:')
            self.__display_numerical_tree__(node.children["right"], spacing + "  ")
        except Exception as e:
            return
    
    def calc_Entropy(self, X):
        '''
        Calculates the entropy of a feature
        
        Args:
            X - feature as <np.array>
        Returns:
            entropy value as <float>
        Exception:
            -
        '''
        entropy = 0
        X_len = len(X)
        for x in np.unique(X):
            prob = (np.count_nonzero(X == x) / X_len)
            entropy += (-prob * np.log2(prob))
        return entropy
    
    def calc_ConditionalEntropy(self, targets, features):
        '''
        Calculates the conditional entropy
        
        Args:
            targets - <np.array>
            features - <np.array>
        Returns:
            conditional entropy as <float>
        Exception:
            -
        '''
        cond_Entropy = 0
        features_length = len(features)
        for feature_type in np.unique(features):
            feature_args = np.argwhere(features == feature_type)
            features_count = len(feature_args)
            target_feature_args = targets[feature_args]
            cond_Entropy += (features_count/features_length * self.calc_Entropy(target_feature_args))
        return cond_Entropy
    
    def calc_InformationGain(self, target, feature):
        '''
        Calculates the information gain
        
        Args:
            targets - <np.array>
            features - <np.array>
        Returns:
            information gain as float
        Exception:
            -
        '''
        return (self.calc_Entropy(target) - self.calc_ConditionalEntropy(target, feature))
    
    def calc_GainRatio(self, target, feature):
        '''
        Calculates the Gain Ratio
        
        Args:
            targets - <np.array>
            features - <np.array>
        Returns:
            gain ratio as float
        Exception:
            -
        '''
        return self.calc_InformationGain(target, feature) / np.sum([-(np.count_nonzero(feature == unique_feature)/len(feature))*np.log2(np.count_nonzero(feature == unique_feature)/len(feature)) for unique_feature in np.unique(feature)])
    
    def get_split_points(self, features, method = "percentile", perc_skip = 3, round_to = 2, toRound =  True):
        '''
        decides the split point
        
        Args:
            features - <np.array>
            method - <str> 'percentile' to calculate the percentiles / 'mean' to take mean between values
        Returns:
            split points as a <list>
        Exception:
            -
        '''
        if toRound:
            features = np.unique(np.round(features, round_to)) # round values
        if method == "percentile" and len(features) > (100/perc_skip): # percentile - okay but faster
                return [np.percentile(features, i) for i in range(10,100, perc_skip)]
        return [np.mean([features[idx], features[idx+1]]) for idx in range(len(features[:-1]))] # mean approach - accuracte but expensive
    
    def add_child_categorical(self, data):
        
        '''
        builds tree for categorical type
        
        Args:
            data - <pd.DataFrame>
        Returns:
            built tree with nodes of types <Node>
        Exception:
            -
        '''
        if data.empty: return self.parent_node_class
                
        targets = data[self.target_name].values
        if len(np.unique(targets)) < 2: return np.squeeze(np.unique(targets)).tolist() # pure set
        max_target = np.unique(targets)[np.argmax([np.count_nonzero(targets == ut) for ut in np.squeeze(np.unique(targets))])]
        
        features = list(data.columns)
        features.remove(self.target_name)
        if len(features) < 2: return max_target # no more feature left
        
        # compute the feature to split on based on the information gain
        split_feature = features[self.decider_basis([self.decider_function(targets, data[feature].values) for feature in features])]
        
        node = Node(feature = split_feature, isNum = False, threshold = None)
        features.remove(split_feature)
        self.parent_node_class = max_target
        for unique_value in np.unique(np.array(data[split_feature])):
            new_data = data.where(data[split_feature]==unique_value).dropna()
            new_data = new_data[features + [self.target_name]] # create subset of data after removing the split feature
            node.children[unique_value] = self.add_child(new_data) # do recursion till leaf node
        return node
    
    def get_class_categorical(self, tree, data):
        '''
        Decides the class given the data
        
        Args:
            tree - with nodes as type <Node>            
            data - data as type <dict>
        Returns:
            class type
        Exception:
            -
        '''
        if not isinstance(tree, Node): return tree
        try: return self.get_class(tree.children[data[tree.feature]], data)
        except: return self.majority_target
    
    def add_child_numerical(self, data):
        '''
        Builds tree for numerical data
        
        Args:
            data - pd.<DataFrame>
        Returns:
            built tree with nodes of types <Node>
        Exception:
            -
        '''
        if data.empty: return self.parent_node_class
        
        targets = data[self.target_name].values
        unique_targets = np.unique(targets)
        if len(unique_targets) < 2: return np.squeeze(unique_targets).tolist() # pure set
        max_target = unique_targets[np.argmax([np.count_nonzero(targets == ut) for ut in np.squeeze(unique_targets)])]
        
        features = list(data.columns)
        features.remove(self.target_name)
        if len(features) < 2:
            return max_target # no more feature left
                        
        # find the best feature and point to split on
        best_split_feature, best_split_point, best_info_gain = None, None, -1
        
        for column in features:
            feature_values = data[column].values
            split_points = self.get_split_points(feature_values)
            
            for split_point in split_points:
                
                left_feature = feature_values[np.where(feature_values < split_point)]
                right_feature = feature_values[np.where(feature_values >= split_point)]
                
                left_target = targets[np.where(feature_values<split_point)]
                right_target = targets[np.where(feature_values>=split_point)]
                
                total_info_gain = sum([self.calc_InformationGain(left_target, left_feature), self.calc_InformationGain(right_target, right_feature)])/2
                
                if total_info_gain >= best_info_gain:
                    best_split_point = split_point
                    best_info_gain = total_info_gain
                    best_split_feature = column
                    
        node = Node(feature = best_split_feature, isNum = True, threshold = best_split_point)
        try:
            features.remove(best_split_feature)
        except:
            return self.parent_node_class
        self.parent_node_class = max_target
                
        left_data = data.where(data[best_split_feature] < best_split_point).dropna()
        left_data = left_data[features + [self.target_name]]
        node.children["left"] = self.add_child(left_data)
        
        right_data = data.where(data[best_split_feature] >= best_split_point).dropna()
        right_data = right_data[features + [self.target_name]]
        node.children["right"] = self.add_child(right_data)
        return node
        
    def get_class_numerical(self, tree, data):
        '''
        Decides the class given the data
        
        Args:
            tree - with nodes as type <Node>            
            data - data as type <dict>
        Returns:
            class type
        Exception:
            -
        '''
        if not isinstance(tree, Node): return tree
        next_node = "right"
        if data[tree.feature] < tree.threshold: next_node = "left"
        try: return self.get_class_numerical(tree.children[next_node], data)
        except:
            print("Coundn't find right target, returning majority label")
            return self.majority_target
    
    def prep_data(self, data, target_name):
        '''
        Prepares the data to build the ID3 tree
        
        Args:
            -
        Returns:
            -
        Exception:
            -
        '''
        self.node_count = 0
        self.target_name = target_name
        self.majority_target = np.unique(data[self.target_name].values)[np.argmax([np.count_nonzero(data[self.target_name].values == ut) for ut in np.unique(data[self.target_name].values)])]
        self.parent_node_class = self.majority_target
        self.data = data
        self.check_DataNature()
        self.feature_ids = dict([(f, idx) for idx, f in enumerate(list(self.data.columns))])
        self.targetless_features_list = list(self.feature_ids.keys())
        self.targetless_features_list.remove(self.target_name)
        
    def fit(self, data, target_name):
        
        '''
        Fits the data to a decision tree
        
        Args:
            data - <pd.DataFrame>
            target_name - label column name as <str>
        Returns:
            -
        Exception:
            -
        '''
        self.prep_data(data, target_name)
        self.tree = self.add_child(self.data)
        self.max_depth = self.__depth__(self.tree)
        if self.debug:
            print("Tree created with depth %d"%(self.__depth__(self.tree)))
    
    def predict(self, data):
        '''
        Predicts the label for test data
        
        Args:
            data - data as type <np.array>
        Returns:
            class type
        Exception:
            -
        '''
        if len(data) > (len(self.feature_ids) - 1):
            data = np.delete(data, self.feature_ids[self.target_name])
        data = dict([(f,d) for f,d in zip(self.targetless_features_list, data)])
        return self.get_class(self.tree, data)
    
    def predict_data(self, data):
        '''
        Predicts the label on a batch of test data
        
        Args:
            data - data as type <pd.DataFrame>
        Returns:
            <list> of predictions, accuracy score as <float>
        Exception:
            -
        '''
        y_pred = [self.predict(data_row) for data_row in data.values]
        return y_pred, accuracy_score(data[self.target_name].values, y_pred)        