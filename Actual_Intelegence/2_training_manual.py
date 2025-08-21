import pandas as pd
import numpy as np
import pickle

class ManualDecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = {}
    
    def calculate_gini(self, y):
        if len(y) == 0:
            return 0
        p1 = np.sum(y) / len(y)
        p0 = 1 - p1
        return 1 - (p0**2 + p1**2)
    
    def find_best_split(self, X, y):
        best_gini = float('inf')
        best_feature = None
        best_value = None
        
        for feature in X.columns:
            values = np.unique(X[feature])
            for value in values:
                left_mask = X[feature] <= value
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                gini_left = self.calculate_gini(y[left_mask])
                gini_right = self.calculate_gini(y[right_mask])
                
                total_gini = (np.sum(left_mask) * gini_left + 
                             np.sum(right_mask) * gini_right) / len(y)
                
                if total_gini < best_gini:
                    best_gini = total_gini
                    best_feature = feature
                    best_value = value
        
        return best_feature, best_value, best_gini
    
    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.mean(y)  # Return probability
        
        feature, value, gini = self.find_best_split(X, y)
        
        if feature is None:
            return np.mean(y)
        
        left_mask = X[feature] <= value
        right_mask = ~left_mask
        
        node = {
            'feature': feature,
            'value': value,
            'left': self.build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self.build_tree(X[right_mask], y[right_mask], depth + 1)
        }
        
        return node
    
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)
    
    def predict_single(self, x, node):
        if isinstance(node, dict):
            if x[node['feature']] <= node['value']:
                return self.predict_single(x, node['left'])
            else:
                return self.predict_single(x, node['right'])
        else:
            return 1 if node > 0.5 else 0
    
    def predict(self, X):
        return [self.predict_single(row, self.tree) for _, row in X.iterrows()]

def train_model():
    data = pd.read_pickle('data/preprocessed_data.pkl')
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Train test split manual
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(0.8 * len(X))
    
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model = ManualDecisionTree(max_depth=3)
    model.fit(X_train, y_train)
    
    # Simpan model
    with open('model_jantung_manual.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model manual berhasil disimpan!")

if __name__ == "__main__":
    train_model()