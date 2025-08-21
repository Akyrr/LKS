import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def evaluate_model():
    data = pd.read_pickle('data/preprocessed_data.pkl')
    
    with open('model_jantung_manual.pkl', 'rb') as f:
        model = pickle.load(f)
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Train test split manual
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(0.8 * len(X))
    test_idx = indices[split_idx:]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test.values, y_pred)
    print(f"Akurasi: {acc:.2f}")
    
    cm = confusion_matrix(y_test.values, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

if __name__ == "__main__":
    evaluate_model()