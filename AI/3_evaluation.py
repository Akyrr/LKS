import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  

def evaluate_model():
    data = pd.read_pickle('data/preprocessed_data.pkl')
    model = joblib.load('model_jantung.pkl')
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_pred = model.predict(X_test)
    
    print("Akurasi:", accuracy_score(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()

if __name__ == "__main__":
    evaluate_model()