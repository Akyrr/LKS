import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_model():
    data = pd.read_pickle('data/preprocessed_data.pkl')
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Simpan model
    joblib.dump(model, 'model_jantung.pkl')
    print("Model berhasil disimpan!")

if __name__ == "__main__":
    train_model()