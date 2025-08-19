import pandas as pd
import matplotlib.pyplot as plt

def load_and_analyze_data():
    data = pd.read_csv('data/Datasset LKS AI Kabupaten Malang 2025.csv')
    
    # Cek data hilang
    print("Data yang hilang:\n", data.isnull().sum())
    
    # Analisis sederhana
    print("\nDistribusi target:\n", data['target'].value_counts())
    
    # Visualisasi
    data['age'].hist()
    plt.title('Distribusi Usia Pasien')
    plt.show()
    
    return data

if __name__ == "__main__":
    data = load_and_analyze_data()
    data.to_pickle('data/preprocessed_data.pkl')  # Simpan data yang sudah dibersihkan