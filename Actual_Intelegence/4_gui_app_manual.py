import tkinter as tk
from tkinter import messagebox
import pandas as pd
import pickle

def load_model():
    with open('model_jantung_manual.pkl', 'rb') as f:
        model = pickle.load(f)
    data_columns = pd.read_pickle('data/preprocessed_data.pkl').columns.drop('target')
    return model, data_columns

model, columns = load_model()

def predict():
    try:
        input_data = {}
        for col in columns:
            input_data[col] = float(entries[col].get())
        
        # Convert to pandas Series for prediction
        input_series = pd.Series(input_data)
        prediction = model.predict_single(input_series, model.tree)
        
        result = "Berpotensi penyakit jantung" if prediction == 1 else "Sehat"
        messagebox.showinfo("Hasil", result)
    except Exception as e:
        messagebox.showerror("Error", f"Input tidak valid: {str(e)}")

app = tk.Tk()
app.title("Prediksi Penyakit Jantung (Manual)")

entries = {}
for i, col in enumerate(columns):
    tk.Label(app, text=col).grid(row=i, column=0)
    entry = tk.Entry(app)
    entry.grid(row=i, column=1)
    entries[col] = entry

tk.Button(app, text="Prediksi", command=predict).grid(row=len(columns), columnspan=2)
app.mainloop()