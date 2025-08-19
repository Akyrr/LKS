import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd

def load_model():
    model = joblib.load('model_jantung.pkl')
    data_columns = pd.read_pickle('data/preprocessed_data.pkl').columns.drop('target')
    return model, data_columns

model, columns = load_model()

def predict():
    try:
        input_data = [float(entries[col].get()) for col in columns]
        prediction = model.predict([input_data])
        
        result = "Berpotensi penyakit jantung" if prediction[0] == 1 else "Sehat"
        messagebox.showinfo("Hasil", result)
    except:
        messagebox.showerror("Error", "Input tidak valid")

app = tk.Tk()
app.title("Prediksi Penyakit Jantung")

entries = {}
for i, col in enumerate(columns):
    tk.Label(app, text=col).grid(row=i, column=0)
    entry = tk.Entry(app)
    entry.grid(row=i, column=1)
    entries[col] = entry

tk.Button(app, text="Prediksi", command=predict).grid(row=len(columns), columnspan=2)
app.mainloop()