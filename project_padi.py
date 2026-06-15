# PREDIKSI PRODUKSI PADI SULAWESI TENGAH
# Menggunakan Regresi Linear dan Support Vector Classification
# Dataset: BPS Sulawesi Tengah 2018-2025
# 1. Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_repor

# 2. Membaca dataset
file_path = r"E:\PHYTON\Project Statistika\DATASET_PRODUKSI_PADI_2018_2025.xlsx"

df = pd.read_excel(file_path, sheet_name="Dataset_Model")

print("Data berhasil dibaca")
print(df.to_string())

# 3. Merapikan nama kolom
df.columns = df.columns.str.lower().str.strip()

# Pastikan kolom yang dipakai sesuai
print("\nNama kolom dataset:")
print(df.columns)

# 4. Membersihkan data
df = df.dropna()

# Menghapus baris total provinsi jika ada
df = df[df["kabupaten_kota"].str.lower() != "sulawesi tengah"]