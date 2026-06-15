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

# 5. Membuat kategori produksi untuk SVC
batas_rendah = df["produksi_padi_ton"].quantile(1/3)
batas_tinggi = df["produksi_padi_ton"].quantile(2/3)

def buat_kategori(nilai):
    if nilai <= batas_rendah:
        return "Rendah"
    elif nilai <= batas_tinggi:
        return "Sedang"
    else:
        return "Tinggi"

df["kategori_produksi"] = df["produksi_padi_ton"].apply(buat_kategori)

print("\nBatas kategori produksi:")
print(f"Rendah  : <= {batas_rendah:.2f} ton")
print(f"Sedang  : > {batas_rendah:.2f} sampai <= {batas_tinggi:.2f} ton")
print(f"Tinggi  : > {batas_tinggi:.2f} ton")

print("\nJumlah kategori:")
print(df["kategori_produksi"].value_counts())

# 6. Menentukan fitur/input
fitur = ["tahun", "kabupaten_kota", "luas_panen_ha", "produktivitas_ku_ha"]

X = df[fitur]

# Target regresi
y_regresi = df["produksi_padi_ton"]

# Target klasifikasi SVC
y_svc = df["kategori_produksi"]

# 7. Preprocessing data
preprocessor = ColumnTransformer(
    transformers=[
        ("angka", StandardScaler(), ["tahun", "luas_panen_ha", "produktivitas_ku_ha"]),
        ("kategori", OneHotEncoder(handle_unknown="ignore"), ["kabupaten_kota"])
    ]
)

# ============================================================
# MODEL 1: REGRESI LINEAR
# ============================================================

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_regresi, test_size=0.2, random_state=42
)

model_regresi = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

model_regresi.fit(X_train_reg, y_train_reg)

prediksi_regresi = model_regresi.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, prediksi_regresi)
mse = mean_squared_error(y_test_reg, prediksi_regresi)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, prediksi_regresi)

print("\n===================================================")
print("HASIL EVALUASI REGRESI LINEAR")
print("===================================================")
print(f"MAE  : {mae:.2f}") 
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# ============================================================
# MODEL 2: SUPPORT VECTOR CLASSIFICATION
# ============================================================

X_train_svc, X_test_svc, y_train_svc, y_test_svc = train_test_split(
    X, y_svc, test_size=0.2, random_state=42, stratify=y_svc
)

model_svc = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", SVC(kernel="rbf"))
    ]
)

model_svc.fit(X_train_svc, y_train_svc)

prediksi_svc = model_svc.predict(X_test_svc)

akurasi = accuracy_score(y_test_svc, prediksi_svc)

print("\n===================================================")
print("HASIL EVALUASI SUPPORT VECTOR CLASSIFICATION")
print("===================================================")
print(f"Akurasi SVC: {akurasi:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_svc, prediksi_svc))