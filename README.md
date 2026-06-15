Prediksi Produksi Padi Kabupaten/Kota di Sulawesi Tengah Berdasarkan Luas Panen dan Produktivitas Menggunakan Regresi dan SVC
Repositori ini berisi implementasi proyek komputasi Machine Learning untuk menganalisis dan memprediksi produksi padi kabupaten/kota di Provinsi Sulawesi Tengah berdasarkan data luas panen, produktivitas, dan produksi padi. Proyek ini dikembangkan menggunakan bahasa pemrograman Python dengan menerapkan metode Regresi Linear dan Support Vector Classification (SVC).

Proyek ini disusun sebagai bagian dari pemenuhan tugas mata kuliah Statistik dan Probabilitas.
📌 Latar Belakang Studi
Padi merupakan salah satu komoditas pangan utama yang memiliki peran penting dalam mendukung ketahanan pangan masyarakat. Di Provinsi Sulawesi Tengah, produksi padi setiap kabupaten/kota dapat berbeda-beda karena dipengaruhi oleh beberapa faktor, seperti luas panen dan produktivitas lahan.

Luas panen menunjukkan seberapa besar area pertanian yang berhasil dipanen, sedangkan produktivitas menunjukkan hasil produksi padi per satuan luas lahan. Kedua variabel ini memiliki hubungan yang kuat dengan jumlah produksi padi. Oleh karena itu, data luas panen dan produktivitas dapat digunakan sebagai dasar untuk membangun model prediksi produksi padi.

Dalam proyek ini, metode Regresi Linear digunakan untuk memprediksi jumlah produksi padi dalam satuan ton. Sementara itu, metode Support Vector Classification digunakan untuk mengklasifikasikan tingkat produksi padi ke dalam tiga kategori, yaitu rendah, sedang, dan tinggi.

📁 Struktur Dataset
Dataset memiliki beberapa kolom utama sebagai berikut:

Kolom	Keterangan
tahun	Tahun data produksi padi
kabupaten_kota	Nama kabupaten/kota di Provinsi Sulawesi Tengah
luas_panen_ha	Luas panen padi dalam satuan hektare
produktivitas_ku_ha	Produktivitas padi dalam satuan kuintal per hektare
produksi_padi_ton	Jumlah produksi padi dalam satuan ton
kategori_produksi	Kategori produksi padi: rendah, sedang, atau tinggi
🏷️ Kategori Produksi
Untuk kebutuhan metode Support Vector Classification, data produksi padi diklasifikasikan ke dalam tiga kategori, yaitu rendah, sedang, dan tinggi. Pembagian kategori dilakukan berdasarkan distribusi nilai produksi padi pada dataset.

Batas kategori yang digunakan adalah sebagai berikut:

Kategori	Batas Produksi
Rendah	Produksi ≤ 26.337,74 ton
Sedang	26.337,74 < Produksi ≤ 71.057,97 ton
Tinggi	Produksi > 71.057,97 ton

🧠 Metode yang Digunakan
Regresi Linear
Regresi Linear digunakan untuk memprediksi nilai numerik produksi padi. Metode ini cocok digunakan karena produksi padi memiliki hubungan yang kuat dengan luas panen dan produktivitas.
Support Vector Classification
Support Vector Classification digunakan untuk proses klasifikasi. Dalam proyek ini, SVC digunakan untuk mengelompokkan produksi padi ke dalam kategori rendah, sedang, dan tinggi.

Project ini menunjukkan penerapan metode Regresi Linear dan Support Vector Classification pada data produksi padi kabupaten/kota di Provinsi Sulawesi Tengah. Regresi Linear digunakan untuk memprediksi jumlah produksi padi, sedangkan SVC digunakan untuk mengklasifikasikan produksi padi ke dalam kategori rendah, sedang, dan tinggi.Dengan menggunakan data resmi BPS tahun 2018–2025, project ini dapat menjadi contoh penerapan machine learning sederhana dalam bidang pertanian, khususnya untuk menganalisis dan memprediksi produksi padi berdasarkan luas panen dan produktivitas.
