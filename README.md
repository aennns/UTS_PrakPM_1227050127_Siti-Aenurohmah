# Tahapan Pembuatan Model Klasifikasi menggunakan Algoritma Decision Tree<br>
1. Pengumpulan dan Persiapan Data<br>
Mengunduh dataset dari tautan.<br>
Dataset diimpor menggunakan pandas.

2. Eksplorasi Data Awal<br>
Melihat bentuk data, tipe data, nilai unik, dan apakah ada nilai yang hilang.

3. Pra-Pemrosesan Data<br>
Encoding Variabel Kategorikal: Algoritma Decision Tree umumnya bekerja dengan data numerik. Ubah variabel kategorikal menjadi format numerik menggunakan teknik seperti LabelEncoder atau OneHotEncoder.

4. Pembagian Data<br>
Bagi dataset menjadi dua bagian:<br>
- Training Set: Digunakan untuk melatih model.<br>
- Testing Set: Digunakan untuk mengevaluasi kinerja model.<br>
Proporsi pembagian biasanya 80:20 atau 70:30.

5. Membangun Model<br>
Gunakan DecisionTreeClassifier dari sklearn.tree.

6. Pelatihan Model<br>
Latih model menggunakan training set dengan memanggil metode fit()

7. Evaluasi Model<br>
Gunakan metrik akurasi, confusion matrix, precision, recall, dan F1-score.
