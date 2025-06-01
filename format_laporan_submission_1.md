# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek

Penilaian kualitas anggur secara tradisional mengandalkan uji sensorik yang bersifat subjektif dan tidak efisien. Proyek ini bertujuan untuk memprediksi kualitas anggur putih Vinho Verde asal Portugal berdasarkan data fisikokimia, seperti kadar alkohol, keasaman, dan pH. Dengan memformulasikannya sebagai masalah regresi, model dapat mengestimasi skor kualitas (0–10) secara objektif. Selain prediksi, proyek ini juga membuka peluang untuk analisis fitur penting dan deteksi outlier guna mengidentifikasi anggur berkualitas.



## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana cara memprediksi kualitas anggur secara objektif tanpa bergantung pada uji manual yang subjektif dan memakan waktu?
- Apakah variabel-variabel fisikokimia seperti alkohol, pH, dan kadar asam memiliki hubungan signifikan terhadap skor kualitas anggur?
- Model regresi apa yang paling efektif dalam memprediksi nilai kualitas anggur berdasarkan data yang tersedia?
- Bagaimana perusahaan dapat meminimalkan kerugian akibat produk gagal atau kualitas rendah yang luput dari kontrol manual?
### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Proyek ini bertujuan untuk mengembangkan model regresi yang mampu memprediksi skor kualitas anggur putih Vinho Verde berdasarkan data uji laboratorium terhadap sifat fisikokimia anggur, seperti kadar alkohol, keasaman, pH, dan kandungan sulfur. Dengan demikian, proses penilaian kualitas dapat dilakukan secara objektif, konsisten, dan otomatis tanpa harus bergantung pada panel uji sensorik manusia yang cenderung mahal, subjektif, dan sulit diulang secara identik. Dalam konteks bisnis, tujuan ini mendukung pengambilan keputusan berbasis data (data-driven decision making) dalam proses kontrol mutu dan efisiensi produksi
-  Salah satu tujuan dari analisis eksplorasi data adalah untuk memahami kontribusi relatif masing-masing variabel fisikokimia terhadap skor kualitas anggur. Dengan mengidentifikasi variabel yang memiliki korelasi tinggi terhadap target (seperti alkohol yang memiliki korelasi sebesar 0.45), kita dapat mengarahkan fokus model pada fitur-fitur yang paling berpengaruh. Selain itu, pemahaman ini dapat digunakan untuk menyederhanakan proses produksi dan pengujian di industri dengan mengurangi pengukuran terhadap variabel-variabel yang memiliki pengaruh kecil, sehingga efisiensi biaya dan waktu dapat tercapai. Tujuan ini juga mendukung seleksi fitur dalam pembangunan model machine learning agar lebih ringan dan akurat.
- Tujuan dari evaluasi ini adalah untuk membandingkan berbagai algoritma regresi dalam konteks akurasi prediksi kualitas anggur, agar dapat memilih model terbaik yang memberikan error prediksi paling rendah. Dengan menggunakan metrik evaluasi seperti Mean Squared Error (MSE), proyek ini bertujuan menemukan metode yang tidak hanya andal dalam prediksi tetapi juga cukup general (tidak overfitting). Hasil menunjukkan bahwa model seperti K-Nearest Neighbors (KNN) dan AdaBoost Regressor memiliki performa yang lebih baik dibanding Random Forest dalam konteks dataset ini. Pemilihan model yang optimal ini diharapkan mampu meningkatkan keandalan sistem prediksi kualitas untuk digunakan dalam proses kontrol kualitas otomatis di industri wine.



## Data Understanding
Dataset yang digunakan dalam proyek ini adalah Wine Quality Dataset, yang berisi data mengenai kualitas anggur putih Vinho Verde dari Portugal berdasarkan hasil uji laboratorium terhadap sifat fisikokimia anggur. Dataset ini tersedia secara publik melalui [UCI Wine Quality Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) dan bersumber dari penelitian oleh Cortez et al. (2009).

Dataset ini banyak digunakan untuk tugas regresi maupun klasifikasi, karena label target berupa nilai kualitas yang dapat ditafsirkan sebagai kelas atau skor kontinu. Dalam proyek ini, pendekatan yang digunakan adalah regresi untuk memprediksi nilai kualitas anggur.


### Variabel-variabel pada  UCI Wine Quality dataset adalah sebagai berikut:
- accepts                 : Kadar asam tetap (misalnya asam tartarat), yang berkontribusi pada rasa segar anggur.
- cuisine                 : Kadar asam volatil (misalnya asam asetat), jumlah yang tinggi dapat menyebabkan rasa tidak enak (off-flavor).
- citric acid             : Asam sitrat yang menambah kesegaran rasa; kadar rendah membuat anggur terasa lemah.
- residual sugar          : Sisa gula setelah fermentasi; berkontribusi pada rasa manis dalam anggur
- chlorides               : Konsentrasi garam, terutama natrium klorida, yang mempengaruhi rasa dan stabilitas anggur.
- free sulfur dioxide     : SO₂ bebas yang bertindak sebagai antioksidan dan pengawet terhadap mikroorganisme
- total sulfur dioxide    : Jumlah total SO₂ (bebas dan terikat); jumlah tinggi dapat menyebabkan iritasi.
- density                 : Massa jenis cairan anggur, dipengaruhi oleh kadar gula dan alkohol.
- pH                      : Ukuran keasaman atau kebasaan anggur; pH yang terlalu tinggi atau rendah dapat memengaruhi rasa dan stabilitas.
- sulphates               : Senyawa yang berfungsi sebagai pengawet dan agen antimikroba; berkontribusi pada rasa pahit jika berlebihan.
- alcohol                 : Persentase kandungan alkohol dalam anggur; kadar alkohol umumnya mempengaruhi persepsi kualitas secara signifikan.
Berikut merupakan cuplikan isi dataset:
![image](https://github.com/user-attachments/assets/812b7373-37e3-4661-8424-cc85d31c40d4)



## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.


## Modeling
Dalam proyek ini, digunakan pendekatan regresi untuk memprediksi kualitas anggur putih berdasarkan data fisikokimia. Proses dimulai dengan membagi dataset menjadi data latih dan data uji menggunakan metode train_test_split dengan rasio 80:20. Karena skala antar fitur sangat bervariasi, dilakukan proses standardisasi menggunakan StandardScaler agar seluruh fitur berada pada skala yang sebanding, sehingga model dapat bekerja secara optimal.

Tiga algoritma regresi digunakan dalam pemodelan ini, yaitu Random Forest Regressor, K-Nearest Neighbors (KNN) Regressor, dan AdaBoost Regressor. Random Forest digunakan sebagai baseline model karena kemampuannya dalam menangani data tabular dan menangkap hubungan nonlinier antar fitur. KNN digunakan untuk melihat performa model berbasis tetangga terdekat dengan parameter utama jumlah tetangga (n_neighbors=5). Sedangkan AdaBoost dipilih karena sifatnya yang mampu meningkatkan akurasi secara progresif dengan menggabungkan banyak model lemah.

Evaluasi kinerja model dilakukan menggunakan Mean Squared Error (MSE), yaitu metrik yang mengukur rata-rata kesalahan kuadrat antara nilai prediksi dan nilai aktual. Model dengan nilai MSE paling rendah dianggap memiliki performa terbaik. Hasil evaluasi menunjukkan bahwa model KNN dan AdaBoost menghasilkan nilai MSE yang lebih rendah dibanding baseline model Random Forest, sehingga lebih tepat digunakan untuk memprediksi kualitas anggur dalam konteks data ini.


## Evaluation
Pada proyek ini, metrik evaluasi yang digunakan adalah Mean Squared Error (MSE), yang sangat relevan untuk permasalahan regresi karena mengukur rata-rata dari kuadrat selisih antara nilai prediksi dan nilai aktual. MSE memberikan penalti yang lebih besar terhadap kesalahan yang ekstrem, sehingga efektif dalam menilai akurasi model secara keseluruhan. Tiga model regresi yang diuji adalah K-Nearest Neighbors (KNN), Random Forest (RF), dan AdaBoost Regressor. Berdasarkan hasil evaluasi, model Random Forest memiliki nilai MSE yang sangat kecil pada data latih (0.000092), tetapi meningkat cukup signifikan pada data uji (0.000535). Hal ini menunjukkan bahwa model cenderung mengalami overfitting, yaitu terlalu menyesuaikan diri terhadap data pelatihan dan kurang mampu menggeneralisasi ke data baru. Sebaliknya, model AdaBoost menunjukkan performa yang lebih stabil dengan MSE sebesar 0.000532 pada data latih dan 0.000526 pada data uji, menjadikannya model dengan performa terbaik dalam proyek ini. Sementara itu, model KNN mencatatkan MSE sebesar 0.000508 (train) dan 0.000595 (test), yang juga mengindikasikan sedikit overfitting. Secara keseluruhan, hasil evaluasi ini menunjukkan bahwa model AdaBoost lebih unggul dalam memberikan prediksi yang akurat dan konsisten terhadap kualitas anggur putih berdasarkan parameter fisikokimia.


