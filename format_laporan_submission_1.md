# Laporan Proyek Machine Learning - Khusnil Arifandi Purnama

## Domain Proyek

Penilaian kualitas anggur secara tradisional mengandalkan uji sensorik yang bersifat subjektif dan tidak efisien. Proyek ini bertujuan untuk memprediksi kualitas anggur putih Vinho Verde asal Portugal berdasarkan data fisikokimia, seperti kadar alkohol, keasaman, dan pH. Dengan memformulasikannya sebagai masalah regresi, model dapat mengestimasi skor kualitas (0–10) secara objektif. Selain prediksi, proyek ini juga membuka peluang untuk analisis fitur penting dan deteksi outlier guna mengidentifikasi anggur berkualitas.



## Business Understanding

### Problem Statements

- Bagaimana cara memprediksi kualitas anggur secara objektif tanpa bergantung pada uji manual yang subjektif dan memakan waktu?
- Apakah variabel-variabel fisikokimia seperti alkohol, pH, dan kadar asam memiliki hubungan signifikan terhadap skor kualitas anggur?
- Model regresi apa yang paling efektif dalam memprediksi nilai kualitas anggur berdasarkan data yang tersedia?
- Bagaimana perusahaan dapat meminimalkan kerugian akibat produk gagal atau kualitas rendah yang luput dari kontrol manual?
### Goals


- Proyek ini bertujuan untuk mengembangkan model regresi yang mampu memprediksi skor kualitas anggur putih Vinho Verde berdasarkan data uji laboratorium terhadap sifat fisikokimia anggur, seperti kadar alkohol, keasaman, pH, dan kandungan sulfur. Dengan demikian, proses penilaian kualitas dapat dilakukan secara objektif, konsisten, dan otomatis tanpa harus bergantung pada panel uji sensorik manusia yang cenderung mahal, subjektif, dan sulit diulang secara identik. Dalam konteks bisnis, tujuan ini mendukung pengambilan keputusan berbasis data (data-driven decision making) dalam proses kontrol mutu dan efisiensi produksi
-  Salah satu tujuan dari analisis eksplorasi data adalah untuk memahami hubungan masing-masing variabel fisikokimia terhadap skor kualitas anggur. Dengan mengidentifikasi variabel yang memiliki korelasi tinggi terhadap target (seperti alkohol yang memiliki korelasi sebesar 0.45), kita dapat mengarahkan fokus model pada fitur-fitur yang paling berpengaruh. Selain itu, pemahaman ini dapat digunakan untuk menyederhanakan proses produksi dan pengujian di industri dengan mengurangi pengukuran terhadap variabel-variabel yang memiliki pengaruh kecil, sehingga efisiensi biaya dan waktu dapat tercapai. Tujuan ini juga mendukung seleksi fitur dalam pembangunan model machine learning agar lebih ringan dan akurat.
- Tujuan dari evaluasi ini adalah untuk membandingkan berbagai algoritma regresi dalam konteks akurasi prediksi kualitas anggur, agar dapat memilih model terbaik yang memberikan error prediksi paling rendah. Dengan menggunakan metrik evaluasi seperti Mean Squared Error (MSE), proyek ini bertujuan menemukan metode yang tidak hanya andal dalam prediksi tetapi juga cukup general (tidak overfitting). Hasil menunjukkan bahwa model seperti K-Nearest Neighbors (KNN) dan AdaBoost Regressor memiliki performa yang lebih baik dibanding Random Forest dalam konteks dataset ini. Pemilihan model yang optimal ini diharapkan mampu meningkatkan keandalan sistem prediksi kualitas untuk digunakan dalam proses kontrol kualitas otomatis di industri wine.
- Perusahaan dapat meminimalkan kerugian akibat produk gagal atau kualitas rendah yang luput dari kontrol manual dengan mengimplementasikan model regresi yang dikembangkan untuk prediksi skor kualitas anggur secara otomatis dan objektif. Dengan memahami hubungan antara variabel fisikokimia dan kualitas, serta memilih algoritma prediksi terbaik seperti KNN atau AdaBoost yang memiliki error rendah, perusahaan dapat melakukan kontrol mutu secara lebih andal, konsisten, dan efisien, sehingga mengurangi ketergantungan pada uji sensorik manual yang subjektif dan mahal


## Data Understanding
Dataset yang digunakan dalam proyek ini adalah Wine Quality Dataset, yang berisi data mengenai kualitas anggur putih Vinho Verde dari Portugal berdasarkan hasil uji laboratorium terhadap sifat fisikokimia anggur. Dataset ini tersedia secara publik melalui [UCI Wine Quality Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) dan bersumber dari penelitian oleh Cortez et al. (2009). Dataset berjumlah 4898 baris dan 12 fitur.

Dataset ini banyak digunakan untuk tugas regresi maupun klasifikasi, karena label target berupa nilai kualitas yang dapat ditafsirkan sebagai kelas atau skor kontinu. Dalam proyek ini, pendekatan yang digunakan adalah regresi untuk memprediksi nilai kualitas anggur.


### Variabel-variabel pada  UCI Wine Quality dataset adalah sebagai berikut:
- fixed acidity           : Kadar asam tetap (misalnya asam tartarat), yang berkontribusi pada rasa segar anggur.
- volatile acidity        : Kadar asam volatil (misalnya asam asetat), jumlah yang tinggi dapat menyebabkan rasa tidak enak (off-flavor).
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

Dalam data understanding ini, akan di cek data duplikasi menggunakan .duplicated().sum() sehingga didapat data duplikat sebanyak 937 yaitu sekitar 5% dari dataset.
![image](https://github.com/user-attachments/assets/7ff6f986-4484-4ff8-9f34-7ce69150298d)


Setelah itu, akan di cek missing value yang terkandung dalam dataset, ternyata dataset tersebut tidak terdapat missing value yang dapat dilihat sebagai berikut:

![image](https://github.com/user-attachments/assets/fc2cf9d9-d330-4978-b735-52b3e60eaf50)
Namun, setelah menggunakan .describe(), terdapat nilai 0 berjumlah 19 pada citric acid.
![image](https://github.com/user-attachments/assets/f730eb2a-3391-47f3-bbf2-5a3478f6e690)

![image](https://github.com/user-attachments/assets/768e6815-1cb5-4e47-8852-0394d11e06f7)

Dalam dataset kualitas anggur putih ini, outlier merujuk pada sampel anggur yang memiliki karakteristik fisikokimia ekstrem yang sangat berbeda dari mayoritas data. Berikut merupakan outlier dari dataset white_df disajikan dalam bentuk bloxpot:
![image](https://github.com/user-attachments/assets/46a6a920-5faf-4ac4-8b99-56f9515ec149)
![image](https://github.com/user-attachments/assets/17e594f0-0e9c-4278-87f8-f5cb4dfff590)
![image](https://github.com/user-attachments/assets/ee645891-0e6a-42df-bf8a-0a215a000d26)
![image](https://github.com/user-attachments/assets/411ec29e-d935-45ab-8bbd-65943f5c7d0b)
![image](https://github.com/user-attachments/assets/bddd1567-d3af-4433-bf9a-f572e60e99c9)
![image](https://github.com/user-attachments/assets/6e8d1afc-323b-437c-a7aa-f69037dd6f5f)

Selanjutnya, akan dilakukan penyajian analisis deskriptif dari dataset dalam bentuk histogram  dan correlation matrix dari fitur dalam dataset:

![image](https://github.com/user-attachments/assets/4e7995dc-b4ef-439b-b2e1-3b02250d060e)
Dalam gambar tersebut, didapat jumlah sampel berdasar nilai quality. Nilai quality 6 mengandung sampel terbanyak dan diikuti dengan quality 5 dan 7. Hal ini menunjukkan bahwa distribusi sampil banyak tersebar di kualitas anggur 6.

![image](https://github.com/user-attachments/assets/759e4698-9060-47f1-94a9-bbf1490ee86d)
Gambar diatas menunjukkan persebaran sample data dengan fitur dari dataset menggunakan histogram. Dalam gambar tersebut dapat dilihat bahwa fixed acidity, volatile acidity, citric acid, pH, free sulfur dioxide, total sulfur dioxide mempunyai persebaran distribusi normal.

Terakhir, akan disajikan correlation matrix dari fitur numerik untuk menunjukkan hubungan antar fitur, correlation matrix sebagai berikut:

![image](https://github.com/user-attachments/assets/32bfac42-1574-4f5f-ada2-3c9d85ee9b1e)



## Data Preparation
Tahapan data preparation dalam proyek ini mencakup beberapa langkah penting untuk memastikan kualitas data sebelum pemodelan. Pertama, dataset dibaca dan diperiksa untuk mengetahui struktur dan kelengkapan datanya. Tidak ditemukan nilai kosong, sehingga tidak diperlukan proses imputasi.
Namun, dalam data understanding sebelumnya ditemukan duplikasi data sehingga  .drop_duplicates(inplace=True) untuk mengapus data duplikat tersebut.

selanjutnya adalah menangani nilai 0 dalam citric acid, tindakan yang dilakukan adalah akan diambil semua baris citric acid yang tidak mempunyai nilai 0.

Pada data undertanding, diketahui bahwa terdapat outlier dibeberapa fitur dalam dataset, untuk itu digunakan metode interquartile untuk menangani outlier tersebut.

Selanjutnya, akan dilakukan feature selection menggunakan metode PCA untuk menyeleksi fitur yang berkolerasi tinggi satu sama lain. Didapat fitur residual sugar, total sulfur dioxide dan density mempunyai korelasi sedikit tinggi satu sama lain.

![image](https://github.com/user-attachments/assets/f4f9659a-fa4a-4398-b189-2c3986b37972)

Maka, ketiga fitur tersebut yang akan di reduksi menggunakan PCA. Didapat explained_variance_ratio_ array sebagai berikut:
![image](https://github.com/user-attachments/assets/28cb6893-6cba-4656-ae85-c4fb97048da9)
Nilai PCA komponen satu memiliki nilai 0.99, hal ini mengartikan bahwa komponen tersebut akan menjelaskan ketiga fitur tadi menjadi 1 fitur. Setelah itu, komponen tersebut akan dimasukkan ke dataframe white_df dan akan dihapus ketiga fitur sebelumnya.

Selanjutnya adalah splitting data, memisahkan data menjadi data latih dan data test dengan perbandingan 80:20. Pembagian data sebagai berikut:
![image](https://github.com/user-attachments/assets/f4debfc0-bb74-40b8-b790-4c9b83902cf8)

Karena skala antar fitur sangat bervariasi, dilakukan proses standardisasi menggunakan StandardScaler agar seluruh fitur berada pada skala yang sebanding, sehingga model dapat bekerja secara optimal.
![image](https://github.com/user-attachments/assets/82b27b7b-c38a-4b61-a231-dff7c94831a5)



## Modeling
Dalam proyek ini, digunakan pendekatan regresi untuk memprediksi kualitas anggur putih berdasarkan data fisikokimia. Tiga algoritma regresi digunakan dalam pemodelan ini, yaitu Random Forest Regressor, K-Nearest Neighbors (KNN) Regressor, dan AdaBoost Regressor. 
![image](https://github.com/user-attachments/assets/77ad902d-ae71-4752-9425-e3fa09e462c4)

Random Forest digunakan sebagai salah satu model utama karena kemampuannya yang baik dalam menangani data tabular kompleks dan menangkap hubungan nonlinier antar fitur tanpa memerlukan asumsi distribusi data tertentu. Algoritma ini bekerja dengan membangun sejumlah besar decision trees (pohon keputusan) secara acak selama proses pelatihan. Setiap pohon dibangun dari sampel bootstrap data latih dan mempertimbangkan subset acak dari fitur pada setiap pemisahan. Untuk prediksi regresi, hasil dari setiap pohon individu kemudian dirata-ratakan untuk menghasilkan prediksi akhir. Pendekatan ansambel ini membantu mengurangi overfitting yang sering terjadi pada pohon keputusan tunggal dan meningkatkan robustisitas model.

Parameter yang digunakan dalam model ini adalah n_estimators sebanyak 50, yang berarti 50 pohon keputusan dibangun, dan max_depth sebesar 16, yang membatasi kedalaman maksimum setiap pohon untuk mengontrol kompleksitas
![image](https://github.com/user-attachments/assets/2fd938a7-1888-43b6-a5de-b9b44fdef4d4)

K-Nearest Neighbors (KNN) adalah algoritma pembelajaran berbasis instans yang sederhana namun seringkali efektif. Untuk prediksi regresi, KNN bekerja dengan mengidentifikasi 'k' titik data terdekat (tetangga) dari titik data baru dalam ruang fitur, berdasarkan metrik jarak tertentu (misalnya, jarak Euclidean). Nilai prediksi untuk titik data baru kemudian dihitung sebagai rata-rata (atau median) dari nilai target para 'k' tetangga terdekat tersebut. Performa KNN sangat bergantung pada pemilihan nilai 'k' dan metrik jarak yang sesuai.

Dalam implementasi ini, parameter utama yang ditetapkan adalah jumlah tetangga n_neighbors=10
![image](https://github.com/user-attachments/assets/03d8cecb-0f47-4a7c-891d-4412ecd0889a)

AdaBoost (Adaptive Boosting) Regressor dipilih karena merupakan salah satu algoritma boosting yang populer dan efektif. Algoritma ini bekerja dengan membangun model secara sekuensial, di mana setiap model berikutnya mencoba untuk memperbaiki kesalahan yang dibuat oleh model sebelumnya. Secara spesifik, AdaBoost menyesuaikan bobot dari instans data latih pada setiap iterasi. Instans yang salah diprediksi oleh model sebelumnya akan diberi bobot yang lebih tinggi, sehingga model berikutnya akan lebih fokus pada instans-instans yang sulit tersebut. Prediksi akhir merupakan kombinasi tertimbang dari prediksi semua model lemah yang telah dibangun.

Parameter kunci yang digunakan adalah learning_rate sebesar 0.05, yang mengontrol kontribusi setiap model lemah terhadap ansambel akhir, membantu mencegah overfitting dengan membuat proses pembelajaran lebih bertahap.
![image](https://github.com/user-attachments/assets/ca63ca85-36f3-4a48-a22c-261dc63d2b8d)



## Evaluation
Pada proyek ini, metrik evaluasi yang digunakan adalah Mean Squared Error (MSE), yang sangat relevan untuk permasalahan regresi karena mengukur rata-rata dari kuadrat selisih antara nilai prediksi dan nilai aktual.
![image](https://github.com/user-attachments/assets/d59d0e5d-f712-4c7d-890c-7aacb327f351)

MSE memberikan penalti yang lebih besar terhadap kesalahan yang ekstrem, sehingga efektif dalam menilai akurasi model secara keseluruhan. Tiga model regresi yang diuji adalah K-Nearest Neighbors (KNN), Random Forest (RF), dan AdaBoost Regressor. 
![image](https://github.com/user-attachments/assets/8f78e3ed-4bda-4fe9-944a-1f08e1ceeff4)
![image](https://github.com/user-attachments/assets/1a5fff92-ca95-452e-b9c8-b0e604692570)

Berdasarkan hasil evaluasi, model Random Forest memiliki nilai MSE yang sangat kecil pada data latih (0.000092), tetapi meningkat cukup signifikan pada data uji (0.000535). Hal ini menunjukkan bahwa model cenderung mengalami overfitting, yaitu terlalu menyesuaikan diri terhadap data pelatihan dan kurang mampu menggeneralisasi ke data baru. Sebaliknya, model AdaBoost menunjukkan performa yang lebih stabil dengan MSE sebesar 0.000532 pada data latih dan 0.000526 pada data uji, menjadikannya model dengan performa terbaik dalam proyek ini. Sementara itu, model KNN mencatatkan MSE sebesar 0.000508 (train) dan 0.000595 (test), yang juga mengindikasikan sedikit overfitting. 
![image](https://github.com/user-attachments/assets/3e75ae90-96fa-49ce-ab27-1773ab7f1ade)

1.  **Menjawab Problem Statement:**
    * **"Bagaimana cara memprediksi kualitas anggur secara objektif tanpa bergantung pada uji manual yang subjektif dan memakan waktu?"** Evaluasi ini menunjukkan bahwa model AdaBoost dengan MSE 0.000526 pada data uji mampu memberikan prediksi kualitas anggur secara objektif dan otomatis, menggantikan kebutuhan uji manual yang subjektif.
    * **"Apakah variabel-variabel fisikokimia seperti alkohol, pH, dan kadar asam memiliki hubungan signifikan terhadap skor kualitas anggur?"** Analisis korelasi (seperti yang terlihat pada Correlation Matrix) menunjukkan bahwa beberapa variabel fisikokimia memang memiliki hubungan dengan skor kualitas. Sebagai contoh, alkohol menunjukkan korelasi positif yang cukup terlihat dengan kualitas (koefisien sekitar 0.44), mengindikasikan bahwa kadar alkohol yang lebih tinggi cenderung berhubungan dengan kualitas yang lebih baik. Sebaliknya, density memiliki korelasi negatif sedang (sekitar -0.31), yang berarti semakin tinggi densitas, kualitas cenderung menurun. Variabel lain seperti pH menunjukkan korelasi positif yang lebih lemah (sekitar 0.1), sementara volatile acidity memiliki korelasi negatif lemah (sekitar -0.19). Keberhasilan model AdaBoost yang dilatih menggunakan keseluruhan set variabel fisikokimia ini dalam mencapai MSE yang rendah (0.000526 pada data uji) mengkonfirmasi secara praktis bahwa kombinasi variabel-variabel ini, dengan berbagai tingkat korelasi individualnya, secara kolektif signifikan dan prediktif terhadap kualitas anggur. Ini menunjukkan bahwa model mampu menangkap pola kompleks dari interaksi fitur-fitur tersebut, bahkan jika tidak semua fitur memiliki korelasi individual yang kuat.
    * **"Model regresi apa yang paling efektif dalam memprediksi nilai kualitas anggur berdasarkan data yang tersedia?"** Evaluasi ini secara langsung menjawab pertanyaan ini dengan mengidentifikasi AdaBoost Regressor sebagai model paling efektif dibandingkan KNN dan Random Forest, berdasarkan stabilitas dan akurasi (MSE rendah) pada data uji.
    * **"Bagaimana perusahaan dapat meminimalkan kerugian akibat produk gagal atau kualitas rendah yang luput dari kontrol manual?"** Dengan memilih dan mengimplementasikan model AdaBoost yang terbukti andal, perusahaan dapat meningkatkan akurasi deteksi kualitas rendah secara otomatis, sehingga meminimalkan produk gagal yang lolos ke pasar dan mengurangi kerugian terkait.

2.  **Mencapai Goals yang Diharapkan:**
    * **Goal pertama** (mengembangkan model regresi untuk prediksi objektif, konsisten, dan otomatis) tercapai dengan terpilihnya AdaBoost yang menawarkan prediksi dengan error rendah dan stabil, mendukung pengambilan keputusan berbasis data dalam kontrol mutu dan efisiensi produksi.
    * **Goal kedua** (memahami hubungan variabel fisikokimia dan mengidentifikasi fitur berpengaruh) didukung oleh hasil evaluasi, karena model yang baik (AdaBoost) berhasil dibangun menggunakan fitur-fitur yang diidentifikasi sebelumnya (seperti alkohol). Ini menunjukkan bahwa fokus pada fitur berpengaruh telah efektif.
    * **Goal ketiga** (membandingkan algoritma dan memilih model terbaik dengan error rendah dan tidak *overfitting*) tercapai sepenuhnya melalui proses evaluasi ini, yang menghasilkan AdaBoost sebagai model optimal. Ini meningkatkan keandalan sistem prediksi untuk kontrol kualitas otomatis.
    * **Goal keempat** (meminimalkan kerugian akibat produk gagal) dapat dicapai dengan implementasi model AdaBoost yang telah dievaluasi ini, karena menyediakan alat kontrol kualitas yang lebih akurat dan efisien daripada metode manual.

3.  **Dampak Solusi Statement yang Direncanakan:**
    * Solusi untuk mengembangkan model regresi berdampak positif karena menghasilkan AdaBoost Regressor, sebuah alat prediksi kualitas yang konkret dan terukur.
    * Solusi untuk memahami hubungan variabel fisikokimia (melalui EDA dan dikonfirmasi oleh performa model) berdampak pada efisiensi, karena fokus dapat diarahkan pada pengukuran variabel yang paling relevan, berpotensi mengurangi biaya dan waktu pengujian di masa depan.
    * Solusi untuk memilih model terbaik melalui evaluasi komprehensif (seperti yang dilakukan di sini) berdampak langsung pada keandalan dan akurasi sistem prediksi, memastikan bahwa investasi dalam pengembangan model memberikan hasil yang optimal dan dapat dipertanggungjawabkan untuk aplikasi industri.

Dengan demikian, evaluasi ini tidak hanya mengidentifikasi model prediksi terbaik tetapi juga memvalidasi bahwa pendekatan yang diambil selaras dengan tujuan bisnis, mampu menjawab permasalahan yang ada, dan solusi yang diusulkan memberikan dampak positif yang diharapkan dalam meningkatkan kontrol kualitas dan efisiensi produksi anggur.

