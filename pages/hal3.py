import streamlit as st
st.subheader("""Kode 1 dan kode 2 mengeksekusi fungsi yang berbeda dan memiliki logika yang berbeda pula.

Pada kode 1, model visualisasi data dipilih melalui antarmuka pengguna (UI) yang memungkinkan pengguna memilih tipe visualisasi yang ingin mereka lihat. Visualisasi yang ditampilkan tidak secara langsung terkait dengan prediksi kegagalan. Data yang divisualisasikan adalah data mentah tanpa melakukan prediksi kegagalan.

Di sisi lain, pada kode 2, model klasifikasi dipilih melalui antarmuka pengguna. Setelah model dipilih, model tersebut dilatih dan dievaluasi menggunakan data yang sama. Kemudian, hasil prediksi kegagalan ditampilkan berdasarkan model yang dipilih.

Jadi, perbedaan yang terlihat dalam jumlah kegagalan antara kedua kode tersebut mungkin disebabkan oleh cara data ditangani dan diproses dalam masing-masing kode. Kode pertama tidak melakukan prediksi kegagalan, sehingga tidak ada informasi tentang jumlah kegagalan yang ditampilkan. Sedangkan kode kedua fokus pada prediksi kegagalan berdasarkan model klasifikasi yang dipilih, sehingga menampilkan informasi tentang jumlah kegagalan.""")