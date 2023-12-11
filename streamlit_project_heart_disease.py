import streamlit as st
import pandas as pd
import pickle
import time
import numpy as np
import sklearn
from PIL import Image


import requests
from io import BytesIO
# import subprocess
# scikit_learn_version = "1.2.2"
# subprocess.check_call(["pip", "install", f"scikit-learn=={scikit_learn_version}"])

#page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon=":heart:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.sidebar.title("Navigation")

nav = st.sidebar.selectbox("Go to", ("Home",  "Dataset", "Exploratory Data Analysis", "Modelling", "Prediction", "About"))

#dataset page
url = "https://storage.googleapis.com/dqlab-dataset/heart_disease.csv"
df = pd.read_csv(url)
############################################## FUNGSI UNTUK PREDIKSI ############################################################################################
def heart():
    st.write("""
        Dataset ini didapatkan dari [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
        """)
    st.sidebar.header('User Input Features:')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_manual():
            st.sidebar.header("Manual Input")
            age = st.sidebar.slider("Age", 0, 100, 25)
            cp = st.sidebar.selectbox("Chest Pain Type", ("Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"))
            if cp == "Typical Angina":
                cp = 1
            elif cp == "Atypical Angina":
                cp = 2
            elif cp == "Non-anginal Pain":
                cp = 3
            elif cp == "Asymptomatic":
                cp = 4
            thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 0, 200, 100)
            slope = st.sidebar.selectbox("Slope", ("Upsloping", "Flat", "Downsloping"))
            if slope == "Upsloping":
                slope = 1
            elif slope == "Flat":
                slope = 2
            elif slope == "Downsloping":
                slope = 3
            ca = st.sidebar.slider("Number of Major Vessels", 0, 4, 0)
            oldpeak = st.sidebar.slider("ST Depression Induced by Exercise Relative to Rest", 0.0, 10.0, 0.0)
            exang = st.sidebar.selectbox("Exercise Induced Angina", ("Yes", "No"))
            if exang == "Yes":
                exang = 1
            elif exang == "No":
                exang = 0
            thal = st.sidebar.selectbox("Thal", ("Normal", "Fixed Defect", "Reversable Defect"))
            if thal == "Normal":
                thal = 1
            elif thal == "Fixed Defect":
                thal = 2
            elif thal == "Reversable Defect":
                thal = 3
            sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
            if sex == "Male":
                sex = 1
            else:
                sex =0
            data = {'cp': cp,
                    'thalach': thalach,
                    'slope': slope,
                    'oldpeak': oldpeak,
                    'exang': exang,
                    'ca': ca,
                    'thal': thal,
                    'sex': sex,
                    'age': age}
            features = pd.DataFrame(data, index=[0])
            return features

        input_df = user_input_manual()
        # Data df
    st.image('https://media.istockphoto.com/id/1210336572/id/foto/serangan-jantung-dan-penyakit-jantung.jpg?s=170667a&w=0&k=20&c=icnsTuCTLJ04C5yPZ_JjttoTmEMsgecOj9x7HVugFSo=', width=700)
    if st.sidebar.button("GO!"):
        df = input_df.copy()
        st.write(df)
        with open("modelFinal.pkl", "rb") as f:
            model = pickle.load(f)
        prediction = model.predict(df)
        result = ['No Heart Disease' if prediction == 0 else 'Heart Disease']
        with st.spinner('Wait for it...'):
            time.sleep(3)
            if prediction == 0:
                st.success('This patient has {}'.format(result[0]) )
                st.balloons()
            else:
                st.error('This patient has {}'.format(result[0]) )
            

############################################## UDAH MASUK KE WIDGET YA ########################################################################
#Home page
if nav == "Home":
    st.title("Heart Disease Prediction")
    st.write('''
    **Machine Learning & AI track**

    Hallo, perkenalkan saya William Deli, saya adalah mahasiswa di Binus University dengan mengambil jurusan computer engineering, 
    Saya membuat aplikasi ini untuk mengaplikasikan apa yang saya pelajari di bidang machine learning dan data science.
    ''')
    st.image("https://media.istockphoto.com/id/1128931450/photo/heart-attack-concept.jpg?s=612x612&w=0&k=20&c=XHOhTXhpZMSV6XIhXLbH6uvNQjZQS93b1UetGfqQXtI=", width=700, caption="Heart Attack Concept")
    st.write('''
    **Apa itu Heart Disease?**

    Heart disease adalah kondisi medis yang terjadi ketika terdapat kerusakan pada jantung dan pembuluh darah di sekitarnya.
    dan penyakit ini merupakan penyebab kematian nomor satu secara global dengan 17,9 
    juta kasus kematian setiap tahunnya. Penyakit jantung disebabkan oleh hipertensi, obesitas, dan gaya hidup yang 
    tidak sehat. Deteksi dini penyakit jantung perlu dilakukan pada kelompok risiko tinggi agar dapat segera mendapatkan 
    penanganan dan pencegahan. Project ini bertujuan untuk memprediksi apakah seseorang memiliki penyakit jantung atau 
    tidak berdasarkan beberapa kriteria tertentu. Dataset yang digunakan adalah dataset penyakit jantung dari [UCI 
    Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease).
    ''')
    st.write('''
    **Project Objective** 

    Tujuan dari project ini adalah untuk membuat model machine learning yang dapat memprediksi apakah seseorang memiliki penyakit jantung atau tidak berdasarkan
    beberapa kriteria tertentu.
    ''')

elif nav == "Dataset":
    st.title("Dataset")
    st.write('''
     **Dataset Overview**

    Dataset yang digunakan adalah dataset penyakit jantung dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease).
    Dataset ini memiliki 303 baris dan 14 kolom. Kolom target adalah kolom `target` yang menunjukkan apakah seseorang
    memiliki penyakit jantung atau tidak. Jika memiliki penyakit jantung, maka nilai kolom `target` adalah 1, jika tidak
    memiliki penyakit jantung, maka nilai kolom `target` adalah 0.
    ''')
    st.write('''
    **Dataset Description**
    
    Berikut adalah deskripsi dari dataset yang digunakan.
    
    1. `age` : usia dalam tahun (umur)
    2. `sex` : jenis kelamin (1 = laki-laki; 0 = perempuan)
    3. `cp` : tipe nyeri dada
        - 0: typical angina
        - 1: atypical angina
        - 2: non-anginal pain
        - 3: asymptomatic
    4. `trestbps` : tekanan darah istirahat (dalam mm Hg saat masuk ke rumah sakit)
    5. `chol` : serum kolestoral dalam mg/dl
    6. `fbs` : gula darah puasa > 120 mg/dl (1 = true; 0 = false)
    7. `restecg` : hasil elektrokardiografi istirahat
        - 0: normal
        - 1: memiliki ST-T wave abnormalitas (T wave inversions and/or ST elevation or depression of > 0.05 mV)
        - 2: menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri menurut kriteria Estes
    8. `thalach` : detak jantung maksimum yang dicapai
    9. `exang` : angina yang diinduksi oleh olahraga (1 = yes; 0 = no)
    10. `oldpeak` : ST depression yang disebabkan oleh olahraga relatif terhadap istirahat
    11. `slope` : kemiringan segmen ST latihan puncak
        - 1: naik
        - 2: datar
        - 3: turun
    12. `ca` : jumlah pembuluh darah utama (0-3) yang diwarnai dengan flourosopy
    13. `thal` : 3 = normal; 6 = cacat tetap; 7 = cacat yang dapat dibalik
    14. `target` : memiliki penyakit jantung atau tidak (1 = yes; 0 = no)
    ''')
    # show dataset
    st.write('''
    **Show Dataset**
    ''')
    st.dataframe(df.head())

    # show dataset shape
    st.write(f'''**Dataset Shape:** {df.shape}''')

    # show dataset description
    st.write('''
    **Dataset Description**
    ''')
    st.dataframe(df.describe())

    # show dataset count visualization
    st.write('''
    **Dataset Count Visualization**
    ''')
    views = st.selectbox("Select Visualization", ("", "Target", "Age", "Sex", "Cp", "Fbs", 'Restecg', 'Exang', 'Slope'))
    if views == "Target":
        st.bar_chart(df.target.value_counts())
        st.write('''
        `Target` adalah kolom yang menunjukkan apakah seseorang memiliki penyakit jantung atau tidak. Jika memiliki penyakit
        jantung, maka nilai kolom `target` adalah 1, jika tidak memiliki penyakit jantung, maka nilai kolom `target` adalah 0.
        Berdasarkan visualisasi di atas, dapat dilihat bahwa jumlah orang yang memiliki penyakit jantung lebih banyak daripada
        yang tidak memiliki penyakit jantung sejumlah 526 orang dibandingkan 499 orang.
        ''')
        st.write('Persentase antara memiliki penyakit jantung (1) dan tidak memiliki penyakit jantung (0) ', df.target.value_counts()/len(df) * 100)
    elif views == "Age":
        st.bar_chart(df['age'].value_counts())
        st.write('''
        Berdasarkan visualisasi di atas, dapat dilihat bahwa jumlah orang yang memiliki penyakit jantung paling banyak berada
        pada usia 58 tahun sebanyak 68 orang. Sedangkan jumlah orang yang tidak memiliki penyakit jantung paling banyak berada
        rentang 74-76 tahun sebanyak 9 orang.''')
    
    elif views == "Sex":
        st.bar_chart(df['sex'].value_counts())
        st.write('''
        Berdasarkan visualisasi di atas, dapat dilihat bahwa jumlah laki-laki (1) yang memiliki penyakit jantung lebih banyak
        daripada perempuan (0) yang memiliki penyakit jantung sejumlah 312 orang dibandingkan 713 orang.
        ''')
        st.write('Persentase antara Perempuan (0) dan Laki-Laki (1) ', df.sex.value_counts()/len(df) * 100)
    
    elif views == "Cp":
        st.bar_chart(df['cp'].value_counts())
        st.write('''
        Berdasarkan visualisasi di atas, dapat dilihat bahwa jumlah orang yang memiliki penyakit jantung paling banyak berada
        pada tipe nyeri dada 0 (Typical angina) sebanyak 497 orang, pada tipe nyeri dada 1 (Atypical angina) sebanyak 167 orang, pada tipe nyeri dada 2 (Non Anginal pain) sebanyak 284 orang, dan pada tipe nyeri dada 3 (Asymptomatic) sebanyak 77 orang.
         Sedangkan jumlah orang yang tidak memiliki penyakit jantung paling banyak
        berada pada tipe nyeri dada 0 sebanyak 39 orang.
        ''')
        st.write('Persentase antara tipe nyeri dada 0 (Typical angina), 1 (Atypical angina), 2 (Non Anginal pain), dan 3 (Asymptomatic) ', df.cp.value_counts()/len(df) * 100)
    elif views == 'Fbs':
        st.bar_chart(df['fbs'].value_counts())
        st.write('''
        Berdasarkan visualisasi di atas, dapat dilihat bahwa jumlah orang yang memiliki penyakit jantung paling banyak berada
        pada gula darah puasa > 120 mg/dl (1) sebanyak 153 orang. Sedangkan jumlah orang yang tidak memiliki penyakit jantung
        paling banyak berada pada gula darah puasa <= 120 mg/dl (0) sebanyak 872 orang.
        ''')
        st.write('Persentase antara gula darah puasa > 120 mg/dl (1) dan <= 120 mg/dl (0) ', df.fbs.value_counts()/len(df) * 100)

    elif views == 'Restecg':
        st.bar_chart(df['restecg'].value_counts())
        st.write('''
        Berdasarkan visualisasi di atas, dapat dilihat bahwa jumlah orang yang memiliki penyakit jantung paling banyak berada
        pada hasil elektrokardiografi istirahat 1 (Memiliki ST-T wave abnormalitas) sebanyak 513 orang. Sedangkan jumlah orang
        yang tidak memiliki penyakit jantung paling banyak berada pada hasil elektrokardiografi istirahat 0 (Normal) sebanyak
        497 orang dan pada hasil elektrokardiografi istirahat 2 (Menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri menurut kriteria Estes) sebanyak 15 orang,
        ''')
        st.write('Persentase antara hasil elektrokardiografi istirahat 0 (Normal), 1 (Memiliki ST-T wave abnormalitas), dan 2 (Menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri menurut kriteria Estes) ', df.restecg.value_counts()/len(df) * 100)

    elif views == 'Exang':
        st.bar_chart(df['exang'].value_counts())
        st.write('''
        Berdasarkan visualisasi di atas, dapat dilihat bahwa jumlah orang yang memiliki penyakit jantung paling banyak berada
        pada angina yang diinduksi oleh olahraga (1) sebanyak 345 orang. Sedangkan jumlah orang yang tidak memiliki penyakit
        jantung paling banyak berada pada angina yang diinduksi oleh olahraga (0) sebanyak 680 orang.
        ''')
        st.write('Persentase antara angina yang diinduksi oleh olahraga (1) dan tidak diinduksi oleh olahraga (0) ', df.exang.value_counts()/len(df) * 100)
    
    elif views == 'Slope':
        st.bar_chart(df['slope'].value_counts())
        st.write('''
        Berdasarkan visualisasi di atas, dapat dilihat bahwa jumlah orang yang memiliki penyakit jantung paling banyak berada
        pada kemiringan segmen ST latihan puncak 1 (Datar) dan  2 (Turun) sebanyak 482 orang dan 469 orang. Sedangkan jumlah orang yang tidak memiliki
        penyakit jantung paling banyak berada pada kemiringan segmen ST latihan puncak 0 (naik) sebanyak 309 orang.
        ''')
        st.write('Persentase antara kemiringan segmen ST latihan puncak 0 (Naik), 1 (Datar), dan 2 (Turun) ', df.slope.value_counts()/len(df) * 100)


    elif nav == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")
        st.write('''
        **Data Cleaning**

        Pada tahap ini kita akan melakukan data cleaning, data cleaning adalah proses mengubah data mentah menjadi data yang dapat digunakan untuk analisis.
        contohnya melakukan pengecekan terhadap missing value, duplikasi, outlier, korelasi dan sebagainya. Jika terdapat data yang
        kosong, maka data tersebut akan dihapus atau diisi dengan nilai lain. Jika terdapat data yang duplikat, maka data tersebut 
        akan dihapus. Jika terdapat data yang outlier, maka data tersebut akan dihapus atau menggunakan teknik np.log.
        ''')
        st.write('''
        Informasi yang akan kita gali adalah feature pada kesalahan penulisan:
        1. Feature `CA`: Memiliki 5 nilai dari rentang 0-4, maka dari itu nilai 4 diubah menjadi NaN (karena seharusnya tidak ada)
        2. Feature `thal`: Memiliki 4 nilai dari rentang 0-3, maka dari itu nulai 0 diubah menjadi NaN (karena seharusnya tidak ada)
        ''')
        views = st.radio("Show Data CA and Thal", ("CA", "Thal"))
        if views == "CA":
            st.write('''
            **Feature CA**
            
            Feature CA memiliki 5 nilai dari rentang 0-4, maka dari itu nilai 4 diubah menjadi NaN (karena seharusnya tidak ada)
            ''')
            st.dataframe(df.ca.value_counts().to_frame().transpose())
            st.write('''
            **Show Data After Cleaning**
            ''')
            st.dataframe(df.ca.replace(4, np.nan).value_counts().to_frame().transpose())
            
        elif views == "Thal":
            st.write('''
            **Feature Thal**
            
            Feature Thal memiliki 4 nilai dari rentang 0-3, maka dari itu nulai 0 diubah menjadi NaN (karena seharusnya tidak ada)
            ''')
            st.dataframe(df.thal.value_counts().to_frame().transpose())
            st.write('''
            **Show Data After Cleaning**
            ''')
            st.dataframe(df.thal.replace(0, np.nan).value_counts().to_frame().transpose())
            
        st.write('''
        Kita juga akan menggali informasi dari korelasi antar feature, berikut adalah hasil korelasi antar feature.
        ''')
        views_DR = st.radio("**Feature Selection and Feature extraction**", ("Correlation Value", "PCA"))
        if views_DR == "Correlation Value":
            st.write('''
            **Correlation Value (Feature Selection)**
            
            Berikut adalah hasil korelasi antar feature.
            ''')
            st.dataframe(df.corr())
            cor_matrix = df.corr()
            st.write('''
            Pengurutan korelasi antar feature terhadap target.
            ''')
            st.dataframe(cor_matrix['target'].sort_values())
            st.write('''
            Korelasi target(penyakit jantung) dengan variabel lainnya. 
            Korelasi positif dengan variabel tertentu berarti semakin tinggi variabel tersebut maka akan semakin tinggi 
            juga kemungkinan terkena penyakit jantung, sedangkan korelasi negatif ialah semakin rendah nilai variabel tersebut
            maka kemungkinan terkena penyakit jantung lebih tinggi.

            - `ca` -0.456989 (Korelasi Negatif Kuat)
            - `oldpeak` -0.434108 (Korelasi Negatif Kuat)
            - `exang` -0.431599 (Korelasi Negatif Kuat)
            - `thal` -0.370759 (Korelasi Negatif Kuat)
            - `sex` -0.318896 (Korelasi Negatif Kuat)
            - `age` -0.222416 (Korelasi Negatif)
            - `trestbps` -0.115614 (Korelasi Negatif Lemah)
            - `chol` -0.0105627 (Korelasi Negatif Lemah)
            - `fbs` 0.027210 (Korelasi Positif Lemah)
            - `restecg` 0.171453 (Korelasi Positif Lemah)
            - `slope` 0.326473 (korelasi Positif Kuat)
            - `cp` 0.422559 (korelasi Positif Kuat)
            - `thalach` 0.432211 (korelasi Positif Kuat)

            Jadi, dari data korelasi diatas faktor yang paling berpengaruh terhadap penyakit jantung ialah, sebagai berikut:

            - `ca` (semakin banyak major vessels, maka akan semakin tinggi resiko terkena penyakit jantung)
            - `oldpeak` (Semakin rendah depresi ST yang disebabkan oleh latihan relatif terhadap istirahat, maka resiko terkena penyakit jantung akan semakin tinggi)
            - `exang` (Apibila exercise induced angina rendah, maka resiko terkena penyakit jantung akan semakin tinggi)
            - `thal` (semakin rendah tipe jenis defek jantung, maka resiko terkena penyakit jantung semakin tinggi)
            - `sex` (Perempuan memiliki resiko terkena penyakit jantung lebih tinggi dibandingkan laki-laki)
            - `age` (semakin muda umur, ternyata semakin tinggi terkena penyakit jantung)
            - `slope` (Semakin tinggi kemiringan segmen latihan ST maka, resiko terkena penyakit jantung semakin tinggi)
            - `cp` (Semakin tinggi tipe Jenis rasa sakit pada dada, maka resiko terkena penyakit jantung semakin tinggi)
            - `thalach` (semakin tinggi detak jantung maksimum yang dicapai pasien selama tes latihan, maka resiko terkena penyakit jantung semakin tinggi)
            ''')
            st.write('''
            **Correlation Heatmap**
            
            Berikut adalah hasil korelasi antar feature.
            ''')
            st.image("https://drive.google.com/uc?id=1cFri-3vAHWqj0HCsCZNksarhw5QaUkTZ", width=700)
            st.write('''
            **Kesimpulan :**

            1) 'cp', 'thalach', dan 'slope' berkorelasi positif cukup kuat dengan 'target'.

            2) 'oldpeak', 'exang', 'ca', 'thal', 'sex', dan 'age' berkorelasi cukup kuat dengan 'target'.

            3) 'fbs', 'chol', 'trestbps', dan 'restecg' memiliki korelasi yang lemah dengan 'target'.

            Feature yang dipilih yaitu: 'cp', 'thalach', 'slope', 'oldpeak', 'exang', 'ca', 'thal', 'sex', dan 'age' untuk dianalisa lebih lanjut.
            ''')
        elif views_DR == "PCA":
            st.write('''**PCA (Feature Extraction)** 
            Sebelum melakukan PCA, kita akan melakukan scaling terlebih dahulu. Karena PCA membutuhkan data yang sudah di scaling.
            sesudah itu kita akan membuat eigen value analysis dan Cumulative Explained Variance untuk mengetahui berapa banyak komponen yang akan kita gunakan.
            ''')
            st.write('''**Graph cummulative explained variance**''')
            st.image("https://drive.google.com/uc?id=1Rv5nCXEeeCDesOPjRJlmh4uLtt-iSfZB", width=700)
            st.write('''Dari plot, kita dapat melihat bahwa 9 komponen utama pertama menjelaskan sekitar 90% variasi. Berdasarkan grafik ini, 
            kita dapat memutuskan berapa banyak komponen utama yang ingin kita miliki tergantung pada variabilitas yang dijelaskan.''')
            st.write('''**Eigen Value Analysis**''')
            st.image("https://drive.google.com/uc?id=1s43MppdOHSbwuJgsIqc1B4-htbNPqDyq", width=700)
            st.write('''Pada awalnya, setiap komponen utama memberikan kontribusi yang signifikan terhadap total varians. 
            Namun, saat Anda bergerak ke komponen utama berikutnya, penurunan dalam varians yang dijelaskan oleh setiap komponen mungkin mulai berkurang.
            Poin "patah" atau "puncak" pada Scree Plot menunjukkan titik di mana penurunan ini melambat secara signifikan, 
            dan komponen berikutnya memberikan kontribusi yang lebih kecil terhadap total varians''')

        st.write('''
        ***Kesimpulan :***
        Dimensi Reduksi yang dipilih adalah 9 dimensi, karena pada feature selection / feature extraction
        9 dimensi tersebut dapat menjelaskan 90% varians dari data yang ada dan memiliki korelasi yang cukup kuat dengan target.
        ''')

elif nav == "Modelling":
    st.header("Modelling")
    var = st.selectbox("Select a model", ("Before Tuning", "After Tuning", "Roc-Auc", "Tresholds", "Kesimpulan"))
    var = st.selectbox("Select a model", ("Before Tuning", "Before Tuning with Smote","After Tuning","After Tuning With Smote", "Roc-Auc", "Tresholds", "Kesimpulan"))
    if var == "Before Tuning without Smote":
        accuracy_score_nosm = {
            'Logistic Regression': 0.8070175438596491,
            'Decision Tree': 0.7543859649122807,
            'Random Forest': 0.8070175438596491,
            'MLP Classifier': 0.7719298245614035,
        }
        st.write('''
        **Model Before Tuning**
        
        Berikut adalah hasil akurasi dari model sebelum dilakukan tuning tanpa SMOTE.
        ''')
        st.dataframe(pd.DataFrame(accuracy_score_nosm.items(), columns=['Model', 'Accuracy Score']))
        st.write('''
        Berdasarkan hasil akurasi dari model sebelum dilakukan tuning, dapat dilihat bahwa model dengan akurasi tertinggi
        adalah Logistic Regression dan random forest dengan akurasi 0.8070175438596491
        ''') 
    elif var == "Before Tuning with Smote":
        accuracy_score_sm = {
            'Logistic Regression': 0.7894736842105263,
            'Decision Tree':  0.8245614035087719,
            'Random Forest': 0.8245614035087719,
            'MLP Classifier': 0.7894736842105263,
        }
        st.write('''
        **Model Before Tuning With Smote**
        
        Berikut adalah hasil akurasi dari model sebelum dilakukan tuning menggunakan SMOTE.
        ''')
        st.dataframe(pd.DataFrame(accuracy_score_sm.items(), columns=['Model', 'Accuracy Score']))
        st.write('''
        Berdasarkan hasil akurasi dari model sebelum dilakukan tuning tapi menggunakan SMOTE, dapat dilihat bahwa model dengan akurasi tertinggi
        adalah decision tree dengan akurasi 0.8245614035087719
        ''')
    elif var == "After Tuning Without Smote":
        accuracy_score_nosm_tune = {
            'Logistic Regression': 0.8070175438596491,
            'Decision Tree': 0.7543859649122807,
            'Random Forest': 0.8596491228070176,
            'MLP Classifier': 0.8421052631578947,
        }
        st.write('''
        **Model After Tuning without Smote**
        
        Berikut adalah hasil akurasi dari model setelah dilakukan tuning tanpa menggunakan SMOTE.
        ''')
        st.dataframe(pd.DataFrame(accuracy_score_nosm_tune.items(), columns=['Model', 'Accuracy Score']))
        st.write('''
        Berdasarkan hasil akurasi dari model setelah dilakukan tuning, dapat dilihat bahwa model dengan akurasi tertinggi
        adalah Random Forest dengan akurasi 0.8596491228070176.
        ''')
    elif var == "After Tuning with Smote":
        accuracy_score_sm_tune = {
            'Logistic Regression': 0.7894736842105263,
            'Decision Tree': 0.6842105263157895,
            'Random Forest': 0.7894736842105263,
            'MLP Classifier': 0.8070175438596491,
        }
        st.write('''
        **Model After Tuning with SMOTE**
        
        Berikut adalah hasil akurasi dari model setelah dilakukan tuning tapi menggunakan SMOTE.
        ''')
        st.dataframe(pd.DataFrame(accuracy_score_sm_tune.items(), columns=['Model', 'Accuracy Score']))
        st.write('''
        Berdasarkan hasil akurasi dari model setelah dilakukan tuning menggunakan smote, dapat dilihat bahwa model dengan akurasi tertinggi
        adalah MLP dengan akurasi 0.8070175438596491.
        ''')
    elif var == "Roc-Auc":
        st.write('''
        **ROC-AUC**
        
        Berikut adalah hasil ROC-AUC dari model sebelum dan setelah dilakukan tuning.
        ''')
        image_url = "https://drive.google.com/uc?id=1LMIv3O4QdBgLV6BLLmLpCMSALN9AkVDn"
        # Display the image using st.image
        st.image(image_url, width=700)
        st.write('''
        Berdasarkan hasil ROC-AUC dari model sebelum dan setelah dilakukan tuning, dapat dilihat bahwa model dengan
        ROC-AUC tertinggi adalah Random Forest dengan ROC-AUC 0.91.
        ''')
    elif var == "Tresholds":
        st.write('''
        **Tresholds**
        
        Berikut adalah hasil treshold terbaik dari model Logistic Regression, Decision Tree, Random Forest, dan MLP.
        ''')
        image_url = "https://drive.google.com/uc?id=1qEqaYUZfHYAl0W8xj44F5FKMLhtdKLQT"
        
        # Display the image using st.image
        st.image(image_url, width=700)
        st.write('''
        Karena pada kasus ini kita ingin memprediksi mengurangi kesalahan dalam memprediksi kasus negatif sebagai positif (False Positive), 
        maka kita akan memilih treshold yang lebih tinggi yaitu Random Forest dengan treshold 0.42. 
        Namun, ini dapat mengurangi sensitivitas model (menyebabkan lebih banyak True Negative yang salah diprediksi negatif)
        ''')

    elif var == "Kesimpulan":
        st.write('''
        **Kesimpulan**
        
        Berdasarkan hasil akurasi dari model sebelum dan setelah dilakukan tuning, dapat disimpulkan bahwa model dengan
        akurasi tertinggi adalah Random Forest Classifier dengan akurasi 0.86. Jika kamu ingin mendownload model ini, maka kamu akan mendapatkan
        di link berikut ini [Download Model](https://drive.google.com/file/d/1k6fhTvDvUN3-2LBgST8UVO_kDyxt2kj9/view?usp=sharing).
        ''')

elif nav == 'Prediction':
    st.header("My Apps Prediction")
    st.write('''
    Aplikasi prediksi ini menggunakan model Random Forest Classsifier, untuk pengaplikasiannya adalah anda harus mengisi feature yang ada pada sidebar.
    Hal ini bertujuan untuk memprediksi apakah seseorang memiliki penyakit jantung atau tidak berdasarkan beberapa kriteria tertentu. 
    ''')
    heart()

elif nav == "About":
    st.title("About Me")
    st.image("https://drive.google.com/uc?id=1vl7B15t1n9jBqM3gTgZAkdGrA6KChYCb", width=200)
    st.write('''
    **William Deli**
    
    Saya adalah mahasiswa di Binus University. Saya mengambil jurusan Computer Engineering. Saya tertarik dengan bidang Machine Learning dan AI.
    ini adalah aplikasi prediksi yang saya buat menggunakan streamlit. Semoga aplikasi ini dapat bermanfaat bagi kalian semua.
    jika ada pertanyaan / saran dan kritik silahkan hubungi saya di link berikut ini [Contact Me](https://www.linkedin.com/in/william-deli-9b7a1b1b0/).
    ''')
    st.write('''
    **Contact Me**
    
    - [LinkedIn](https://www.linkedin.com/in/william-deli)
    - [Github](https://github.com/williamdeli)
    - [Instagram](https://www.instagram.com/william_deli)
    ''')
    st.write('''
    Selain membuat project machine learning dan AI, saya juga membuat project lainnya seperti: IoT(Internet of Things), 
    Flutter mobile apps, Digital Signal Processing, dan lainnya. Jika kamu ingin melihat project lainnya, kamu bisa mengunjungi dibawah ini.
    ''')
    select_item = st.selectbox("My Project Preview", ('', 'Project link'))
    if select_item == "Project link":
        '[Other Project see here](https://linktr.ee/William_deli)',
    # elif select_item == "Housing Price Prediction":
    #     st.header("Ini Contoh Housing Price Prediction")
