import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Risiko Diabetes",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    try:
        # Memuat model 'best_rf' dan 'scaler' yang baru saja diexport
        rf_model = pickle.load(open('diabetes_rf.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return rf_model, scaler
    except Exception as e:
        st.error(f"Gagal memuat model/scaler: {e}")
        return None, None

rf_model, scaler = load_models()

# --- HEADER UI ---
st.title("🩺 Aplikasi Prediksi Risiko Diabetes")
st.markdown("""
Aplikasi ini menggunakan model **Machine Learning (Random Forest Classifier)** untuk menganalisis kemungkinan risiko diabetes berdasarkan **8 Parameter Klinis & Gaya Hidup** pasien.
""")
st.divider()

# --- INPUT FORM ---
with st.form("prediction_form"):
    st.subheader("📝 Masukkan Data Pasien")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🧑‍⚕️ Profil & Gaya Hidup")
        age = st.number_input("Usia (Tahun)", min_value=1, max_value=120, value=45)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=0, max_value=60.0, value=25.0, step=0.1)
        physical_activity = st.number_input("Aktivitas Fisik (Menit/Minggu)", min_value=0, max_value=1000, value=150)
        
    with col2:
        st.markdown("#### 🩸 Metrik Medis / Hasil Laboratorium")
        hba1c = st.number_input("HbA1c Level (%)", min_value=0, max_value=15.0, value=5.5, step=0.1)
        glucose_fasting = st.number_input("Gula Darah Puasa (mg/dL)", min_value=0, max_value=300.0, value=100.0, step=1.0)
        glucose_post = st.number_input("Gula Darah Pasca Makan / 2 Jam PP (mg/dL)", min_value=50.0, max_value=400.0, value=140.0, step=1.0)
        triglycerides = st.number_input("Trigliserida (mg/dL)", min_value=0, max_value=500.0, value=150.0, step=1.0)
        insulin = st.number_input("Kadar Insulin (μIU/mL)", min_value=0.0, max_value=300.0, value=10.0, step=1.0)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🚀 Analisis Risiko Sekarang", use_container_width=True)

# --- LOGIKA PREDIKSI ---
if submitted:
    if rf_model and scaler:
        with st.spinner('Menghitung parameter medis...'):
            # Susun urutan data sesuai persis dengan selected_features di Notebook kamu
            input_data = np.array([[
                hba1c,
                glucose_post,
                glucose_fasting,
                physical_activity,
                age,
                bmi,
                triglycerides,
                insulin
            ]])
            
            try:
                # 1. Lakukan transformasi scaling
                input_scaled = scaler.transform(input_data)
                
                # 2. Lakukan Prediksi dan hitung Probabilitas
                prediction = rf_model.predict(input_scaled)[0]
                probability = rf_model.predict_proba(input_scaled)[0][1]
                
                # --- MENAMPILKAN HASIL ---
                st.divider()
                st.subheader("🎯 Hasil Analisis Risiko")
                
                # Komponen visual Score/Metric metric
                st.metric(label="Probabilitas Terdiagnosis Diabetes", value=f"{probability:.2%}")
                
                if prediction == 1:
                    st.error(f"⚠️ **Terdiagnosis / Risiko Tinggi Diabetes.** \n\nSistem mengidentifikasi indikator kesehatan pasien mengarah ke kondisi diabetes. Sangat disarankan untuk segera melakukan konsultasi lebih lanjut dengan dokter spesialis.")
                else:
                    st.success(f"✅ **Tidak Terdiagnosis / Risiko Rendah.** \n\nBerdasarkan data klinis saat ini, indikator berada pada ambang batas aman. Tetap pertahankan gaya hidup sehat, pola makan bergizi, dan olahraga teratur!")
                
                # Edukasi Tambahan
                st.info("💡 *Catatan Medis Umum: Batas normal HbA1c berada di bawah 5.7%, Gula Darah Puasa di bawah 100 mg/dL, dan Gula Darah Pasca Makan di bawah 140 mg/dL.*")
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")