# Standard library imports
from collections import OrderedDict
from io import BytesIO
import time

# Third-party numerical and data processing
import numpy as np
import pandas as pd
from scipy.stats import chi2, kstest

# Statistical modeling
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from linearmodels.system import SUR

# Machine learning metrics
from sklearn.metrics import mean_squared_error, r2_score

# Geolocation
from geopy.geocoders import Nominatim

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Streamlit
import streamlit as st

# =========================
# Konfigurasi halaman
# =========================
st.set_page_config(
    page_title="Prediksi Harga Cabai Merah Besar",
    page_icon="🌶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling dengan dark/light mode support
st.markdown("""
<style>
    /* Header styling dengan adaptasi dark/light mode */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #D62828;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #2A9D8F;
        margin-bottom: 2rem;
    }
    
    /* Card styling dengan adaptasi theme */
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: var(--background-color);
        border: 1px solid var(--border-color);
        margin: 10px 0;
    }
    
    /* Light mode - Biru terang dengan contrast tinggi */
    [data-testid="stAppViewContainer"] {
        --card-bg-start: #e3f2fd;
        --card-bg-end: #bbdefb;
        --card-border: #1976d2;
        --card-text: #1a237e;
        --card-shadow: rgba(25, 118, 210, 0.15);
    }
    
    /* Dark mode - Biru gelap dengan contrast tinggi */
    [data-testid="stAppViewContainer"][data-theme="dark"] {
        --card-bg-start: #1a3a52;
        --card-bg-end: #0d47a1;
        --card-border: #64b5f6;
        --card-text: #e3f2fd;
        --card-shadow: rgba(100, 181, 246, 0.2);
    }
    
    .card {
        background: linear-gradient(135deg, var(--card-bg-start) 0%, var(--card-bg-end) 100%);
        border: 2px solid var(--card-border);
        color: var(--card-text);
        box-shadow: 0 2px 8px var(--card-shadow);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px var(--card-shadow);
    }
    
    .card h3 {
        color: var(--card-text);
        margin-bottom: 10px;
    }
    
    .card p {
        color: var(--card-text);
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: #D62828;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #B02020;
    }
    
    /* Image container untuk kontrol ukuran */
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
    }
    .image-container img {
        max-width: 600px;
        width: 100%;
        height: auto;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Sidebar & Navigasi
# =========================
pages = ["Home", "Data & Analisis", "Pemodelan GSTAR-SUR", "Hasil Prediksi"]

if "page_index" not in st.session_state:
    st.session_state.page_index = 0

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f336.png", width=100)
    st.title("Navigasi")
    
    menu = st.radio(
        "Pilih Halaman:",
        pages,
        index=st.session_state.page_index,
        key="menu_radio"
    )
    
    # Update page index based on radio selection
    if st.session_state.menu_radio:
        st.session_state.page_index = pages.index(st.session_state.menu_radio)
    
    st.markdown("---")
    
    # Tampilkan info model terbaik jika sudah ada
    if 'best_model_name' in st.session_state:
        st.success("**Model Terbaik:**")
        st.write(f"**{st.session_state['best_model_name']}**")
        if 'differenced' in st.session_state and st.session_state['differenced']:
            st.caption("I(1) - Differencing 1 kali")
        else:
            st.caption("I(0) - Data asli")
    
    st.markdown("---")
    st.info("**Tips:** Unggah data CSV/Excel dengan format:\n\n`Tanggal, Kota1, Kota2, ...`")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <small>
    <b>© 2025</b><br>
    Sandria Amelia Putri<br>
    UPN Veteran Jawa Timur
    </small>
    """, unsafe_allow_html=True)

# =========================
# FUNGSI-FUNGSI UTILITY
# =========================

def create_gstar_variables(df, cities, W, max_lag=3):
    """Membuat variabel lag temporal dan weighted spatial lag untuk model GSTAR"""
    gstar = df.copy()
    n = len(cities)
    
    if isinstance(W, pd.DataFrame):
        W = W.values
    
    # Lag temporal
    for lag in range(1, max_lag + 2):
        for city in cities:
            gstar[f'lag{lag}_{city}'] = gstar[city].shift(lag)
    
    # Weighted spatial lag
    for lag in range(1, max_lag + 1):
        for i, city in enumerate(cities):
            wlag = np.zeros(len(df))
            for j in range(n):
                if i != j:
                    lag_diff = (gstar[f'lag{lag}_{cities[j]}'] - gstar[f'lag{lag+1}_{cities[j]}']).values
                    wlag += W[i, j] * lag_diff
            gstar[f'wlag{lag}_{city}'] = wlag
    
    return gstar

def backward_gstar_sur(train_data, cities, max_lag=3, threshold=0.05, progress_bar=None):
    """Backward elimination untuk model GSTAR-SUR"""
    included = {
        city: [f'lag{j}_{city}' for j in range(1, max_lag + 1)] + 
              [f'wlag{j}_{city}' for j in range(1, max_lag + 1)]
        for city in cities
    }
    
    iteration = 0
    elimination_log = []
    
    while True:
        iteration += 1
        if progress_bar:
            progress_bar.progress(min(iteration * 10, 100))
        
        sur_data = OrderedDict()
        for city in cities:
            sur_data[f"g_{city.lower()}"] = {
                "dependent": train_data[city],
                "exog": train_data[included[city]]
            }
        
        model = SUR(sur_data).fit(cov_type="unadjusted")
        
        pvalues = {}
        for city in cities:
            prefix = f"g_{city.lower()}_"
            city_pvals = model.pvalues[model.pvalues.index.str.startswith(prefix)]
            for var_full, pval in city_pvals.items():
                var_name = var_full.replace(prefix, "")
                pvalues[(city, var_name)] = pval
        
        worst_var, worst_pval = max(pvalues.items(), key=lambda x: x[1])
        
        if worst_pval > threshold:
            worst_city, worst_name = worst_var
            included[worst_city].remove(worst_name)
            elimination_log.append(f"Iterasi {iteration}: Drop {worst_name} dari {worst_city} (p-value: {worst_pval:.4f})")
        else:
            break
    
    return included, model, elimination_log

def calculate_metrics(y_true, y_pred, n_params=None, n_obs=None):
    """Hitung metrik evaluasi model: MAPE, Adj. R², AIC, BIC"""
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    metrics = {'MAPE': mape}
    
    if n_params is not None and n_obs is not None:
        # Adjusted R²
        adj_r2 = 1 - (1 - r2) * (n_obs - 1) / (n_obs - n_params - 1)
        metrics['Adj. R²'] = adj_r2
        
        # AIC = n * ln(MSE) + 2 * k
        metrics['AIC'] = n_obs * np.log(mse) + 2 * n_params
        
        # BIC = n * ln(MSE) + k * ln(n)
        metrics['BIC'] = n_obs * np.log(mse) + n_params * np.log(n_obs)
    else:
        metrics['Adj. R²'] = r2
        metrics['AIC'] = np.nan
        metrics['BIC'] = np.nan
    
    return metrics

# =========================
# HOME PAGE
# =========================
if menu == "Home":
    # Header dengan emoji dan styling
    st.markdown('<p class="main-header">Sistem Prediksi Harga Cabai Merah Besar</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Menggunakan Model GSTAR-SUR untuk Wilayah Jawa Timur</p>', unsafe_allow_html=True)
    
    # Banner image cabai merah besar
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://images.unsplash.com/photo-1741210492202-b078dcab2387?q=80&w=1074&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", 
                 caption="Cabai Merah Besar", width=600)
    
    st.markdown("---")
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
        <h3>Analisis Data</h3>
        <p>Upload data harga cabai dalam format CSV/Excel dan lakukan analisis eksploratori lengkap dengan visualisasi interaktif.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
        <h3>Pemodelan GSTAR-SUR</h3>
        <p>Model spatio-temporal yang menangkap hubungan antar wilayah dan prediksi multi-lokasi secara bersamaan.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
        <h3>Prediksi Akurat</h3>
        <p>Prediksi harga dengan metrik evaluasi lengkap (MAPE, Adj. R², AIC, BIC) dan visualisasi interaktif.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Keunggulan
    st.subheader("Keunggulan Aplikasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Fleksibel**: Mendukung jumlah kota dinamis
        - **Komprehensif**: Analisis lengkap dari EDA hingga prediksi
        - **Interaktif**: Visualisasi dan parameter yang dapat disesuaikan
        """)
    
    with col2:
        st.markdown("""
        - **Akurat**: Model GSTAR-SUR dengan backward elimination
        - **User-friendly**: Interface intuitif dan mudah digunakan
        - **Export Ready**: Download hasil dalam format CSV & Excel
        """)
    
    st.markdown("---")
    
    # Cara Penggunaan
    with st.expander("Cara Menggunakan Aplikasi"):
        st.markdown("""
        ### Langkah-langkah:
        
        1. **Upload Data**
           - Klik menu "Data & Analisis"
           - Upload file CSV/Excel dengan format: `Tanggal, Kota1, Kota2, ...`
           - Contoh: `Tanggal, Malang, Banyuwangi, Surabaya`
        
        2. **Eksplorasi Data**
           - Pilih rentang tanggal yang ingin dianalisis
           - Lihat statistik deskriptif dan grafik tren harga
        
        3. **Jalankan Pemodelan**
           - Klik menu "Pemodelan GSTAR-SUR"
           - Sistem otomatis melakukan:
             * Uji stasioneritas
             * Identifikasi lag optimal
             * Pembobotan spasial
             * Model selection dengan backward elimination
        
        4. **Lihat Hasil Prediksi**
           - Klik menu "Hasil Prediksi"
           - Lihat prediksi harga ke depan (custom hari)
           - Download hasil dalam CSV/Excel
        """)
    
    # Call to action
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Mulai Prediksi Sekarang", key="start_btn"):
            st.session_state.page_index = 1
            st.rerun()
    
    st.markdown("---")
    
    # Info footer
    st.info("**Catatan**: Aplikasi ini dapat mengakomodasi data dari berbagai kabupaten/kota di Jawa Timur. Jumlah lokasi bersifat fleksibel sesuai data yang Anda upload.")

# =========================
# DATA & ANALISIS PAGE
# =========================
elif menu == "Data & Analisis":
    st.header("Data & Analisis Eksploratori")
    st.write("Upload data harga cabai untuk memulai analisis:")
    
    uploaded_file = st.file_uploader("Unggah File Data (CSV/Excel)", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    
    if uploaded_file.name.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        
            if df.shape[1] == 1:
                df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')
            
        except Exception:
            df = pd.read_csv(uploaded_file, sep=';', encoding='latin1')
    else:
        df = pd.read_excel(uploaded_file)
    
    if uploaded_file is not None:
        st.success(f"Berhasil mengunggah: {uploaded_file.name}")
        
        try:
            # Support untuk CSV dan Excel
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')
            else:
                df = pd.read_excel(uploaded_file)
            
            # Deteksi kolom tanggal
            guess_date = None
            common_date_names = ["Tanggal", "tanggal", "date", "Date", "tgl", "Tanggal_Data"]
            for c in df.columns:
                if c in common_date_names:
                    guess_date = c
                    break
            
            if guess_date is None:
                for c in df.columns:
                    try:
                        pd.to_datetime(df[c], errors="raise")
                        guess_date = c
                        break
                    except:
                        continue
            
            if guess_date is None:
                guess_date = df.columns[0]
            
            # Konversi tanggal
            df[guess_date] = pd.to_datetime(df[guess_date], errors="coerce")
            df = df.dropna(subset=[guess_date]).sort_values(by=guess_date)
            
            # Ambil daftar kota
            cities = [c for c in df.columns if c != guess_date]
            n_cities = len(cities)
            
            st.info(f"Terdeteksi **{n_cities} kota/kabupaten**: {', '.join(cities)}")
            
            # Preview data
            st.subheader("Preview Data")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Pilihan rentang waktu
            st.subheader("Pilih Rentang Waktu Analisis")
            min_date, max_date = df[guess_date].min(), df[guess_date].max()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Tanggal Mulai:", value=min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("Tanggal Akhir:", value=max_date, min_value=min_date, max_value=max_date)
            
            mask = (df[guess_date] >= pd.to_datetime(start_date)) & (df[guess_date] <= pd.to_datetime(end_date))
            filtered_df = df.loc[mask].set_index(guess_date)
            
            st.write(f"Menampilkan **{len(filtered_df)} observasi** dari **{start_date}** hingga **{end_date}**")
            
            # Simpan ke session state
            st.session_state["filtered_df"] = filtered_df
            st.session_state["cities"] = cities
            st.session_state["n_cities"] = n_cities
            
            # Statistik Deskriptif
            st.subheader("Statistik Deskriptif")
            st.dataframe(filtered_df[cities].describe(), use_container_width=True)
            
            # Visualisasi Tren
            st.subheader("Grafik Tren Harga per Wilayah")
            
            color_palette = ['#D62828', '#2A9D8F', '#1D4ED8', '#F4A261', '#264653', 
                           '#E76F51', '#8338EC', '#FF006E', '#FB5607', '#FFBE0B']
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            for i, city in enumerate(cities):
                color = color_palette[i % len(color_palette)]
                ax.plot(filtered_df.index, filtered_df[city], 
                       marker='o', label=city, color=color, linewidth=2)
            
            ax.set_title('Plot Time-Series Harga Cabai Merah Besar Harian', fontsize=16, fontweight='bold')
            ax.set_xlabel('Tanggal', fontsize=12)
            ax.set_ylabel('Harga /Kg (Rp)', fontsize=12)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(len(filtered_df)//10, 1)))
            plt.xticks(rotation=45, ha='right')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(title='Kab/Kota', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Korelasi Spearman
            st.subheader("Matriks Korelasi Spearman")
            corr_matrix = filtered_df[cities].corr(method='spearman')
            
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                       square=True, linewidths=1, ax=ax)
            ax.set_title('Korelasi Spearman Harga Antar Kota', fontsize=8, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.success("Data berhasil dimuat dan siap untuk pemodelan")
            
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
    
    else:
        st.warning("Belum ada file diunggah. Silakan unggah data harga dalam format CSV/Excel.")
        
        # Contoh format data
        with st.expander("Lihat Contoh Format Data"):
            example_data = pd.DataFrame({
                'Tanggal': pd.date_range('2025-01-01', periods=5),
                'Malang': [30000, 30500, 31000, 31500, 32000],
                'Surabaya': [29000, 29500, 30000, 30500, 31000],
                'Banyuwangi': [28000, 28500, 29000, 29500, 30000]
            })
            st.dataframe(example_data)
            st.code("""
# Format CSV/Excel:
Tanggal,Kota1,Kota2,Kota3,...
2025-01-01,30000,29000,28000
2025-01-02,30500,29500,28500
...
            """)

# =========================
# PEMODELAN GSTAR-SUR PAGE
# =========================
elif menu == "Pemodelan GSTAR-SUR":
    st.header("Pemodelan GSTAR-SUR")
    
    if 'filtered_df' not in st.session_state:
        st.warning("Silakan upload dan pilih data terlebih dahulu di tab **Data & Analisis**")
        st.stop()
    
    df = st.session_state['filtered_df']
    cities = st.session_state['cities']
    n_cities = st.session_state['n_cities']
    
    st.info(f"Membangun model untuk **{n_cities} kota**: {', '.join(cities)}")
    
    # Tab untuk proses modeling
    tabs = st.tabs(["Stasioneritas", "Lag Optimal", "Pembobot Spasial", "Estimasi Model", "Evaluasi"])
    
    # TAB 1: Stasioneritas
    with tabs[0]:
        st.subheader("Uji Stasioneritas (ADF Test)")
        
        def adf_test_trend(df, kolom_list, alpha=0.05):
            hasil = []
            for kolom in kolom_list:
                series = df[kolom].dropna()
                
                # Check if series is constant
                if series.nunique() <= 1 or series.std() == 0:
                    hasil.append({
                        'Kab_Kota': kolom,
                        'ADF-statistic': 'N/A',
                        'p-value': 'N/A',
                        'Critical Value (5%)': 'N/A',
                        'Kesimpulan': 'Data Konstan'
                    })
                    continue
                
                try:
                    result = adfuller(series, regression='ct', autolag='AIC')
                    adf_stat, p_value, usedlag, nobs, critical_values, icbest = result
                    kesimpulan = "Stasioner" if p_value < alpha else "Tidak Stasioner"
                    hasil.append({
                        'Kab_Kota': kolom,
                        'ADF-statistic': round(adf_stat, 4),
                        'p-value': round(p_value, 4),
                        'Critical Value (5%)': round(critical_values['5%'], 4),
                        'Kesimpulan': kesimpulan
                    })
                except ValueError as e:
                    hasil.append({
                        'Kab_Kota': kolom,
                        'ADF-statistic': 'Error',
                        'p-value': 'Error',
                        'Critical Value (5%)': 'Error',
                        'Kesimpulan': f'Error: {str(e)}'
                    })
            return pd.DataFrame(hasil)
        
        with st.spinner("Melakukan uji ADF..."):
            tabel_adf_asli = adf_test_trend(df, cities)
        
        st.write("**Hasil Uji ADF Data Asli:**")
        st.dataframe(tabel_adf_asli, use_container_width=True)
        
        # Tampilkan statistik data untuk diagnostik
        with st.expander("Diagnostik Data"):
            st.write(f"**Jumlah Observasi:** {len(df)} hari")
            st.write("**Statistik Variasi Data:**")
            
            stats_data = []
            for city in cities:
                stats_data.append({
                    'Kota': city,
                    'Mean': f"Rp {df[city].mean():,.0f}",
                    'Std Dev': f"Rp {df[city].std():,.0f}",
                    'Min': f"Rp {df[city].min():,.0f}",
                    'Max': f"Rp {df[city].max():,.0f}",
                    'Range': f"Rp {(df[city].max() - df[city].min()):,.0f}",
                    'CV (%)': f"{(df[city].std() / df[city].mean() * 100):.2f}%"
                })
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
            
            st.info("""
            **Interpretasi Coefficient of Variation (CV):**
            - CV < 5%: Variasi sangat rendah (data hampir konstan)
            - CV 5-15%: Variasi rendah-sedang
            - CV > 15%: Variasi cukup baik untuk modeling
            """)
        
        # Check for constant data or errors
        if any(tabel_adf_asli['Kesimpulan'].str.contains('Konstan|Error', na=False)):
            st.error("Beberapa data memiliki masalah (konstan atau error). Silakan periksa data input Anda.")
            st.warning("**Saran**: Pastikan data memiliki variasi yang cukup untuk setiap kota.")
            st.session_state['df_diff'] = df[cities]
            st.session_state['differenced'] = False
        elif any(tabel_adf_asli['Kesimpulan'] == 'Tidak Stasioner'):
            st.warning("Data tidak stasioner. Melakukan differencing...")
            df_diff = df[cities].diff().dropna()
            
            # Check if differenced data is valid
            if df_diff.isnull().all().all() or (df_diff.std() == 0).any():
                st.error("Data setelah differencing menjadi tidak valid. Menggunakan data asli.")
                
                # Tampilkan detail masalah
                st.write("**Detail Masalah Setelah Differencing:**")
                problem_cities = []
                for city in cities:
                    std_val = df_diff[city].std()
                    unique_vals = df_diff[city].nunique()
                    problem_cities.append({
                        'Kota': city,
                        'Std Dev': f"{std_val:.4f}",
                        'Unique Values': unique_vals,
                        'Status': '❌ Konstan' if std_val == 0 or unique_vals <= 1 else '✅ OK'
                    })
                st.dataframe(pd.DataFrame(problem_cities), use_container_width=True)
                
                st.warning("""
                **Kemungkinan Penyebab:**
                - Data terlalu pendek (minimal 50-100 observasi direkomendasikan)
                - Variasi harga sangat kecil (harga hampir tidak berubah)
                - Data memiliki pola yang sangat linear
                
                **Solusi:**
                - Gunakan data dengan periode lebih panjang
                - Pastikan data mencakup fluktuasi harga yang cukup
                - Model akan tetap jalan dengan data asli (non-difference)
                """)
                
                st.session_state['df_diff'] = df[cities]
                st.session_state['differenced'] = False
            else:
                tabel_adf_diff = adf_test_trend(df_diff, cities)
                st.write("**Hasil Uji ADF Setelah Differencing:**")
                st.dataframe(tabel_adf_diff, use_container_width=True)
                
                # Check if differencing helped
                if any(tabel_adf_diff['Kesimpulan'].str.contains('Konstan|Error', na=False)):
                    st.warning("Differencing menyebabkan data konstan. Menggunakan data asli.")
                    st.session_state['df_diff'] = df[cities]
                    st.session_state['differenced'] = False
                else:
                    st.success("Data berhasil distasionerkan dengan differencing")
                    st.session_state['df_diff'] = df_diff
                    st.session_state['differenced'] = True
        else:
            st.success("Semua data sudah stasioner")
            st.session_state['df_diff'] = df[cities]
            st.session_state['differenced'] = False
    
    # TAB 2: Lag Optimal
    with tabs[1]:
        st.subheader("Identifikasi Lag Optimal (VAR)")
        
        max_lags = st.slider("Tentukan jumlah lag maksimal:", 1, 7)
        
        if st.button("Hitung Lag Optimal"):
            with st.spinner("Menghitung..."):
                df_var = st.session_state['df_diff'].dropna()
                model = VAR(df_var)
                lag_order_results = model.select_order(maxlags=max_lags)
                
                aic_values = lag_order_results.ics['aic']
                min_aic = min(aic_values)
                optimal_lag = lag_order_results.aic
                
                lag_results = []
                for lag, aic_value in enumerate(aic_values):
                    is_optimal = "Ya" if aic_value == min_aic else ""
                    lag_results.append({
                        "Lag": lag,
                        "AIC": f"{aic_value:.4f}",
                        "Optimal": is_optimal
                    })
                
                lag_df = pd.DataFrame(lag_results)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.dataframe(lag_df, use_container_width=True)
                with col2:
                    st.metric("Lag Optimal", optimal_lag, f"AIC: {min_aic:.4f}")
                
                st.session_state['optimal_lag'] = optimal_lag
                st.success(f"Lag optimal: **{optimal_lag}**")
    
    # TAB 3: Pembobot Spasial
    with tabs[2]:
        st.subheader("Matriks Pembobot Spasial")
        
        if st.button("Buat Matriks Pembobot Korelasi Silang"):
            with st.spinner("Membuat matriks pembobot..."):
                n = len(cities)
                k = st.session_state.get('optimal_lag', 3)
                
                # Cross-correlation
                cross = pd.DataFrame(0.0, index=cities, columns=cities)
                
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            Zi = df[cities[i]][k:].values
                            Zj = df[cities[j]][:-k].values
                            num = np.sum((Zi - Zi.mean()) * (Zj - Zj.mean()))
                            den = np.sqrt(np.sum((Zi - Zi.mean())**2) * np.sum((Zj - Zj.mean())**2))
                            if den != 0:
                                cross.iloc[i, j] = num / den
                
                np.fill_diagonal(cross.values, 0)
                
                st.write("**Matriks Cross-Correlation (lag {})**:".format(k))
                st.dataframe(cross.round(4))
                
                # Normalisasi
                norm_cross = cross.copy()
                for i in range(n):
                    row_sum = np.abs(cross.iloc[i, :]).sum()
                    if row_sum != 0:
                        norm_cross.iloc[i, :] = cross.iloc[i, :] / row_sum
                
                st.write("**Matriks Pembobot Normalisasi Korelasi Silang:**")
                st.dataframe(norm_cross.round(4))
                
                # Simpan ke session state
                st.session_state['norm_cross'] = norm_cross.values
                
                st.success("Matriks pembobot berhasil dibuat")
    
    # TAB 4: Estimasi Model
    with tabs[3]:
        st.subheader("Estimasi Model GSTAR-SUR (Backward Elimination)")
        
        if st.button("Jalankan Pemodelan"):
            if 'norm_cross' not in st.session_state:
                st.error("Silakan buat matriks pembobot terlebih dahulu di tab sebelumnya")
                st.stop()
            
            with st.spinner("Membangun model... Proses ini memakan waktu beberapa menit..."):
                # Buat variabel GSTAR
                gstar_data = create_gstar_variables(df, cities, st.session_state['norm_cross'], max_lag=3)
                gstar_clean = gstar_data.dropna()
                
                # Split data
                test_size = 14
                train = gstar_clean.iloc[:-test_size, :]
                test = gstar_clean.iloc[-test_size:, :]
                
                st.session_state['train'] = train
                st.session_state['test'] = test
                
                # Backward elimination
                progress_bar = st.progress(0)
                selected_vars, model, log = backward_gstar_sur(train, cities, max_lag=3, progress_bar=progress_bar)
                progress_bar.empty()
                
                st.session_state['selected_vars'] = selected_vars
                st.session_state['model'] = model
                
                # Tampilkan log
                with st.expander("Log Backward Elimination"):
                    for line in log:
                        st.text(line)
                
                # Tampilkan variabel terpilih
                st.write("**Variabel Terpilih:**")
                for city in cities:
                    st.write(f"- **{city}**: {len(selected_vars[city])} variabel")
                    st.caption(f"  {', '.join(selected_vars[city])}")
                
                # Tampilkan nilai estimasi parameter (koefisien)
                st.write("**Nilai Estimasi Parameter (Koefisien):**")
                params_data = []
                for city in cities:
                    equation_name = f"g_{city.lower()}_"
                    city_params = model.params[model.params.index.str.startswith(equation_name)]
                    for var_full, coef in city_params.items():
                        var_name = var_full.replace(equation_name, "")
                        params_data.append({
                            'Kota': city,
                            'Variabel': var_name,
                            'Koefisien': coef,
                            'Std Error': model.std_errors[var_full],
                            'P-value': model.pvalues[var_full]
                        })
                
                df_params = pd.DataFrame(params_data)
                st.dataframe(df_params.round(4), use_container_width=True)
                
                # Simpan info model untuk sidebar
                lag_order = st.session_state.get('optimal_lag', 3)
                diff_order = 1 if st.session_state.get('differenced', False) else 0
                st.session_state['best_model_name'] = f"GSTAR({lag_order},1)-I({diff_order})"
                
                st.success("Model berhasil diestimasi")
    
    # TAB 5: Evaluasi
    with tabs[4]:
        st.subheader("Evaluasi Model")
        
        if 'model' not in st.session_state:
            st.warning("Silakan jalankan pemodelan terlebih dahulu")
            st.stop()
        
        model = st.session_state['model']
        selected_vars = st.session_state['selected_vars']
        train = st.session_state['train']
        test = st.session_state['test']
        
        # Prediksi
        train_pred = pd.DataFrame(index=train.index)
        for city in cities:
            equation_name = f"g_{city.lower()}"
            train_pred[city] = model.fitted_values[equation_name]
        
        sur_data_test = OrderedDict()
        for city in cities:
            sur_data_test[f"g_{city.lower()}"] = {
                "dependent": test[city],
                "exog": test[selected_vars[city]]
            }
        
        test_pred_dict = model.predict(sur_data_test)
        test_pred = pd.DataFrame.from_dict(test_pred_dict)
        test_pred.columns = cities
        test_pred.index = test.index
        
        st.session_state['train_pred'] = train_pred
        st.session_state['test_pred'] = test_pred
        
        # Hitung metrik dengan parameter count
        n_params_per_city = {}
        for city in cities:
            equation_name = f"g_{city.lower()}_"
            city_params = model.params[model.params.index.str.startswith(equation_name)]
            n_params_per_city[city] = len(city_params)
        
        n_obs_test = len(test)
        
        test_metrics = {}
        for city in cities:
            test_metrics[city] = calculate_metrics(
                test[city], test_pred[city],
                n_params_per_city[city], n_obs_test
            )
        
        df_metrics = pd.DataFrame(test_metrics).T
        df_metrics.loc['Rata-rata'] = df_metrics.mean()
        
        st.write("**Metrik Evaluasi (Out-Sample):**")
        st.dataframe(df_metrics.round(4), use_container_width=True)
        
        st.write("**Jumlah Parameter per Kota:**")
        for city in cities:
            st.write(f"- {city}: {n_params_per_city[city]} parameter")
        
        # st.write("**Jumlah Parameter per Kota:**")
        # for city in cities:
        #     st.write(f"- {city}: {n_params_per_city[city]} parameter")
        
        # Visualisasi
        st.write("**Visualisasi Prediksi:**")
        
        color_palette = ['#D62828', '#2A9D8F', '#1D4ED8', '#F4A261', '#264653']
        
        fig, axes = plt.subplots(n_cities, 1, figsize=(14, 4*n_cities))
        if n_cities == 1:
            axes = [axes]
        
        for i, city in enumerate(cities):
            color = color_palette[i % len(color_palette)]
            axes[i].plot(test.index, test[city], marker='o', label='Aktual', color=color, linewidth=2)
            axes[i].plot(test_pred.index, test_pred[city], marker='s', label='Prediksi', 
                        color='black', linestyle='--', linewidth=2)
            axes[i].set_title(f'{city} - Out-Sample Prediction', fontweight='bold')
            axes[i].set_ylabel('Harga (Rp/Kg)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.success("Evaluasi model selesai")

# =========================
# HASIL PREDIKSI PAGE
# =========================
elif menu == "Hasil Prediksi":
    st.header("Hasil Prediksi Harga Cabai")
    
    if 'model' not in st.session_state or 'selected_vars' not in st.session_state:
        st.warning("Silakan jalankan pemodelan terlebih dahulu di menu **Pemodelan GSTAR-SUR**")
        st.stop()
    
    model = st.session_state['model']
    selected_vars = st.session_state['selected_vars']
    df = st.session_state['filtered_df']
    cities = st.session_state['cities']
    n_cities = st.session_state['n_cities']
    W = st.session_state['norm_cross']
    
    # Input untuk jumlah hari prediksi (fleksibel)
    st.subheader("Pengaturan Prediksi")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        horizon = st.number_input(
            "Jumlah Hari Prediksi:",
            min_value=1,
            max_value=60,
            value=14,
            step=1,
            help="Tentukan berapa hari ke depan yang ingin diprediksi (1-60 hari)"
        )
    with col2:
        st.metric("Hari Dipilih", f"{horizon} hari")
    
    st.markdown("---")
    
    # Forecast function
    def forecast_future(df, cities, W, model, selected_vars, horizon=14, max_lag=3):
        forecast_data = df[cities].copy()
        n = len(cities)
        forecast_results = []
        
        for step in range(1, horizon + 1):
            lags = {}
            for i in range(1, max_lag + 2):
                for city in cities:
                    lags[f'lag{i}_{city}'] = forecast_data[city].iloc[-i]
            
            wlags = {}
            for i in range(1, max_lag + 1):
                for j, city in enumerate(cities):
                    wlag = 0
                    for k in range(n):
                        if j != k:
                            wlag += W[j, k] * (lags[f'lag{i}_{cities[k]}'] - lags[f'lag{i+1}_{cities[k]}'])
                    wlags[f'wlag{i}_{city}'] = wlag
            
            all_vars = {**lags, **wlags}
            
            exog_forecast = OrderedDict()
            for city in cities:
                vars_dict = {var: all_vars[var] for var in selected_vars[city]}
                exog_forecast[f"g_{city.lower()}"] = {"exog": pd.DataFrame([vars_dict])}
            
            y_pred = model.predict(exog_forecast)
            predictions = [y_pred[f'g_{city.lower()}'].iloc[0] for city in cities]
            
            forecast_results.append(predictions)
            
            new_date = forecast_data.index[-1] + pd.Timedelta(days=1)
            new_row = pd.DataFrame([predictions], columns=cities, index=[new_date])
            forecast_data = pd.concat([forecast_data, new_row])
        
        forecast_df = pd.DataFrame(
            forecast_results,
            columns=cities,
            index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
        )
        
        return forecast_df
    
    # Lakukan forecast
    with st.spinner(f"Melakukan prediksi {horizon} hari ke depan..."):
        forecast_df = forecast_future(df, cities, W, model, selected_vars, horizon=horizon, max_lag=3)
    
    st.session_state['forecast_df'] = forecast_df
    
    st.success("Prediksi berhasil dibuat")
    
    # Tabel hasil
    st.subheader("Tabel Hasil Prediksi")
    st.dataframe(forecast_df.round(2), use_container_width=True)
    
    # Statistik
    st.subheader("Statistik Ringkas")
    cols = st.columns(n_cities)
    
    for i, city in enumerate(cities):
        with cols[i]:
            avg = forecast_df[city].mean()
            change = forecast_df[city].iloc[-1] - forecast_df[city].iloc[0]
            st.metric(
                f"{city}",
                f"Rp {avg:,.0f}",
                f"{change:,.0f}",
                delta_color="inverse"
            )
            st.caption(f"Min: Rp {forecast_df[city].min():,.0f}")
            st.caption(f"Max: Rp {forecast_df[city].max():,.0f}")
    
    # Visualisasi
    st.subheader("Visualisasi Prediksi")
    
    # Tampilkan data historis sebanyak hari yang diprediksi (max 30 hari)
    hist_days = min(horizon, 30)
    historical_last = df[cities].iloc[-hist_days:]
    plot_data = pd.concat([historical_last, forecast_df])
    
    color_palette = ['#D62828', '#2A9D8F', '#1D4ED8', '#F4A261', '#264653', 
                     '#E76F51', '#8338EC', '#FF006E', '#FB5607', '#FFBE0B']
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for i, city in enumerate(cities):
        color = color_palette[i % len(color_palette)]
        
        ax.plot(historical_last.index, historical_last[city],
               color=color, linewidth=2.5, label=f"{city} (Historis)")
        
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        ax.plot(forecast_df.index, forecast_df[city],
               marker=markers[i % len(markers)], linestyle='--', color=color,
               linewidth=2.5, markersize=6, label=f"{city} (Forecast)")
    
    ax.axvline(x=historical_last.index[-1], color='gray', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_title(f"Prediksi Harga Cabai Merah Besar {horizon} Hari ke Depan ({n_cities} Kota)",
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Tanggal", fontsize=14, fontweight='bold')
    ax.set_ylabel("Harga /Kg (Rp)", fontsize=14, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    # Adjust interval based on forecast horizon
    interval = max(1, (hist_days + horizon) // 10)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    plt.xticks(rotation=45, ha='right')
    
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Kabupaten/Kota", fontsize=10, ncol=2, loc='best')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Download
    st.subheader("Download Hasil")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = forecast_df.to_csv()
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"forecast_{n_cities}kota_{horizon}hari.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            forecast_df.to_excel(writer, sheet_name='Forecast')
        excel_data = output.getvalue()
        
        st.download_button(
            label="Download Excel",
            data=excel_data,
            file_name=f"forecast_{n_cities}kota_{horizon}hari.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # Insight
    st.subheader("Insight Prediksi")
    
    insights = []
    for city in cities:
        trend = "naik" if forecast_df[city].iloc[-1] > forecast_df[city].iloc[0] else "turun"
        persen_change = ((forecast_df[city].iloc[-1] - forecast_df[city].iloc[0]) / forecast_df[city].iloc[0]) * 100
        insights.append(f"- **{city}**: Tren {trend} ({persen_change:+.2f}%)")
    
    st.markdown("\n".join(insights))
    
    last_prices = forecast_df.iloc[-1]
    kota_termurah = last_prices.idxmin()
    kota_termahal = last_prices.idxmax()
    
    st.info(f"""
    **Ringkasan:**
    - Kota dengan harga tertinggi: **{kota_termahal}** (Rp {last_prices[kota_termahal]:,.2f}/kg)
    - Kota dengan harga terendah: **{kota_termurah}** (Rp {last_prices[kota_termurah]:,.2f}/kg)
    - Selisih harga: Rp {(last_prices[kota_termahal] - last_prices[kota_termurah]):,.2f}/kg
    """)
