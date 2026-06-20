import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff

# =====================================================
# LOAD MODEL & ENCODER
# =====================================================
df = pd.read_csv("customer_segmented.csv")
model = joblib.load("random_forest_model.pkl")

gender_encoder = joblib.load("gender_encoder.pkl")
city_encoder = joblib.load("city_encoder.pkl")
membership_encoder = joblib.load("membership_encoder.pkl")

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background: linear-gradient(
        180deg,
        #FFF7FA 0%,
        #FDF2F8 50%,
        #FCE7F3 100%
    );
}

section[data-testid="stSidebar"]{
    background: #FFFFFF;
    border-right: 2px solid #FBCFE8;
}

.hero{
    background: linear-gradient(
        135deg,
        #F9A8D4,
        #F472B6,
        #EC4899
    );

    padding:45px;
    border-radius:30px;
    text-align:center;
    color:white;

    box-shadow:
        0px 10px 30px rgba(236,72,153,0.25);

    margin-bottom:25px;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:25px;
    text-align:center;

    border:2px solid #FBCFE8;

    box-shadow:
        0px 5px 15px rgba(236,72,153,0.08);
}
.metric-card:hover,
.feature-card:hover{

    transform: translateY(-8px);

    box-shadow:
    0px 15px 35px rgba(236,72,153,0.20);

    transition:0.3s;
}
.feature-card{
    background:white;
    padding:25px;
    border-radius:25px;

    border:2px solid #F9A8D4;

    box-shadow:
        0px 5px 15px rgba(236,72,153,0.08);

    transition:0.3s;
}

.feature-card:hover{
    transform:translateY(-5px);

    box-shadow:
        0px 10px 25px rgba(236,72,153,0.15);
}

h1,h2,h3{
    color:#DB2777;
}

div[data-testid="stMetric"]{
    background:white;
    border-radius:20px;
    padding:15px;
    border:2px solid #FBCFE8;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* Sidebar */
section[data-testid="stSidebar"]{
    background: linear-gradient(
        180deg,
        #FFF7FA,
        #FCE7F3
    );
    border-right: 3px solid #F9A8D4;
}

/* Judul sidebar */
section[data-testid="stSidebar"] h1{
    color:#DB2777;
    text-align:center;
}

/* Radio menu */
div[role="radiogroup"]{
    background:white;
    padding:15px;
    border-radius:20px;
    box-shadow:0px 4px 10px rgba(236,72,153,0.1);
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown("""
<div style="
background:linear-gradient(135deg,#F9A8D4,#EC4899);
padding:20px;
border-radius:20px;
text-align:center;
color:white;
margin-bottom:15px;
">

<h2>🛍️ Customer Segmentation</h2>
<p>Hybrid K-Means & Random Forest</p>

</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🌸 Menu Navigasi")

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "🏠 Dashboard",
        "👥 Customer Prediction",
        "📈 Cluster Analysis",
        "ℹ️ Tentang Model"
    ]
)
# =====================================================
# DASHBOARD
# =====================================================
if menu == "🏠 Dashboard":

    # =====================================================
    # HERO BANNER
    # =====================================================

    st.markdown("""
    <div class="hero">
        <h1>🛍️ Dashboard Segmentasi Pelanggan</h1>
<h3>Metode K-Means Clustering dan Random Forest</h3>
<p>
Menganalisis perilaku pelanggan,
mengidentifikasi segmen pelanggan,
serta mendukung pengambilan keputusan bisnis.
</p>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # KPI
    # =====================================================
    total_customer = len(df)
    st.subheader("📊 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.markdown(f"""
    <div class='metric-card'>
        <h2>👥</h2>
        <h1>{total_customer}</h1>
        <p>Total Pelanggan</p>
    </div>
    """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
    <div class='metric-card'>
        <h2>📊</h2>
        <h1>3</h1>
        <p>Clusters</p>
    </div>
    """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
    <div class='metric-card'>
        <h2>🎯</h2>
        <h1>98.15%</h1>
        <p>Akurasi</p>
    </div>
    """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
    <div class='metric-card'>
        <h2>📈</h2>
        <h1>0.295</h1>
        <p>Silhouette Score</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.subheader("📋 Statistik Utama Dataset")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
        "Rata-rata Usia",
        round(df["Age"].mean(), 1)
    )

    with col2:
        st.metric(
        "Rata-rata Rating",
        round(df["Average_Rating"].mean(), 2)
    )

    with col3:
     st.metric(
        "Rata-rata Total Spend",
        f"${round(df['Total_Spend'].mean(), 2)}"
    )

    with col4:
        st.metric(
        "Rata-rata Item",
        round(df["Items_Purchased"].mean(), 1)
    )

    st.divider()

    # =====================================================
    # CUSTOMER SEGMENT DISTRIBUTION
    # =====================================================
    st.subheader("📊 Distribusi Segmen Pelanggan & Insight Bisnis")

    col_left, col_right = st.columns([3,2])
# ==========================
# KIRI
# ==========================
    with col_left:

        segment_data = (
        df["Customer_Segment"]
        .value_counts()
        .reset_index()
    )

        segment_data.columns = [
        "Segment",
        "Jumlah Customer"
    ]

        fig = px.pie(
        segment_data,
        values="Jumlah Customer",
        names="Segment",
        hole=0.55,
        color="Segment",
        color_discrete_map={
            "High Value Customer":"#EC4899",
            "Inactive Customer":"#F9A8D4",
            "Low Value Customer":"#FBCFE8"
        }
    )

        fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True
    )

        st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================
# KANAN
# ==========================
    with col_right:

        st.markdown("""
    <div style="
    background:#FFE4F1;
    padding:15px;
    border-radius:15px;
    margin-bottom:10px;
    ">
    <b> High Value Customer</b><br>
   Memberikan kontribusi terbesar terhadap pendapatan perusahaan.
    </div>
    """, unsafe_allow_html=True)

        st.markdown("""
    <div style="
    background:#FFF7ED;
    padding:15px;
    border-radius:15px;
    margin-bottom:10px;
    ">
    <b>Inactive Customer</b><br>
    Berisiko tinggi berhenti menggunakan layanan.
    </div>
    """, unsafe_allow_html=True)

        st.markdown("""
    <div style="
    background:#FDF2F8;
    padding:15px;
    border-radius:15px;
    ">
    <b>Low Value Customer</b><br>
    Memiliki potensi untuk ditingkatkan menjadi pelanggan loyal.
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # AVAILABLE FEATURES
    # =====================================================

    st.subheader("🚀 Fitur yang Tersedia")

    col1,col2 = st.columns(2)

    with col1:

        st.markdown("""
    <div class='feature-card'>
    <h3> Prediksi Segmen Pelanggan</h3>

Memprediksi segmen pelanggan
menggunakan Random Forest.
                    
High Value Customer
Inactive Customer
Low Value Customer
    </div>
    """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
    <div class='feature-card'>
    <h3>📈 Analisis Cluster</h3>

Menganalisis distribusi cluster
dan perilaku pelanggan.
                    
    📊 K-Means
    📌 Business Insights
    """, unsafe_allow_html=True)

    # =====================================================
    # PROJECT SUMMARY
    # =====================================================

    st.subheader("📌 Ringkasan Proyek")

    st.markdown("""
Proyek ini menggabungkan metode **K-Means Clustering**
dan **Random Forest Classification** untuk melakukan
segmentasi pelanggan serta memprediksi segmen pelanggan baru secara otomatis.

Sistem ini membantu perusahaan untuk:

- Memahami perilaku pelanggan
- Mengidentifikasi pelanggan bernilai tinggi
- Mengurangi risiko kehilangan pelanggan
- Meningkatkan efektivitas strategi pemasaran
- Mendukung pengambilan keputusan bisnis
""")

    
# =====================================================
# CUSTOMER PREDICTION
# =====================================================
elif menu == "👥 Customer Prediction":

    st.title("Prediksi Segmen Pelanggan")

    st.markdown("""
    Masukkan informasi pelanggan untuk memprediksi segment customer
    menggunakan model Random Forest yang telah dilatih dari hasil
    segmentasi K-Means.
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "👤 Age",
            min_value=18,
            max_value=100,
            value=25
        )

        gender = st.selectbox(
            "🚻 Gender",
            ["Female", "Male"]
        )

        membership = st.selectbox(
            "💳 Membership Type",
            ["Bronze", "Gold", "Silver"]
        )

    with col2:

        city = st.selectbox(
            "🏙️ City ",
            [
                "Chicago",
                "Houston",
                "Los Angeles",
                "Miami",
                "New York",
                "San Francisco"
            ]
        )

        rating = st.slider(
            "⭐ Average Rating",
            min_value=1.0,
            max_value=5.0,
            value=4.0,
            step=0.1
        )

    st.divider()

    if st.button("🔍 Predict Segment", use_container_width=True):

        try:

            gender_encoded = gender_encoder.transform([gender])[0]

            membership_encoded = membership_encoder.transform(
                [membership]
            )[0]

            city_encoded = city_encoder.transform(
                [city]
            )[0]

            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender_encoded],
                'Membership_Type': [membership_encoded],
                'City': [city_encoded],
                'Average_Rating': [rating]
            })

            prediction = model.predict(input_data)[0]

            st.subheader("📌 Hasil Prediksi")

            if prediction == "High Value Customer":

                st.markdown("""
<div style="
background:#FDF2F8;
padding:20px;
border-radius:20px;
border-left:6px solid #EC4899;
box-shadow:0 4px 12px rgba(0,0,0,0.05);
margin-bottom:15px;
">

<h3 style="color:#DB2777;">❦ High Value Customer</h3>

<b>Karakteristik:</b>
<ul>
<li>Pelanggan loyal</li>
<li>Frekuensi pembelian tinggi</li>
<li>Nilai transaksi tinggi</li>
<li>Kontributor utama pendapatan</li>
</ul>

<b>Rekomendasi Strategi:</b>
<ul>
<li>Program loyalitas</li>
<li>Promosi eksklusif</li>
<li>Penawaran personal</li>
<li>Layanan premium</li>
</ul>

</div>
""", unsafe_allow_html=True)

            elif prediction == "Inactive Customer":

                st.markdown("""
<div style="
background:#EFF6FF;
padding:20px;
border-radius:20px;
border-left:6px solid #60A5FA;
box-shadow:0 4px 12px rgba(0,0,0,0.05);
margin-bottom:15px;
">

<h3 style="color:#1D4ED8;">❦ Inactive Customer</h3>

<b>Karakteristik:</b>
<ul>
<li>Jarang melakukan pembelian</li>
<li>Frekuensi transaksi rendah</li>
<li>Risiko churn tinggi</li>
<li>Tingkat keterlibatan rendah</li>
</ul>

<b>Rekomendasi Strategi:</b>
<ul>
<li>Program retensi pelanggan</li>
<li>Voucher dan diskon khusus</li>
<li>Email marketing</li>
<li>Program reaktivasi pelanggan</li>
</ul>

</div>
""", unsafe_allow_html=True)

            elif prediction == "Low Value Customer":

                st.markdown("""
<div style="
background:#F0FDF4;
padding:20px;
border-radius:20px;
border-left:6px solid #22C55E;
box-shadow:0 4px 12px rgba(0,0,0,0.05);
margin-bottom:15px;
">

<h3 style="color:#15803D;">❦ Low Value Customer</h3>

<b>Karakteristik:</b>
<ul>
<li>Tingkat pengeluaran rendah</li>
<li>Frekuensi pembelian rendah</li>
<li>Masih memiliki potensi berkembang</li>
<li>Cenderung sensitif terhadap harga</li>
</ul>

<b>Rekomendasi Strategi:</b>
<ul>
<li>Cross-selling</li>
<li>Upselling</li>
<li>Promosi membership</li>
<li>Bundling produk</li>
</ul>

</div>
""", unsafe_allow_html=True)

            else:

                st.write("Prediction:", prediction)

            st.divider()

            st.subheader("📋 Informasi Pelanggan")

            customer_info = pd.DataFrame({
                "Feature": [
                    "Age",
                    "Gender",
                    "Membership",
                    "City",
                    "Average Rating"
                ],
                "Value": [
                    age,
                    gender,
                    membership,
                    city,
                    rating
                ]
            })

            st.dataframe(
                customer_info,
                use_container_width=True
            )

        except Exception as e:

            st.error(f"Error: {e}")

# =====================================================
# CLUSTER ANALYSIS
# =====================================================
elif menu == "📈 Cluster Analysis":

    st.title("📈 Analisis Cluster")

    st.markdown("""
    Analisis hasil segmentasi pelanggan menggunakan algoritma K-Means Clustering.
    """)

    st.divider()

    # Distribusi Cluster

    st.subheader("📊 Distribusi Pelanggan Berdasarkan Segmen")

    cluster_data = (
    df["Customer_Segment"]
    .value_counts()
    .reset_index()
)

    cluster_data.columns = [
    "Segment",
    "Jumlah Customer"
]

    fig_bar = px.bar(
    cluster_data,
    x="Segment",
    y="Jumlah Customer"
)

    fig_bar.update_traces(
    marker_color="#EC4899"
)

    fig_bar.update_layout(
    title="Distribusi Pelanggan Berdasarkan Segmen",
    xaxis_title="Segmen",
    yaxis_title="Jumlah Pelanggan",
    showlegend=False,
    height=400
)

    st.plotly_chart(
    fig_bar,
    use_container_width=True
)

    st.divider()

    X = df[
    [
        'Age',
        'Gender',
        'Membership_Type',
        'City',
        'Average_Rating'
    ]
]

    y = df['Customer_Segment']

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    st.subheader("📊 Actual vs Predicted")

    fig_cm = ff.create_annotated_heatmap(
    z=cm,
    x=list(model.classes_),
    y=list(model.classes_),
    colorscale='Pinkyl'
)

    fig_cm.update_layout(
    title="Confusion Matrix Random Forest",
    xaxis_title="Predicted",
    yaxis_title="Actual"
)

    st.plotly_chart(
    fig_cm,
    use_container_width=True
)
    # Ringkasan Cluster

    st.subheader("Ringkasan Cluster")

    summary = pd.DataFrame({
        "Segment":[
            "High Value Customer",
            "Inactive Customer",
            "Low Value Customer"
        ],
        "Karakteristik":[
        "Memiliki nilai transaksi tinggi dan frekuensi pembelian yang tinggi",
        "Jarang melakukan transaksi dan memiliki risiko churn yang tinggi",
        "Memiliki nilai transaksi rendah dan frekuensi pembelian yang rendah"
    ]
})

    st.dataframe(
        summary,
        use_container_width=True
    )

    st.divider()

    # Interpretasi
    st.subheader("📝 Interpretasi Segmen")

    st.markdown("""
<div style="
background:#FFE6EE;
padding:20px;
border-radius:15px;
border-left:6px solid #38BDF8;
margin-bottom:15px;
">
<h4>❦ High Value Customer</h4>

<b>Karakteristik:</b>
<ul>
<li>Nilai transaksi tinggi</li>
<li>Frekuensi pembelian tinggi</li>
<li>Pelanggan loyal</li>
<li>Kontributor utama pendapatan perusahaan</li>
</ul>

<b>Strategi Bisnis:</b>
<ul>
<li>Program loyalitas pelanggan</li>
<li>Promosi eksklusif</li>
<li>Penawaran yang dipersonalisasi</li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="
background:#FEDCD2;
padding:20px;
border-radius:15px;
border-left:6px solid #60A5FA;
margin-bottom:15px;
">
<h4>❦ Inactive Customer</h4>

<b>Karakteristik:</b>
<ul>
<li>Frekuensi transaksi rendah</li>
<li>Sudah lama tidak melakukan pembelian</li>
<li>Risiko churn tinggi</li>
</ul>

<b>Strategi Bisnis:</b>
<ul>
<li>Program retensi pelanggan</li>
<li>Email marketing</li>
<li>Voucher dan diskon khusus</li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="
background:#FEDCDB;
padding:20px;
border-radius:15px;
border-left:6px solid #2563EB;
">
<h4>❦ Low Value Customer</h4>

<b>Karakteristik:</b>
<ul>
<li>Tingkat pengeluaran rendah</li>
<li>Aktivitas pembelian terbatas</li>
<li>Memiliki potensi untuk menjadi pelanggan loyal</li>
</ul>

<b>Strategi Bisnis:</b>
<ul>
<li>Cross-selling</li>
<li>Upselling</li>
<li>Promosi keanggotaan (membership)</li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.divider()

    st.subheader("📈 Clustering Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Number of Clusters",
            "3"
        )

    with col2:
        st.metric(
            "Silhouette Score",
            "0.295"
        )

# =====================================================
# ABOUT MODEL
# =====================================================
elif menu == "ℹ️ Tentang Model":

    st.title("ℹ️ Tentang Model")

    st.markdown("""
                
    Proyek ini bertujuan untuk melakukan segmentasi pelanggan
    berdasarkan karakteristik dan perilaku pelanggan menggunakan
    pendekatan hybrid yang menggabungkan algoritma
    K-Means Clustering dan Random Forest.

    Hasil segmentasi dapat membantu perusahaan memahami perilaku pelanggan,
    menyusun strategi pemasaran yang lebih efektif,
    meningkatkan loyalitas pelanggan,
    serta mendukung pengambilan keputusan bisnis.
    """)

    st.divider()

    st.subheader("⚙️ Algoritma yang Digunakan")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
    <div style="
    background:#FDF2F8;
    padding:20px;
    border-radius:20px;
    border-left:6px solid #EC4899;
    box-shadow:0px 4px 12px rgba(236,72,153,0.15);
    height:180px;
    ">
    <h4 style="color:#DB2777;">📊 K-Means Clustering</h4>
    <p>
    Digunakan untuk mengelompokkan pelanggan
    ke dalam beberapa segmen berdasarkan
    karakteristik dan perilaku yang serupa.
    </p>
    </div>
    """, unsafe_allow_html=True)

    with col2:
     st.markdown("""
    <div style="
    background:#FDF2F8;
    padding:20px;
    border-radius:20px;
    border-left:6px solid #EC4899;
    box-shadow:0px 4px 12px rgba(236,72,153,0.15);
    height:180px;
    ">
    <h4 style="color:#DB2777;">🌲 Random Forest</h4>
    <p>
    Digunakan untuk memprediksi segmen pelanggan
    berdasarkan atribut atau profil pelanggan.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.subheader("📋 Informasi Model")

    info = pd.DataFrame({
        "Informasi":[
            "Algoritma Clustering",
            "Algoritma Klasifikasi",
            "Jumlah Cluster",
            "Akurasi",
            "Silhouette Score"
        ],
        "Nilai":[
            "K-Means",
            "Random Forest",
            "3",
            "98.15%",
            "0.295"
        ]
    })

    st.dataframe(
        info,
        use_container_width=True
    )

    st.divider()

    st.subheader("📌 Fitur yang Digunakan")

    feature_df = pd.DataFrame({
        "Fitur":[
            "Usia",
            "Jenis Kelamin",
            "Tipe Membership",
            "Kota",
            "Rata-rata Rating"
        ],
        "Deskripsi":[
            "Usia pelanggan",
            "Jenis kelamin pelanggan",
            "Kategori keanggotaan pelanggan",
            "Kota tempat pelanggan berada",
            "Rata-rata penilaian produk"
        ]
    })

    st.dataframe(
        feature_df,
        use_container_width=True
    )

    st.divider()

    st.subheader("⚙️ Metodologi")
    st.markdown("""
<div style='display:flex;
            justify-content:center;
            align-items:center;
            flex-wrap:wrap;
            gap:10px;'>

<div style='background:#F9A8D4;color:white;
            padding:15px;border-radius:15px;
            text-align:center;min-width:120px;'>
📁<br><b>Dataset</b>
</div>

➜

<div style='background:#FBCFE8;color:#831843;
            padding:15px;border-radius:15px;
            text-align:center;min-width:120px;'>
🧹<br><b>Preprocessing</b>
</div>

➜

<div style='background:#DBEAFE;color:#1E3A8A;
            padding:15px;border-radius:15px;
            text-align:center;min-width:120px;'>
🔄<br><b>Encoding</b>
</div>

➜

<div style='background:#BFDBFE;color:#1E3A8A;
            padding:15px;border-radius:15px;
            text-align:center;min-width:120px;'>
📊<br><b>K-Means</b>
</div>

➜

<div style='background:#93C5FD;color:#1E3A8A;
            padding:15px;border-radius:15px;
            text-align:center;min-width:120px;'>
🏷️<br><b>Labeling</b>
</div>

➜

<div style='background:#7DD3FC;color:#1E3A8A;
            padding:15px;border-radius:15px;
            text-align:center;min-width:120px;'>
🌲<br><b>Random Forest</b>
</div>

➜

<div style='background:#A5F3FC;color:#164E63;
            padding:15px;border-radius:15px;
            text-align:center;min-width:120px;'>
🚀<br><b>Deployment</b>
</div>

</div>
""", unsafe_allow_html=True)

    st.divider()

    st.subheader("🌲Fitur yanng Berpengaruh")

    importance_df = pd.DataFrame({
    "Feature": [
        "Age",
        "Gender",
        "Membership_Type",
        "City",
        "Average_Rating"
    ],
    "Importance": model.feature_importances_
})

    importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)

    fig_importance = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale="RdPu"
)

    fig_importance.update_layout(
    title="Pengaruh Variabel terhadap Prediksi Segmen Pelanggan",
    height=450
)

    st.plotly_chart(
    fig_importance,
    use_container_width=True
)
    st.info("""
Feature Importance menunjukkan variabel yang paling berpengaruh
dalam menentukan segmen pelanggan pada model Random Forest.

Semakin tinggi nilai importance, semakin besar kontribusi variabel
terhadap hasil prediksi.
""")
    st.subheader("📈 Performa Model")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Akurasi",
            "98.15%"
        )

    with col2:
        st.metric(
            "Silhouette Score",
            "0.295"
        )

    st.divider()

    st.markdown("""
<div style="
background:#FDF2F8;
padding:25px;
border-radius:20px;
border-left:8px solid #EC4899;
box-shadow:0px 5px 15px rgba(236,72,153,0.15);
">

<h3 style="color:#DB2777;">Kesimpulan</h3>

<p>
Pendekatan <b>Hybrid K-Means Clustering dan Random Forest</b>
berhasil mengelompokkan pelanggan ke dalam tiga segmen utama:
</p>

<ul>
<li><b> ❦ High Value Customer</b> (Pelanggan Bernilai Tinggi)</li>
<li><b> ❦ Low Value Customer</b> (Pelanggan Bernilai Rendah)</li>
<li><b> ❦ Inactive Customer</b> (Pelanggan Tidak Aktif)</li>
</ul>

<p>
Model ini dapat digunakan untuk membantu perusahaan
memahami karakteristik pelanggan,
menyusun strategi pemasaran yang lebih tepat sasaran,
meningkatkan loyalitas pelanggan,
serta mendukung pengambilan keputusan bisnis.
</p>

</div>
""", unsafe_allow_html=True)
# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<br><br>

<div style="
background:white;
padding:20px;
border-radius:20px;
text-align:center;
border:2px solid #FBCFE8;
">

<h4>🛍️ Sistem Segmentasi Pelanggan</h4>

<p>
Dikembangkan oleh <b>Azhara Kumala </b><br>
Sains Data
</p>

</div>
""", unsafe_allow_html=True)
