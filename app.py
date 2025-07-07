import streamlit as st
import pandas as pd
from datetime import datetime
from ihsg_platform import IHSGIntelligencePlatform
from chart_utils import ChartUtils
from ollama_chat import OllamaChat
from css import load_css

# Konfigurasi halaman
st.set_page_config(
    page_title="INVISE - IHSG Navigation & Visual Intelligence System Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Muat CSS kustom
st.markdown(load_css(), unsafe_allow_html=True)

def init_session_state():
    """Initialize all necessary session state variables."""
    if 'platform' not in st.session_state:
        st.session_state.platform = IHSGIntelligencePlatform()
    if 'chart_utils' not in st.session_state:
        st.session_state.chart_utils = ChartUtils()
    if 'ollama_chat' not in st.session_state:
        st.session_state.ollama_chat = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'correlations' not in st.session_state:
        st.session_state.correlations = {}
    if 'model_performance' not in st.session_state:
        st.session_state.model_performance = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'available_models' not in st.session_state:
        st.session_state.available_models = []
    if 'analysis_indonesian' not in st.session_state:
        st.session_state.analysis_indonesian = ""
    if 'analysis_english' not in st.session_state:
        st.session_state.analysis_english = ""

def sidebar_controls():
    """Manages and displays all sidebar controls."""
    st.sidebar.title("🎛️ INVISE Control Panel")
    
    st.sidebar.markdown("### ⚙️ Alur Kerja Utama")
    
    if st.sidebar.button("🔄 1. Muat Data Pasar", use_container_width=True):
        with st.spinner("Mengunduh data pasar..."):
            st.session_state.platform.fetch_data(period='5y')
        st.session_state.data_loaded = True
        st.session_state.models_trained = False
        st.session_state.predictions = None
        st.rerun()

    if st.sidebar.button("🤖 2. Latih Model AI", use_container_width=True, disabled=not st.session_state.data_loaded):
        with st.spinner("Model sedang dilatih..."):
            performance = st.session_state.platform.train_models()
        if performance:
            st.session_state.model_performance = performance
            st.session_state.models_trained = True
        else:
            st.session_state.models_trained = False
        st.session_state.predictions = None
        st.rerun()

    if st.sidebar.button("🔮 3. Buat Prediksi", use_container_width=True, disabled=not st.session_state.models_trained):
        with st.spinner("Menghasilkan prediksi baru..."):
            st.session_state.predictions = st.session_state.platform.make_prediction()
            if st.session_state.predictions:
                st.session_state.correlations = st.session_state.platform.calculate_market_correlations()
        st.rerun()

    st.sidebar.markdown("### 📈 Status Sistem")
    st.sidebar.write(f"{'✅' if st.session_state.data_loaded else '❌'} Data Dimuat")
    st.sidebar.write(f"{'✅' if st.session_state.models_trained else '❌'} Model Dilatih")
    st.sidebar.write(f"{'✅' if st.session_state.predictions else '❌'} Prediksi Siap")

    with st.sidebar.expander("⚙️ Pengaturan Lanjutan", expanded=True):
        horizon_options = range(1,31)        
        current_horizon = getattr(st.session_state.platform, 'prediction_horizon', 1)

        try:
            current_index = horizon_options.index(current_horizon)
        except ValueError:
            current_index = 0
            st.session_state.platform.prediction_horizon = horizon_options[current_index]

        selected_horizon = st.selectbox(
            "Horizon Prediksi (Hari)", 
            horizon_options, 
            index=current_index
        )
        
        if selected_horizon != st.session_state.platform.prediction_horizon:
            st.session_state.platform.prediction_horizon = selected_horizon
            st.warning("Horizon diubah. Latih ulang model & buat prediksi baru.")
            st.session_state.models_trained = False
            st.session_state.predictions = None
            st.rerun()

    st.sidebar.markdown("### 🤖 Konfigurasi AI (Ollama)")
    if st.sidebar.button("🔗 Hubungkan ke Ollama", use_container_width=True):
        st.session_state.ollama_chat = OllamaChat()
        st.session_state.available_models = st.session_state.ollama_chat.check_connection()
        st.rerun()
    
    if st.session_state.ollama_chat and st.session_state.ollama_chat.is_connected():
        st.sidebar.success("✅ Terhubung ke Ollama")
        model_options = ["None"] + st.session_state.available_models
        if 'selected_model' not in st.session_state or st.session_state.selected_model not in model_options:
            st.session_state.selected_model = model_options[0]
        
        selected_model_index = model_options.index(st.session_state.selected_model)
        selected_model = st.sidebar.selectbox(
            "Pilih Model AI", model_options, index=selected_model_index
        )
        if st.session_state.selected_model != selected_model:
            st.session_state.selected_model = selected_model
            st.session_state.ollama_chat.set_model(selected_model)
            st.rerun()
    else:
        st.sidebar.info("Klik tombol di atas untuk terhubung ke Ollama.")


def display_dashboard():
    """Displays the main dashboard content."""
    st.markdown("## 📊 INVISE Dashboard")
    
    if not st.session_state.data_loaded:
        st.info("👆 Mulai dengan memuat data pasar melalui INVISE Control Panel di sidebar.")
        return
    
    st.markdown("### Ringkasan Pasar Saat Ini")
    market_summary = st.session_state.platform.get_market_summary()
    if market_summary:
        cols = st.columns(4)
        assets_to_show = ['IHSG', 'USD_IDR', 'Gold', 'Bitcoin']
        for i, asset in enumerate(assets_to_show):
            if asset in market_summary:
                data = market_summary[asset]
                cols[i].metric(
                    asset, f"{data['current_price']:,.2f}".replace(".00", ""),
                    f"{data['daily_change']:+.2%}",
                    delta_color="normal"
                )
    
    st.markdown("---")
    
    st.markdown("### 🔮 Hasil Peramalan IHSG - INVISE Intelligence")
    if st.session_state.predictions:
        pred = st.session_state.predictions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            direction_class = "up" if pred['direction'] == 'UP' else "down"
            st.markdown(f"""
            <div class="prediction-card {direction_class}">
                <h5>Prediksi {pred['prediction_horizon']} Hari</h5>
                <h2>{pred['direction']} ({pred['magnitude']:.2f}%)</h2>
                <p>Kepercayaan: {pred['confidence']:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="prediction-card neutral">
                <h5>Prediksi Harga</h5>
                <h2>Rp {pred['predicted_price']:,.0f}</h2>
                <p>Perubahan: {pred['price_change']:+,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="prediction-card neutral">
                <h5>Harga Saat Ini</h5>
                <h2>Rp {pred['current_price']:,.0f}</h2>
                <p>Tanggal: {datetime.now().strftime('%d %b %Y')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        forecast_chart = st.session_state.chart_utils.create_forecast_chart(st.session_state.platform, st.session_state.predictions)
        if forecast_chart:
            st.plotly_chart(forecast_chart, use_container_width=True)

        with st.expander("📊 Lihat Detail Model: Prediksi, Performa & Faktor Penggerak"):
            col_a, col_b, col_c = st.columns([1, 2, 1.5])
            
            with col_a:
                st.markdown("**Prediksi Individu**")
                if pred.get('individual_predictions'):
                    preds_df = pd.DataFrame.from_dict(pred['individual_predictions'], orient='index', columns=['Prediksi Harga'])
                    st.dataframe(preds_df.style.format("Rp {:,.0f}"), use_container_width=True)
                    
            with col_b:
                st.markdown("**Performa Model (Data Uji)**")
                if st.session_state.model_performance:
                    perf_df = pd.DataFrame(st.session_state.model_performance).T
                    st.dataframe(perf_df.style.format({
                        'Test_MAE': '{:.2f}', 
                        'Test_RMSE': '{:.2f}', 
                        'Test_R2': '{:.2%}', 
                        'Directional_Accuracy': '{:.2f}%'
                    }), use_container_width=True)
                    
            with col_c:
                st.markdown("**Faktor Penggerak Utama**")
                feature_importance = st.session_state.platform.get_feature_importance()
                if feature_importance:
                    model_to_show = list(feature_importance.keys())[0]
                    top_features = feature_importance[model_to_show][:7]
                    features_df = pd.DataFrame(top_features, columns=['Fitur', 'Tingkat Pengaruh'])
                    features_df['Tingkat Pengaruh'] = (features_df['Tingkat Pengaruh'] * 100).map('{:.2f}%'.format)
                    st.dataframe(features_df, use_container_width=True, hide_index=True)
                    st.caption(f"Top 7 fitur dari model {model_to_show}")
    else:
        st.info("Klik 'Buat Prediksi' di sidebar untuk melihat hasil INVISE Intelligence.")

def generate_analysis_streaming(language):
    """Function to generate analysis via streaming."""
    # Clear both analyses to prevent duplication
    st.session_state.analysis_indonesian = ""
    st.session_state.analysis_english = ""
    
    placeholder = st.empty()
    full_response = ""
    
    try:
        stream = st.session_state.ollama_chat.get_market_analysis_streaming(
            predictions=st.session_state.predictions,
            correlations=st.session_state.correlations,
            market_context="Fokus pada sentimen pasar Indonesia menggunakan INVISE system." if language == 'id' else "Focus on the Indonesian market sentiment using INVISE system.",
            language=language
        )
        
        for chunk in stream:
            full_response += chunk
            placeholder.markdown(full_response + "▌")
        
        # Remove the cursor by showing final response
        placeholder.markdown(full_response)
        
        # Store the final response in session state
        if language == 'id':
            st.session_state.analysis_indonesian = full_response
        else:
            st.session_state.analysis_english = full_response
            
    except Exception as e:
        placeholder.error(f"Gagal menghasilkan analisis: {e}")


# Revised tab content for AI Intelligence - FIXED VERSION
def display_ai_intelligence_tab():
    """Display the AI Intelligence tab content without duplication."""
    st.markdown("## 🧠 INVISE AI Intelligence")
    is_ai_ready = st.session_state.ollama_chat and st.session_state.ollama_chat.is_connected() and st.session_state.ollama_chat.model_name != 'None'
    
    if not is_ai_ready:
        st.warning("Hubungkan ke Ollama dan pilih model di sidebar untuk mengaktifkan INVISE AI Intelligence.")
    elif not st.session_state.predictions:
        st.info("Buat prediksi terlebih dahulu untuk mendapatkan analisis dari INVISE AI Intelligence.")
    else:
        st.info("Pilih bahasa untuk menghasilkan analisis pasar mendalam dari INVISE AI Intelligence.")
        
        # Add a clear button if any analysis exists
        if st.session_state.analysis_indonesian or st.session_state.analysis_english:
            if st.button("🗑️ Hapus Analisis Sebelumnya", use_container_width=True, key="btn_clear_analysis"):
                st.session_state.analysis_indonesian = ""
                st.session_state.analysis_english = ""
                st.rerun()
        
        # Language selection buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🇮🇩 Buat Analisis (Bahasa Indonesia)", use_container_width=True, key="btn_analysis_id"):
                with st.spinner("INVISE AI Intelligence sedang menganalisis dalam Bahasa Indonesia..."):
                    generate_analysis_streaming('id')
                    
        with col2:
            if st.button("🇬🇧 Generate Analysis (English)", use_container_width=True, key="btn_analysis_en"):
                with st.spinner("INVISE AI Intelligence is analyzing in English..."):
                    generate_analysis_streaming('en')
        
        if st.session_state.analysis_indonesian or st.session_state.analysis_english:
            st.markdown("---")

# Updated main function with the revised tab content
def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="INVISE - IHSG Navigation & Visual Intelligence System Engine",
        page_icon="🇮🇩",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    st.markdown(load_css(), unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🇮🇩 INVISE</h1>
        <h2>IHSG Navigation & Visual Intelligence System Engine</h2>
        <p>Sistem Canggih untuk Analisis & Peramalan Pasar Saham Indonesia dengan Dukungan Artificial Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    sidebar_controls()
    
    # Main content tabs
    tab_titles = ["📊 Dashboard", "📈 Visual Analytics", "🧠 AI Intelligence", "💬 AI Assistant"]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        display_dashboard()
        
    with tabs[1]:
        st.markdown("## 📈 INVISE Visual Analytics")
        if not st.session_state.data_loaded:
            st.info("Muat data terlebih dahulu untuk menampilkan visual analytics.")
        else:
            enhanced_chart = st.session_state.chart_utils.create_enhanced_charts(st.session_state.platform)
            if enhanced_chart: st.plotly_chart(enhanced_chart, use_container_width=True)
            
            if st.session_state.models_trained:
                backtest_chart = st.session_state.chart_utils.create_backtest_chart(st.session_state.platform)
                if backtest_chart: st.plotly_chart(backtest_chart, use_container_width=True)

            if st.session_state.predictions and st.session_state.correlations:
                corr_chart = st.session_state.chart_utils.create_correlation_chart(st.session_state.correlations, st.session_state.platform.prediction_horizon)
                if corr_chart: st.plotly_chart(corr_chart, use_container_width=True)
    
    with tabs[2]:
        display_ai_intelligence_tab()

    with tabs[3]: # Tab "AI Assistant"
        st.markdown("## 💬 INVISE AI Assistant")
        is_chat_ready = st.session_state.ollama_chat and st.session_state.ollama_chat.is_connected() and st.session_state.ollama_chat.model_name != 'None'
        
        if not is_chat_ready:
            st.warning("Hubungkan ke Ollama dan pilih model di sidebar untuk memulai chat dengan INVISE AI Assistant.")
        else:
            if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

            # Tampilkan riwayat chat
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
                    st.markdown(message["content"])
            
            # Terima input dari pengguna
            if prompt := st.chat_input("Tanyakan apa saja tentang pasar atau data kepada INVISE AI Assistant..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.markdown(prompt)
                
                # Hasilkan dan tampilkan respons dari AI
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("INVISE AI sedang berpikir..."):
                        context = ""
                        if st.session_state.predictions:
                            pred = st.session_state.predictions
                            context = f"Prediksi IHSG untuk {pred['prediction_horizon']} hari ke depan adalah {pred['direction']} menuju Rp {pred['predicted_price']:,.0f}."
                        
                        # Panggil fungsi yang sudah memiliki persona
                        response_generator = st.session_state.ollama_chat.get_assistant_response(
                            user_question=prompt,
                            context=context
                        )

                        # Tampilkan respons streaming dari generator
                        placeholder = st.empty()
                        full_response = ""
                        for chunk in response_generator:
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")
                        placeholder.markdown(full_response)
                    
                    # Simpan respons ke histori chat jika tidak kosong
                    if full_response:
                        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                        st.rerun()
                        
if __name__ == "__main__":
    main()