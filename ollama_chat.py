import ollama
import streamlit as st
from typing import Generator, Dict, Any, List

class OllamaChat:
    def __init__(self, host: str = "http://localhost:11434", model_name: str = None):
        """Inisialisasi klien chat Ollama."""
        self.host = host
        self.client = ollama.Client(host=host)
        self.model_name = model_name
        self._is_connected = False

    def check_connection(self) -> List[str]:
        """Memeriksa koneksi dan mengembalikan daftar model."""
        try:
            models_data = self.client.list()
            self._is_connected = True
            return [model.model for model in models_data['models']]
        except Exception as e:
            st.error(f"Gagal terhubung ke Ollama di {self.host}. Pastikan server Ollama berjalan.")
            self._is_connected = False
            return []

    def is_connected(self) -> bool:
        """Mengembalikan status koneksi."""
        return self._is_connected

    def set_model(self, model_name: str):
        """Mengatur model yang akan digunakan."""
        self.model_name = model_name

    def chat_streaming(self, message: str, system_prompt: str = None) -> Generator[str, None, None]:
        """Fungsi chat generik dengan dukungan streaming dan system prompt."""
        if not self.model_name or not self.is_connected() or self.model_name == 'None':
            yield "Error: Ollama tidak terhubung atau model tidak dipilih."
            return
        
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': message})
        
        try:
            stream = self.client.chat(
                model=self.model_name, 
                messages=messages, 
                stream=True,
                options={'temperature': 0.7}
            )
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
        except Exception as e:
            yield f"Error saat berkomunikasi dengan Ollama: {str(e)}"

    def get_market_analysis_streaming(self, predictions: Dict[str, Any],
                                      correlations: Dict[str, float],
                                      market_context: str,
                                      language: str = 'id') -> Generator[str, None, None]:
        """Mendapatkan analisis pasar via streaming dari Ollama."""
        if not self.model_name or not self.is_connected():
            yield "Error: Ollama tidak terhubung atau model tidak dipilih."
            return
        prompt = self._create_analysis_prompt(predictions, correlations, market_context, language)
        yield from self.chat_streaming(prompt)

    def get_assistant_response(self, user_question: str, context: str) -> Generator[str, None, None]:
        """
        Membangun persona INVISE AI dan mendapatkan respons chat umum.
        """

        system_persona = f"""
        Anda adalah "INVISE AI", sebuah asisten kecerdasan buatan yang canggih dari sistem INVISE.
        Anda beroperasi di atas model bahasa '{self.model_name}'.
        Anda dibuat oleh tim Golden Delta dari Universitas Sanata Dharma.
        Tugas Anda adalah untuk membantu pengguna dengan menjawab pertanyaan mereka terkait pasar keuangan, analisis data, atau informasi umum dengan menggunakan konteks yang diberikan.
        Selalu pertahankan persona Anda sebagai AI yang berpengetahuan, membantu, dan terintegrasi dengan platform INVISE.
        Jika ada yang menanyakan mengenai seputar anda, anda bisa menjelaskan anda siapa tanpa perlu memberitahukan informasi yang saya berikan dalam konteks.
        """

        full_user_message = f"""
        Konteks pasar saat ini yang saya miliki adalah:
        ---
        {context if context else "Saat ini belum ada data prediksi yang dihasilkan."}
        ---

        Pertanyaan dari pengguna: "{user_question}"
        """

        yield from self.chat_streaming(message=full_user_message, system_prompt=system_persona)

    def _create_analysis_prompt(self, predictions: Dict[str, Any],
                                correlations: Dict[str, float],
                                market_context: str,
                                language: str) -> str:
        """Membuat prompt yang berfokus pada tujuan untuk dua audiens dengan kesimpulan."""
        predicted_return = predictions.get('predicted_return', 0) * 100
        confidence = predictions.get('confidence', 0)
        horizon = predictions.get('prediction_horizon', 1)
        current_price = predictions.get('current_price', 0)
        predicted_price = predictions.get('predicted_price', 0)
        
        top_correlations_str = "Data korelasi tidak tersedia."
        if correlations:
            top_5_corr = list(correlations.items())[:5]
            top_correlations_str = "\n".join([f"  - {asset}: {corr:.3f}" for asset, corr in top_5_corr])
        
        model_performance_str = "Data kinerja model tidak tersedia."
        if 'model_performance' in predictions:
            perf_data = predictions['model_performance']
            model_performance_str = "\n".join([
                f"  - {model}: MAE={data.get('Test_MAE', 0):.2f}, Directional Accuracy={data.get('Directional_Accuracy', 0):.2f}%" 
                for model, data in perf_data.items()
            ])

        if language == 'id':
            return f"""
# PERAN ANDA
Anda adalah seorang **Analis Keuangan Senior di INVISE**. Tugas Anda adalah memberikan analisis pasar yang tajam, berimbang, dan dapat ditindaklanjuti (actionable) berdasarkan data yang disediakan.

# DATA INPUT
- **Harga saat ini** = Rp {current_price:,.0f}
- **Prediksi INVISE ({horizon} Hari):** Arah = {predicted_return:+.2f}% (ke Rp {predicted_price:,.0f}), Kepercayaan = {confidence:.0f}%
- **Korelasi Pasar Teratas:**
{top_correlations_str}
- **Kinerja Model (Backtest):**
{model_performance_str}
- **Konteks Tambahan:** {market_context}

# STRUKTUR LAPORAN YANG DIMINTA
Buat analisis dalam tiga bagian yang jelas:

### BAGIAN 1: ANALISIS STRATEGIS UNTUK INVESTOR INSTITUSIONAL/PERUSAHAAN
Fokus pada gambaran besar dan implikasi strategis.
- **Tesis Utama:** Apa pandangan utama Anda mengenai pasar dalam {horizon} hari ke depan untuk portofolio besar?
- **Alokasi Aset & Sektor:** Berdasarkan prediksi dan korelasi, apakah ada rekomendasi untuk rotasi sektor atau penyesuaian bobot ekuitas?
- **Manajemen Risiko:** Apa risiko utama yang perlu diwaspadai oleh manajer investasi terkait prediksi ini?

### BAGIAN 2: PANDUAN PRAKTIS UNTUK INVESTOR RETAIL/MASYARAKAT
Gunakan bahasa yang sederhana dan berikan saran yang jelas.
- **Apa Artinya Ini Untuk Saya?:** Jelaskan prediksi {predicted_return:+.2f}% dalam bahasa yang mudah dimengerti. Apakah ini sinyal beli yang kuat, atau perlu waspada?
- **Saran Aksi:** Berikan 1-2 saran konkret. Contoh: "Sebaiknya 'wait and see' dan tunggu konfirmasi," atau "Ini bisa menjadi momentum untuk akumulasi bertahap pada saham blue-chip."
- **Faktor Psikologis:** Ingatkan tentang sentimen pasar umum yang mungkin memengaruhi keputusan (misalnya, jangan panik atau FOMO).

### BAGIAN 3: KONKLUSI DAN POIN KUNCI
Ringkas seluruh analisis dalam beberapa poin.
- **Kesimpulan Utama:** Apa satu kalimat yang merangkum pandangan Anda untuk {horizon} hari ke depan?
- **Poin Paling Penting:** Sebutkan 2-3 poin terpenting yang harus diingat oleh semua investor dari analisis ini.

---
**Gaya Penulisan:** Profesional, berbasis data, dan langsung ke tujuan. Pastikan setiap bagian disesuaikan dengan audiensnya.
"""
        else:
            return f"""
# YOUR ROLE
You are a **Senior Financial Analyst at INVISE**. Your task is to provide a sharp, balanced, and actionable market analysis based on the provided data.

# INPUT DATA
- **IHSG Price Right now** = Rp {current_price:,.0f}
- **INVISE Prediction ({horizon}-Day):** Direction = {predicted_return:+.2f}% (to IDR {predicted_price:,.0f}), Confidence = {confidence:.0f}%
- **Top Market Correlations:**
{top_correlations_str}
- **Model Performance (Backtest):**
{model_performance_str}
- **Additional Context:** {market_context}

# REQUESTED REPORT STRUCTURE
Create the analysis in three clear sections:

### SECTION 1: STRATEGIC ANALYSIS FOR INSTITUTIONAL/CORPORATE INVESTORS
Focus on the big picture and strategic implications.
- **Main Thesis:** What is your primary market view for the next {horizon} day(s) for a large portfolio?
- **Asset & Sector Allocation:** Based on the prediction and correlations, are there any recommendations for sector rotation or equity weight adjustments?
- **Risk Management:** What are the key risks investment managers should be aware of regarding this forecast?

### SECTION 2: PRACTICAL GUIDANCE FOR RETAIL/PUBLIC INVESTORS
Use simple language and provide clear advice.
- **What Does This Mean for Me?:** Explain the {predicted_return:+.2f}% prediction in plain English. Is it a strong buy signal, or a reason for caution?
- **Actionable Advice:** Provide 1-2 concrete suggestions. For example: "It may be prudent to 'wait and see' for confirmation," or "This could be an opportunity for gradual accumulation in blue-chip stocks."
- **Psychological Factors:** Remind them of general market sentiment that might affect decisions (e.g., avoiding panic or FOMO).

### SECTION 3: CONCLUSION AND KEY TAKEAWAYS
Summarize the entire analysis in a few points.
- **Final Conclusion:** What is the one sentence that encapsulates your outlook for the next {horizon} day(s)?
- **Most Important Points:** List the 2-3 most critical takeaways that all investors should remember from this analysis.

---
**Writing Style:** Professional, data-driven, and to the point. Ensure each section is tailored to its specific audience.
"""

def display_streaming_response(ollama_client: 'OllamaChat', message: str) -> str:
    placeholder = st.empty()
    full_response = ""
    try:
        for chunk in ollama_client.chat_streaming(message):
            full_response += chunk
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)
    except Exception as e:
        st.error(f"Gagal menampilkan respons streaming: {e}")
        return ""
    return full_response