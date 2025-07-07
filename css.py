def load_css():
    """Mengembalikan string berisi semua CSS kustom untuk aplikasi."""
    return """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
    
    /* === RESET & BASE STYLING === */
    * {
        box-sizing: border-box;
    }
    
    .main .block-container {
        padding: 2rem 2.5rem;
        max-width: 95%;
        margin: 0 auto;
    }
    
    .stApp, .stApp * {
        font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }

    /* === HEADER UTAMA === */
    .main-header {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        margin-bottom: 3rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 100" fill="rgba(255,255,255,0.1)"><polygon points="0,0 1000,100 1000,0"/></svg>');
        background-size: cover;
    }
    
    .main-header h1 {
        color: white !important;
        font-weight: 700;
        font-size: 3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 1rem !important;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95) !important;
        font-size: 1.2rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }

    /* === SIDEBAR STYLING === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%) !important;
        border-right: 2px solid rgba(102, 126, 234, 0.1) !important;
        box-shadow: 5px 0 20px rgba(0,0,0,0.05);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }
    
    /* Styling untuk judul sidebar */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    /* Styling untuk button di sidebar */
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        width: 100%;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    [data-testid="stSidebar"] .stButton > button:active {
        transform: translateY(0);
    }
    
    [data-testid="stSidebar"] .stButton > button:disabled {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e0 100%) !important;
        color: #a0aec0 !important;
        box-shadow: none !important;
        transform: none !important;
        cursor: not-allowed;
    }

    /* === PREDICTION CARDS === */
    .prediction-card {
        padding: 2.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        color: white;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
    }
    
    .prediction-card h2 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin: 1rem 0 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .prediction-card h5 {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        opacity: 0.9;
        margin-bottom: 0.5rem !important;
    }
    
    .up { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
    .down { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }
    .neutral { background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%); }

    /* === TAB STYLING === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 12px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 52px;
        padding: 0 28px;
        border-radius: 12px;
        background-color: transparent;
        color: #6b7280;
        border: none;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        transform: translateY(-2px);
    }

    /* === METRICS STYLING === */
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    /* === SCROLLBAR STYLING === */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 10px; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%); }
</style>
"""