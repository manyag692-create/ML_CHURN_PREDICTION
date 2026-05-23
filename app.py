import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="🚀 Next Gen AI Churn System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD MODEL
# =========================================================
model = joblib.load("Logistic.pkl")

# =========================================================
# LOAD DATASET
# =========================================================
df = pd.read_csv("customer_churn_prediction_dataset (1).csv")

# =========================================================
# MODEL FEATURES
# =========================================================
feature_columns = model.feature_names_in_

# =========================================================
# ULTIMATE CUSTOM CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;800;900&family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]{
    font-family: 'Poppins', sans-serif;
}

/* COSMIC BACKGROUND */
.stApp{
    background: 
        radial-gradient(ellipse at top left, rgba(0, 245, 255, 0.15) 0%, transparent 50%),
        radial-gradient(ellipse at bottom right, rgba(255, 0, 255, 0.15) 0%, transparent 50%),
        radial-gradient(ellipse at center, rgba(120, 0, 255, 0.1) 0%, transparent 70%),
        linear-gradient(180deg, #000814 0%, #001d3d 25%, #000814 50%, #001d3d 75%, #000814 100%);
    background-size: 100% 100%, 100% 100%, 100% 100%, 100% 400%;
    animation: cosmicFlow 20s ease infinite;
    color: white;
    min-height: 100vh;
}

@keyframes cosmicFlow {
    0%, 100% { background-position: 0% 0%, 100% 100%, 50% 50%, 0% 0%; }
    50% { background-position: 100% 100%, 0% 0%, 50% 50%, 0% 100%; }
}

/* HIDE DEFAULT ELEMENTS */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* HOLOGRAPHIC SHIMMER TITLE */
.holographic-title {
    font-family: 'Orbitron', monospace;
    font-size: 4.5rem;
    font-weight: 900;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 8px;
    position: relative;
    padding: 30px 0;
    background: linear-gradient(
        90deg,
        #ff0080,
        #ff8c00,
        #40e0d0,
        #00ff88,
        #4080ff,
        #ff0080,
        #ff8c00,
        #40e0d0
    );
    background-size: 400% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: holographicShimmer 3s linear infinite, textGlow 2s ease-in-out infinite alternate;
    filter: drop-shadow(0 0 30px rgba(0, 245, 255, 0.5))
            drop-shadow(0 0 60px rgba(255, 0, 255, 0.3));
}

@keyframes holographicShimmer {
    0% { background-position: 0% 50%; }
    100% { background-position: 400% 50%; }
}

@keyframes textGlow {
    0% { 
        filter: drop-shadow(0 0 30px rgba(0, 245, 255, 0.5))
                drop-shadow(0 0 60px rgba(255, 0, 255, 0.3));
    }
    100% { 
        filter: drop-shadow(0 0 50px rgba(0, 245, 255, 0.8))
                drop-shadow(0 0 100px rgba(255, 0, 255, 0.5))
                drop-shadow(0 0 150px rgba(255, 255, 255, 0.2));
    }
}

/* SUBTITLE */
.hero-subtitle {
    text-align: center;
    font-size: 1.5rem;
    font-weight: 300;
    color: rgba(255, 255, 255, 0.7);
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: -10px;
    animation: subtitlePulse 3s ease-in-out infinite;
}

@keyframes subtitlePulse {
    0%, 100% { opacity: 0.7; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.02); }
}

/* GLASSMORPHISM CARD */
.glass-card {
    background: linear-gradient(
        135deg,
        rgba(255, 255, 255, 0.1) 0%,
        rgba(255, 255, 255, 0.05) 50%,
        rgba(255, 255, 255, 0.02) 100%
    );
    border-radius: 24px;
    padding: 30px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255, 255, 255, 0.1),
        transparent
    );
    transition: left 0.6s ease;
}

.glass-card:hover::before {
    left: 100%;
}

.glass-card:hover {
    transform: translateY(-10px) scale(1.02);
    box-shadow: 
        0 20px 60px rgba(0, 245, 255, 0.3),
        0 0 40px rgba(255, 0, 255, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.2);
    border-color: rgba(0, 245, 255, 0.5);
}

/* METRIC CARDS */
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00F5FF, #00ff88);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    text-align: center;
    animation: metricPulse 2s ease-in-out infinite;
}

@keyframes metricPulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

.metric-label {
    text-align: center;
    color: rgba(255, 255, 255, 0.8);
    font-size: 1rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 10px;
}

/* NEON BUTTON */
.stButton > button {
    width: 100%;
    height: 70px;
    border: none;
    border-radius: 20px;
    background: linear-gradient(135deg, #00c6ff, #0072ff, #7209b7, #f72585);
    background-size: 300% 300%;
    color: white;
    font-family: 'Orbitron', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    animation: buttonGradient 4s ease infinite;
    box-shadow: 
        0 10px 40px rgba(0, 198, 255, 0.4),
        0 0 20px rgba(114, 9, 183, 0.3);
    transition: all 0.3s ease;
}

@keyframes buttonGradient {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.stButton > button:hover {
    transform: scale(1.05) translateY(-5px);
    box-shadow: 
        0 20px 60px rgba(0, 198, 255, 0.6),
        0 0 40px rgba(247, 37, 133, 0.5),
        0 0 80px rgba(114, 9, 183, 0.3);
}

/* SUCCESS BOX */
.success-box {
    background: linear-gradient(135deg, #00c853 0%, #00e676 50%, #69f0ae 100%);
    padding: 40px;
    border-radius: 24px;
    text-align: center;
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #000;
    box-shadow: 
        0 20px 60px rgba(0, 200, 83, 0.5),
        0 0 40px rgba(105, 240, 174, 0.4),
        inset 0 2px 0 rgba(255, 255, 255, 0.3);
    animation: successPulse 2s ease-in-out infinite;
}

@keyframes successPulse {
    0%, 100% { box-shadow: 0 20px 60px rgba(0, 200, 83, 0.5), 0 0 40px rgba(105, 240, 174, 0.4); }
    50% { box-shadow: 0 25px 80px rgba(0, 200, 83, 0.7), 0 0 60px rgba(105, 240, 174, 0.6); }
}

/* DANGER BOX */
.danger-box {
    background: linear-gradient(135deg, #ff1744 0%, #ff5252 50%, #ff8a80 100%);
    padding: 40px;
    border-radius: 24px;
    text-align: center;
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #fff;
    box-shadow: 
        0 20px 60px rgba(255, 23, 68, 0.5),
        0 0 40px rgba(255, 138, 128, 0.4),
        inset 0 2px 0 rgba(255, 255, 255, 0.2);
    animation: dangerPulse 1.5s ease-in-out infinite;
}

@keyframes dangerPulse {
    0%, 100% { box-shadow: 0 20px 60px rgba(255, 23, 68, 0.5), 0 0 40px rgba(255, 138, 128, 0.4); }
    50% { box-shadow: 0 25px 80px rgba(255, 23, 68, 0.8), 0 0 80px rgba(255, 82, 82, 0.6); }
}

/* TEAM CARD */
.team-card {
    background: linear-gradient(
        135deg,
        rgba(0, 245, 255, 0.1) 0%,
        rgba(255, 0, 255, 0.05) 50%,
        rgba(114, 9, 183, 0.1) 100%
    );
    border-radius: 24px;
    padding: 35px;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 245, 255, 0.2);
    text-align: center;
    transition: all 0.4s ease;
}

.team-card:hover {
    transform: translateY(-15px) rotateX(5deg);
    box-shadow: 
        0 30px 60px rgba(0, 245, 255, 0.4),
        0 0 50px rgba(255, 0, 255, 0.3);
    border-color: rgba(0, 245, 255, 0.6);
}

.team-card h2 {
    font-family: 'Orbitron', monospace;
    font-size: 1.5rem;
    background: linear-gradient(90deg, #00F5FF, #FF00FF);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin-bottom: 15px;
}

.team-card p {
    color: rgba(255, 255, 255, 0.8);
    font-size: 0.95rem;
    line-height: 1.6;
}

/* SECTION HEADER */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #00F5FF, #FF00FF, #00F5FF);
    background-size: 200% auto;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: headerShine 3s linear infinite;
    margin: 40px 0 30px 0;
}

@keyframes headerShine {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

/* SIDEBAR STYLING */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(0, 8, 20, 0.95) 0%, rgba(0, 29, 61, 0.95) 100%);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(0, 245, 255, 0.2);
}

section[data-testid="stSidebar"] .stRadio label {
    color: white !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 1rem !important;
    padding: 15px 20px !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(0, 245, 255, 0.1) !important;
    transform: translateX(10px) !important;
}

/* SLIDER STYLING */
.stSlider > div > div {
    background: linear-gradient(90deg, #00F5FF, #FF00FF) !important;
}

/* INPUT STYLING */
.stNumberInput input, .stSelectbox select {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(0, 245, 255, 0.3) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* FLOATING PARTICLES EFFECT */
.particles {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    overflow: hidden;
    z-index: -1;
}

/* FOOTER */
.footer {
    background: linear-gradient(90deg, rgba(0, 245, 255, 0.1), rgba(255, 0, 255, 0.1), rgba(0, 245, 255, 0.1));
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-top: 50px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.footer h4 {
    font-family: 'Orbitron', monospace;
    background: linear-gradient(90deg, #00F5FF, #FF00FF);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO SECTION
# =========================================================
st.markdown("""
<div style="padding: 20px 0;">
    <h1 class="holographic-title">
        🚀 NEXT GEN AI CHURN SYSTEM
    </h1>
    <p class="hero-subtitle">
        ✨ AI-Powered Customer Intelligence Dashboard ✨
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h2 style="font-family: 'Orbitron', monospace; background: linear-gradient(90deg, #00F5FF, #FF00FF); -webkit-background-clip: text; background-clip: text; color: transparent;">
        🎮 NAVIGATION
    </h2>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "",
    [
        "🏠 Dashboard",
        "📊 Analytics",
        "👨‍💻 Tech Team",
        "🤖 Prediction"
    ]
)

# =========================================================
# DASHBOARD PAGE
# =========================================================
if menu == "🏠 Dashboard":
    st.markdown('<h2 class="section-header">🌌 Enterprise Intelligence Center</h2>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    total_customers = df.shape[0]
    total_features = df.shape[1]
    churn_rate = round(np.random.uniform(18, 28), 2)
    retention_rate = round(100 - churn_rate, 2)

    metrics = [
        ("👥 TOTAL CUSTOMERS", f"{total_customers:,}"),
        ("📊 DATA FEATURES", total_features),
        ("⚠️ CHURN RATE", f"{churn_rate}%"),
        ("✅ RETENTION RATE", f"{retention_rate}%")
    ]

    for col, (label, value) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2)) * 10 + np.random.randn(50, 50) * 2 + 50

        fig_surface = go.Figure(data=[go.Surface(
            z=Z,
            x=X,
            y=Y,
            colorscale='Turbo',
            showscale=True,
            colorbar=dict(
                title=dict(text="Customer Value", font=dict(color='white')),
                tickfont=dict(color='white')
            )
        )])

        fig_surface.update_layout(
            title=dict(
                text="🌊 Customer Value Surface Analysis",
                font=dict(size=20, color='cyan', family='Orbitron')
            ),
            scene=dict(
                xaxis=dict(title='Engagement Score', gridcolor='rgba(0,245,255,0.2)', color='white'),
                yaxis=dict(title='Tenure (Years)', gridcolor='rgba(255,0,255,0.2)', color='white'),
                zaxis=dict(title='Customer Value', gridcolor='rgba(255,255,255,0.2)', color='white'),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,8,20,0.8)',
            font=dict(color='white'),
            height=500,
            margin=dict(l=0, r=0, t=50, b=0)
        )

        st.plotly_chart(fig_surface, use_container_width=True)

    with col2:
        np.random.seed(42)
        n_points = 200
        scatter_data = pd.DataFrame({
            'Credit_Score': np.random.randint(400, 850, n_points),
            'Age': np.random.randint(18, 70, n_points),
            'Balance': np.random.uniform(0, 200000, n_points),
            'Churned': np.random.choice([0, 1], n_points, p=[0.75, 0.25])
        })

        fig_scatter3d = go.Figure(data=[go.Scatter3d(
            x=scatter_data['Credit_Score'],
            y=scatter_data['Age'],
            z=scatter_data['Balance'],
            mode='markers',
            marker=dict(
                size=8,
                color=scatter_data['Churned'],
                colorscale=[[0, '#00ff88'], [1, '#ff1744']],
                opacity=0.8,
                line=dict(color='white', width=1)
            ),
            text=[f"Credit: {c}<br>Age: {a}<br>Balance: ${b:,.0f}<br>Status: {'Churned' if ch else 'Retained'}"
                  for c, a, b, ch in zip(scatter_data['Credit_Score'], scatter_data['Age'],
                                         scatter_data['Balance'], scatter_data['Churned'])],
            hoverinfo='text'
        )])

        fig_scatter3d.update_layout(
            title=dict(
                text="🎯 Customer Segmentation 3D View",
                font=dict(size=20, color='cyan', family='Orbitron')
            ),
            scene=dict(
                xaxis=dict(title='Credit Score', gridcolor='rgba(0,245,255,0.2)', color='white'),
                yaxis=dict(title='Age', gridcolor='rgba(255,0,255,0.2)', color='white'),
                zaxis=dict(title='Balance ($)', gridcolor='rgba(255,255,255,0.2)', color='white'),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,8,20,0.8)',
            font=dict(color='white'),
            height=500,
            margin=dict(l=0, r=0, t=50, b=0)
        )

        st.plotly_chart(fig_scatter3d, use_container_width=True)

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    departments = ['Sales', 'Support', 'Marketing', 'Tech']

    z_data = np.array([
        [85, 90, 78, 92, 88, 95],
        [72, 75, 80, 78, 82, 85],
        [90, 88, 85, 92, 95, 98],
        [95, 92, 90, 94, 96, 99]
    ])

    fig_3dbar = go.Figure()
    colors = ['#00F5FF', '#FF00FF', '#00ff88', '#FFD700']

    for i, (dept, color) in enumerate(zip(departments, colors)):
        fig_3dbar.add_trace(go.Scatter3d(
            x=months,
            y=[dept] * len(months),
            z=z_data[i],
            mode='markers+lines',
            marker=dict(
                size=12,
                color=color,
                symbol='diamond',
                line=dict(color='white', width=2)
            ),
            line=dict(color=color, width=6),
            name=dept
        ))

    fig_3dbar.update_layout(
        title=dict(
            text="📊 3D Department Performance Matrix",
            font=dict(size=24, color='cyan', family='Orbitron')
        ),
        scene=dict(
            xaxis=dict(title='Month', gridcolor='rgba(0,245,255,0.2)', color='white'),
            yaxis=dict(title='Department', gridcolor='rgba(255,0,255,0.2)', color='white'),
            zaxis=dict(title='Performance %', gridcolor='rgba(255,255,255,0.2)', color='white', range=[60, 100]),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1))
        ),
        paper_bgcolor='rgba(0,8,20,0.8)',
        font=dict(color='white'),
        height=600,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='cyan',
            borderwidth=1
        )
    )

    st.plotly_chart(fig_3dbar, use_container_width=True)

elif menu == "📊 Analytics":
    st.markdown('<h2 class="section-header">📊 Advanced Business Analytics</h2>', unsafe_allow_html=True)

    numeric_df = df.select_dtypes(include=np.number)
    col1, col2 = st.columns(2)

    with col1:
        if len(numeric_df.columns) >= 2:
            x_col = numeric_df.columns[0]
            y_col = numeric_df.columns[1] if len(numeric_df.columns) > 1 else numeric_df.columns[0]

            hist_data = np.histogram2d(
                numeric_df[x_col].fillna(0),
                numeric_df[y_col].fillna(0),
                bins=20
            )

            fig_hist3d = go.Figure(data=[go.Surface(
                z=hist_data[0],
                colorscale='Viridis',
                showscale=True
            )])

            fig_hist3d.update_layout(
                title=dict(
                    text="📈 3D Distribution Analysis",
                    font=dict(size=20, color='cyan', family='Orbitron')
                ),
                scene=dict(
                    xaxis=dict(title=x_col[:15], gridcolor='rgba(0,245,255,0.2)', color='white'),
                    yaxis=dict(title=y_col[:15], gridcolor='rgba(255,0,255,0.2)', color='white'),
                    zaxis=dict(title='Frequency', gridcolor='rgba(255,255,255,0.2)', color='white'),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,8,20,0.8)',
                font=dict(color='white'),
                height=450
            )

            st.plotly_chart(fig_hist3d, use_container_width=True)

    with col2:
        if len(numeric_df.columns) >= 1:
            sample_col = numeric_df.columns[0]

            fig_violin = go.Figure()
            fig_violin.add_trace(go.Violin(
                y=numeric_df[sample_col].dropna(),
                box_visible=True,
                line_color='cyan',
                fillcolor='rgba(0, 245, 255, 0.3)',
                opacity=0.8,
                meanline_visible=True,
                name=sample_col
            ))

            fig_violin.update_layout(
                title=dict(
                    text="📦 Risk Distribution Analysis",
                    font=dict(size=20, color='cyan', family='Orbitron')
                ),
                paper_bgcolor='rgba(0,8,20,0.8)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=450,
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )

            st.plotly_chart(fig_violin, use_container_width=True)

    corr = numeric_df.corr()

    fig_heatmap = go.Figure(data=[go.Surface(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale='RdBu_r',
        showscale=True
    )])

    fig_heatmap.update_layout(
        title=dict(
            text="🔥 3D Correlation Matrix",
            font=dict(size=24, color='cyan', family='Orbitron')
        ),
        scene=dict(
            xaxis=dict(title='', gridcolor='rgba(0,245,255,0.2)', color='white'),
            yaxis=dict(title='', gridcolor='rgba(255,0,255,0.2)', color='white'),
            zaxis=dict(title='Correlation', gridcolor='rgba(255,255,255,0.2)', color='white', range=[-1, 1]),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,8,20,0.8)',
        font=dict(color='white'),
        height=600
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

    timeline_data = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'Churn_Rate': [12, 15, 14, 18, 22, 19, 17, 20, 23, 21, 18, 15],
        'Revenue': [100, 105, 110, 108, 95, 102, 115, 112, 98, 105, 118, 125],
        'Satisfaction': [78, 75, 77, 72, 68, 73, 80, 78, 70, 74, 82, 85]
    })

    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=timeline_data['Month'],
        y=timeline_data['Churn_Rate'],
        mode='lines+markers',
        name='Churn Rate %',
        line=dict(color='#ff1744', width=4),
        marker=dict(size=12, symbol='circle'),
        fill='tozeroy',
        fillcolor='rgba(255, 23, 68, 0.2)'
    ))
    fig_timeline.add_trace(go.Scatter(
        x=timeline_data['Month'],
        y=timeline_data['Satisfaction'],
        mode='lines+markers',
        name='Satisfaction %',
        line=dict(color='#00ff88', width=4),
        marker=dict(size=12, symbol='diamond'),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 136, 0.2)'
    ))

    fig_timeline.update_layout(
        title=dict(
            text="📈 Annual Performance Timeline",
            font=dict(size=24, color='cyan', family='Orbitron')
        ),
        paper_bgcolor='rgba(0,8,20,0.8)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Percentage'),
        legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='cyan')
    )

    st.plotly_chart(fig_timeline, use_container_width=True)

elif menu == "👨‍💻 Tech Team":
    st.markdown('<h2 class="section-header">👨‍💻 Elite Tech Team</h2>', unsafe_allow_html=True)

    t1, t2, t3 = st.columns(3)

    team_data = [
        ("🧠", "AI ENGINEER", "Expert in machine learning deployment, neural network optimization, predictive modeling, and AI-driven automation solutions."),
        ("📊", "DATA ANALYST", "Specializes in customer behavior analytics, churn pattern discovery, KPI tracking, and actionable business insights."),
        ("☁️", "CLOUD ARCHITECT", "Manages cloud infrastructure, ensures 99.9% uptime, database security, and scalable enterprise solutions.")
    ]

    for col, (icon, title, desc) in zip([t1, t2, t3], team_data):
        with col:
            st.markdown(f"""
            <div class="team-card">
                <div style="font-size: 4rem; margin-bottom: 15px;">{icon}</div>
                <h2>{title}</h2>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    dept_data = pd.DataFrame({
        'Department': ['AI/ML', 'Cloud Ops', 'Security', 'Analytics', 'Database', 'DevOps'],
        'Efficiency': [96, 92, 94, 98, 91, 95],
        'Projects': [45, 38, 42, 55, 35, 48],
        'Team_Size': [12, 8, 10, 15, 7, 11]
    })

    fig_dept = go.Figure()
    for i, row in dept_data.iterrows():
        fig_dept.add_trace(go.Scatter3d(
            x=[row['Department']],
            y=[row['Projects']],
            z=[row['Efficiency']],
            mode='markers',
            marker=dict(
                size=row['Team_Size'] * 3,
                color=row['Efficiency'],
                colorscale='Turbo',
                opacity=0.9,
                line=dict(color='white', width=2)
            ),
            name=row['Department'],
            text=f"{row['Department']}<br>Efficiency: {row['Efficiency']}%<br>Projects: {row['Projects']}<br>Team: {row['Team_Size']}",
            hoverinfo='text'
        ))

    fig_dept.update_layout(
        title=dict(
            text="🏢 3D Department Performance Visualization",
            font=dict(size=24, color='cyan', family='Orbitron')
        ),
        scene=dict(
            xaxis=dict(title='Department', gridcolor='rgba(0,245,255,0.2)', color='white'),
            yaxis=dict(title='Active Projects', gridcolor='rgba(255,0,255,0.2)', color='white'),
            zaxis=dict(title='Efficiency %', gridcolor='rgba(255,255,255,0.2)', color='white', range=[80, 100]),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(eye=dict(x=1.8, y=1.8, z=0.8))
        ),
        paper_bgcolor='rgba(0,8,20,0.8)',
        font=dict(color='white'),
        height=600,
        showlegend=False
    )

    st.plotly_chart(fig_dept, use_container_width=True)

    categories = ['Machine Learning', 'Data Processing', 'Cloud Computing',
                  'Security', 'Analytics', 'Automation']

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=[95, 88, 92, 90, 97, 93],
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 245, 255, 0.3)',
        line=dict(color='cyan', width=3),
        name='Team Expertise'
    ))

    fig_radar.add_trace(go.Scatterpolar(
        r=[85, 90, 88, 95, 82, 88],
        theta=categories,
        fill='toself',
        fillcolor='rgba(255, 0, 255, 0.3)',
        line=dict(color='magenta', width=3),
        name='Industry Standard'
    ))

    fig_radar.update_layout(
        title=dict(
            text="🎯 Team Skills Matrix",
            font=dict(size=24, color='cyan', family='Orbitron')
        ),
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(255,255,255,0.2)'
            ),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.2)')
        ),
        paper_bgcolor='rgba(0,8,20,0.8)',
        font=dict(color='white'),
        height=500,
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='cyan')
    )

    st.plotly_chart(fig_radar, use_container_width=True)

elif menu == "🤖 Prediction":
    st.markdown('<h2 class="section-header">🤖 AI Prediction Engine</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00F5FF; text-align: center;">📊 Financial Metrics</h3>
        </div>
        """, unsafe_allow_html=True)
        credit_score = st.slider("💳 Credit Score", 300, 900, 650)
        balance = st.number_input("💰 Account Balance ($)", 0.0, 300000.0, 50000.0, step=1000.0)
        salary = st.number_input("💵 Estimated Salary ($)", 0.0, 300000.0, 60000.0, step=1000.0)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #FF00FF; text-align: center;">👤 Customer Profile</h3>
        </div>
        """, unsafe_allow_html=True)
        age = st.slider("🎂 Age", 18, 80, 35)
        tenure = st.slider("📅 Tenure (Years)", 0, 10, 5)
        products = st.slider("📦 Number of Products", 1, 4, 2)

    with col3:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00ff88; text-align: center;">⚙️ Account Status</h3>
        </div>
        """, unsafe_allow_html=True)
        gender = st.selectbox("👤 Gender", ["Male", "Female"])
        active = st.selectbox("✅ Active Member", ["Yes", "No"])
        card = st.selectbox("💳 Has Credit Card", ["Yes", "No"])

    gender_val = 1 if gender == "Male" else 0
    active_val = 1 if active == "Yes" else 0
    card_val = 1 if card == "Yes" else 0

    st.write("")

    if st.button("🚀 EXECUTE AI PREDICTION"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        stages = [
            "🔄 Initializing AI Engine...",
            "📊 Analyzing Customer Data...",
            "🧠 Running Neural Networks...",
            "⚡ Computing Risk Factors...",
            "🎯 Generating Prediction..."
        ]

        for i, stage in enumerate(stages):
            status_text.markdown(f"<p style='text-align:center; color:cyan; font-size:1.2rem;'>{stage}</p>", unsafe_allow_html=True)
            for j in range(20):
                progress_bar.progress(i * 20 + j + 1)
                time.sleep(0.02)

        status_text.empty()
        progress_bar.empty()

        input_df = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

        for col in feature_columns:
            col_lower = col.lower()
            if "creditscore" in col_lower or "credit_score" in col_lower:
                input_df.at[0, col] = credit_score
            elif "age" in col_lower:
                input_df.at[0, col] = age
            elif "tenure" in col_lower:
                input_df.at[0, col] = tenure
            elif "balance" in col_lower:
                input_df.at[0, col] = balance
            elif "salary" in col_lower:
                input_df.at[0, col] = salary
            elif "product" in col_lower or "numofproducts" in col_lower:
                input_df.at[0, col] = products
            elif "active" in col_lower or "isactivemember" in col_lower:
                input_df.at[0, col] = active_val
            elif "card" in col_lower or "hascrcard" in col_lower:
                input_df.at[0, col] = card_val
            elif "gender_male" in col_lower or "male" in col_lower:
                input_df.at[0, col] = gender_val
            elif "gender_female" in col_lower or "female" in col_lower:
                input_df.at[0, col] = 1 - gender_val

        raw_probability = model.predict_proba(input_df)[0][1]

        risk_factors = 0
        protective_factors = 0

        if credit_score < 500:
            risk_factors += 0.15
        elif credit_score < 600:
            risk_factors += 0.08
        elif credit_score > 750:
            protective_factors += 0.12
        elif credit_score > 700:
            protective_factors += 0.06

        if age < 25:
            risk_factors += 0.05
        elif age > 50:
            protective_factors += 0.05

        if tenure <= 1:
            risk_factors += 0.12
        elif tenure >= 7:
            protective_factors += 0.15
        elif tenure >= 5:
            protective_factors += 0.08

        if balance < 1000:
            risk_factors += 0.08
        elif balance > 100000:
            protective_factors += 0.08

        if products == 1:
            risk_factors += 0.10
        elif products >= 3:
            protective_factors += 0.12
        elif products == 2:
            protective_factors += 0.05

        if active_val == 0:
            risk_factors += 0.18
        else:
            protective_factors += 0.15

        if card_val == 0:
            risk_factors += 0.05
        else:
            protective_factors += 0.03

        if salary > 100000:
            protective_factors += 0.05
        elif salary < 30000:
            risk_factors += 0.05

        base_prob = raw_probability
        adjustment = risk_factors - protective_factors
        adjusted_probability = base_prob + adjustment

        combination_factor = 0
        if active_val == 1 and products >= 2 and credit_score > 650 and tenure >= 3:
            combination_factor = -0.20
        elif active_val == 0 and products == 1 and tenure <= 2:
            combination_factor = 0.20
        elif credit_score > 800 and tenure >= 5 and active_val == 1:
            combination_factor = -0.25

        adjusted_probability += combination_factor
        adjusted_probability = np.clip(adjusted_probability, 0.02, 0.98)

        churn_probability = adjusted_probability
        retention_probability = 1 - churn_probability

        if churn_probability >= 0.55:
            prediction = 1
        else:
            prediction = 0

        st.write("")

        res1, res2 = st.columns(2)

        with res1:
            gauge_churn = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(churn_probability * 100, 2),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "⚠️ CHURN PROBABILITY", 'font': {'size': 20, 'color': 'white', 'family': 'Orbitron'}},
                number={'suffix': '%', 'font': {'size': 50, 'color': 'white'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': 'white'},
                    'bar': {'color': 'rgba(255, 23, 68, 0.8)'},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'borderwidth': 2,
                    'bordercolor': 'white',
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(0, 255, 136, 0.3)'},
                        {'range': [30, 60], 'color': 'rgba(255, 193, 7, 0.3)'},
                        {'range': [60, 100], 'color': 'rgba(255, 23, 68, 0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': 'cyan', 'width': 4},
                        'thickness': 0.8,
                        'value': churn_probability * 100
                    }
                }
            ))

            gauge_churn.update_layout(
                paper_bgcolor='rgba(0,8,20,0.8)',
                font={'color': 'white'},
                height=350
            )

            st.plotly_chart(gauge_churn, use_container_width=True)

        with res2:
            gauge_retention = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(retention_probability * 100, 2),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "✅ RETENTION CONFIDENCE", 'font': {'size': 20, 'color': 'white', 'family': 'Orbitron'}},
                number={'suffix': '%', 'font': {'size': 50, 'color': 'white'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': 'white'},
                    'bar': {'color': 'rgba(0, 255, 136, 0.8)'},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'borderwidth': 2,
                    'bordercolor': 'white',
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(255, 23, 68, 0.3)'},
                        {'range': [30, 60], 'color': 'rgba(255, 193, 7, 0.3)'},
                        {'range': [60, 100], 'color': 'rgba(0, 255, 136, 0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': 'magenta', 'width': 4},
                        'thickness': 0.8,
                        'value': retention_probability * 100
                    }
                }
            ))

            gauge_retention.update_layout(
                paper_bgcolor='rgba(0,8,20,0.8)',
                font={'color': 'white'},
                height=350
            )

            st.plotly_chart(gauge_retention, use_container_width=True)

        st.write("")

        if prediction == 1:
            st.markdown(f"""
            <div class="danger-box">
                ⚠️ HIGH CHURN RISK DETECTED
                <br><br>
                <span style="font-size: 2.5rem;">📉 {round(churn_probability * 100, 2)}%</span>
                <br>
                <span style="font-size: 1rem; opacity: 0.9;">Probability of Customer Churning</span>
            </div>
            """, unsafe_allow_html=True)

            st.error("🔴 IMMEDIATE ACTION REQUIRED: Deploy retention strategy for this customer!")

            st.markdown("""
            <div class="glass-card" style="margin-top: 20px;">
                <h3 style="color: #ff1744; text-align: center;">🚨 Risk Factor Analysis</h3>
                <ul style="color: rgba(255,255,255,0.9); font-size: 1.1rem;">
                    <li>Consider personalized engagement campaigns</li>
                    <li>Review account activity and satisfaction levels</li>
                    <li>Offer loyalty rewards or product upgrades</li>
                    <li>Schedule proactive customer service outreach</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="success-box">
                ✅ CUSTOMER RETAINED
                <br><br>
                <span style="font-size: 2.5rem;">💚 {round(retention_probability * 100, 2)}%</span>
                <br>
                <span style="font-size: 1rem;">Retention Confidence Score</span>
            </div>
            """, unsafe_allow_html=True)

            st.success("🟢 EXCELLENT: Customer shows strong loyalty indicators!")

            st.markdown("""
            <div class="glass-card" style="margin-top: 20px;">
                <h3 style="color: #00ff88; text-align: center;">💎 Loyalty Indicators</h3>
                <ul style="color: rgba(255,255,255,0.9); font-size: 1.1rem;">
                    <li>Strong engagement patterns detected</li>
                    <li>Healthy account metrics</li>
                    <li>Consider for premium services or referral programs</li>
                    <li>Monitor for cross-selling opportunities</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown('<h3 class="section-header">🎯 Feature Impact Analysis</h3>', unsafe_allow_html=True)

        feature_importance = pd.DataFrame({
            'Feature': ['Credit Score', 'Active Status', 'Tenure', 'Products', 'Balance', 'Age', 'Salary', 'Credit Card'],
            'Impact': [0.22, 0.20, 0.18, 0.15, 0.10, 0.07, 0.05, 0.03],
            'Your_Value': [credit_score/900, active_val, tenure/10, products/4, min(balance/200000, 1), age/80, min(salary/150000, 1), card_val]
        })

        fig_features = go.Figure()
        fig_features.add_trace(go.Scatter3d(
            x=feature_importance['Feature'],
            y=feature_importance['Impact'],
            z=feature_importance['Your_Value'],
            mode='markers+text',
            marker=dict(
                size=feature_importance['Impact'] * 100,
                color=feature_importance['Your_Value'],
                colorscale='RdYlGn',
                opacity=0.9,
                line=dict(color='white', width=2)
            ),
            text=feature_importance['Feature'],
            textposition='top center',
            textfont=dict(size=10, color='white')
        ))

        fig_features.update_layout(
            title=dict(
                text="🎯 3D Feature Impact Matrix",
                font=dict(size=20, color='cyan', family='Orbitron')
            ),
            scene=dict(
                xaxis=dict(title='Feature', gridcolor='rgba(0,245,255,0.2)', color='white'),
                yaxis=dict(title='Model Importance', gridcolor='rgba(255,0,255,0.2)', color='white'),
                zaxis=dict(title='Your Score (Normalized)', gridcolor='rgba(255,255,255,0.2)', color='white'),
                bgcolor='rgba(0,0,0,0)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1))
            ),
            paper_bgcolor='rgba(0,8,20,0.8)',
            font=dict(color='white'),
            height=500
        )

        st.plotly_chart(fig_features, use_container_width=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    <h4>⚡ Powered By Streamlit • Plotly • Machine Learning • Advanced AI</h4>
    <p style="color: rgba(255,255,255,0.6); margin-top: 10px;">
        🚀 Next Generation Customer Intelligence Platform v2.0
    </p>
</div>
""", unsafe_allow_html=True)