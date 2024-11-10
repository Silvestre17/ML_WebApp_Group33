# =============================================================================
# Machile Learning Project | Group 33 | 2024/25 | MSc in Data Science and Advanced Analytics
#        André Silvestre, 20240502 | João Henriques, 20240499 | Simone Genovese, 20241459
#        Steven Carlson, 20240554 | Vinícius Pinto, 20211682 | Zofia Wojcik, 20240654
# =============================================================================
# streamlit run mlproject_group37_streamlit.py

# Import necessary libraries
import streamlit as st
import pandas as pd
import mlproject_dashboard_functions
import plotly.express as px

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---- Streamlit Page Config ----
st.set_page_config(page_title='ML Project | WCB - Group 33 | 2024/25',
                   page_icon='https://upload.wikimedia.org/wikipedia/en/6/69/NOVA_IMS_Logo.png',
                   layout='wide',
                   initial_sidebar_state='expanded',
                   menu_items={
                       'Report a bug': 'mailto:20240502@novaims.unl.pt',
                       'About': "# ML Project | Group 33 | 2024/25"
                   })

# ---- CSS Styling ----
with open('style.css') as f:
    st.markdown(f'''<style>{f.read()}
                    /* Change the slider color | Sources: https://discuss.streamlit.io/t/how-to-change-st-sidebar-slider-default-color/3900/2 
                                                           https://discuss.streamlit.io/t/customizing-the-appearance-of-tabs/48913 */
                    div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{{
                        background-color: #084594;
                    }}

                    div.stSlider > div[data-baseweb="slider"] > div > div > div > div{{
                        color: #084594; 
                    }}

                    div.stSlider > div[data-baseweb = "slider"] > div > div {{
                        background: #084594;}}

                    div.stSlider > div[data-baseweb = "slider"] > div[data-testid="stTickBar"] > div {{
                        background: rgb(1 1 1 / 0%); }}

                    .stTabs [data-baseweb="tab"] {{
                        color: #084594;
                    }}

                    .stTabs [data-baseweb="tab-highlight"] {{
                        background-color: #084594;
                    }}

                    button[kind="secondary"] {{
                        border: 1px solid #084594;
                    }}

                    button[kind="secondary"]:hover {{
                        font-weight: bold;
                        color: #084594;
                        border: 2px solid #084594;
                    }}

            </style>''', unsafe_allow_html=True)

st.logo(image='static/640px-HD_transparent_picture.png', icon_image='static/NOVAIMS_Logo.png')

# =============================================================================
# -----------------------------
# Banner Image (Top of the Page)
st.markdown("""
    <style>
        .h1, .h2, .h3, .h4, .h5, .h6, h1, h2, h3, h4, h5, h6 {
            font-weight: bold !important;
        }
        
        .banner {
            width: 117%;
            display: block;
            margin-left: -100px;
            margin-top: -60px;
        }
        .banner img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        @media (max-width: 768px) {
            .banner {
                width: 110%;
                display: block;
                margin-left: -20px;
                margin-top: -60px;
            }
            
            .banner img {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }
    </style>

    <!-- Banner Image -->
    <div class="banner">
        <img src='./app/static/ABCDEats_Banner.png' alt="Banner Image">
    </div>
    """, unsafe_allow_html=True)

# -----------------------------

# =============================================================================
# Resumo do Trabalho
st.markdown("""
    <style>
        @import url("https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css");
            
        .team h1,
        .team h2,
        .team h3,
        .team h4,
        .team h5,
        .team h6 {
            color: #3d392d;
            font-weight: bold;
        }
    
        .team .font-weight-medium {
            font-weight: 700;
        }
    
        .team .bg-light {
            background-color: #f4f8fa !important;
        }
    
        .team .subtitle-title {
            color: rgb(0,24,53);
            line-height: 24px;
            font-size: 26px;
            font-weight: 700;
            margin-top: -10px;
        }
    
        .team .subtitle-names {
            color: rgb(0,24,53);
            line-height: 24px;
            font-size: 14px;
            font-weight: 600;
        }
    
        .team ul {
            margin-top: 30px;
        }
    
        .team h5 {
            line-height: 22px;
            font-size: 18px;
        }
    
        .team ul li a {
            color: #8d97ad;
            padding-right: 15px;
            -webkit-transition: 0.1s ease-in;
            -o-transition: 0.1s ease-in;
            transition: 0.1s ease-in;
        }
    
        .team ul li a:hover {
            -webkit-transform: translate3d(0px, -5px, 0px);
            transform: translate3d(0px, -5px, 0px);
            color: #135C9B;
        }
    
        .team .title {
            margin: 30px 0 0 0;
        }
    
        .team .subtitle {
            margin: 0 0 20px 0;
            font-size: 13px;
        }
        
        .st-emotion-cache-1629p8f a {
            display: none;
            pointer-events: none;
        }
        
        .st-emotion-cache-1629p8f h1, .st-emotion-cache-1629p8f h2, .st-emotion-cache-1629p8f h3, .st-emotion-cache-1629p8f h4,
        .st-emotion-cache-1629p8f h5, .st-emotion-cache-1629p8f h6, .st-emotion-cache-1629p8f span {
            font-weight: bolder;
        }
        
    </style>
    
    <!-- Bibliotecas de Icons-->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <script type="module" src="https://unpkg.com/ionicons@5.5.2/dist/ionicons/ionicons.esm.js"></script>
    <script nomodule src="https://unpkg.com/ionicons@5.5.2/dist/ionicons/ionicons.js"></script>
    
    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.4.1/dist/css/bootstrap.min.css"
          integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh" crossorigin="anonymous">
    
    <script src="https://code.jquery.com/jquery-3.4.1.slim.min.js"
            integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n"
            crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js"
            integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo"
            crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.4.1/dist/js/bootstrap.min.js"
            integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6"
            crossorigin="anonymous"></script>
        
    <div class="py-5 team">
        <div class="container" style="margin-top: -80px">
            <div class="row justify-content-center" style="margin-bottom: -40px">
                <div class="col-md-7 text-center">
                    <h1 class="mb-0 title-contactos"></h1>
                    <p class="subtitle-title">Motivação</p>
                </div>
                <br><br>
                <p style="text-align: justify; max-width: 800px; margin: auto;">
                    No dinâmico mercado das Telecomunicações, a análise de dados provenientes de redes sociais revela-se 
                    um recurso inestimável para as equipas de Marketing e Comunicação.
                    <br><br>
                    Este projeto, destinado a estudantes de Data Science, visa explorar tal potencial. 
                    A relevância destes dados reside na sua capacidade de elucidar as necessidades, 
                    preferências, comportamentos e níveis de satisfação dos utilizadores de uma operadora 
                    de telecomunicações, que opera num ambiente bastante competitivo. 
                    <br><br>
                    Adicionalmente, possibilita a identificação de novas oportunidades, tendências e 
                    desafios inerentes ao setor.
                </p>
                <br><br><br>
            </div>
        </div>
    </div>     
    <footer class="footer" style="visibility: visible; margin-top: 40px;">
        <div class="container">
            <div class="row">
                <div class="col-lg-12 text-center">
                    <p class="company-name" style="color: #d3d3d3;">ML Project | Group 37 © 2024/25</p>
                </div>
            </div>
        </div>
    </footer>
    """, unsafe_allow_html=True)

# =============================================================================
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# --------------------------- Load Data ---------------------------







# =============================================================================
# --------------------------- Sidebar Filters ---------------------------
with st.sidebar:
    st.markdown("""aaaaaaaaaaaaaaaaaaaaaaaa""")



# =============================================================================
# --------------------------- Main Content ---------------------------
# Tabs

tab1, tab2 = st.tabs(["Dashboard", "Data Analysis"])

with tab1:
    st.write("### EDA | Exploratory Data Analysis")