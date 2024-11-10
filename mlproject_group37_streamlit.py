# =============================================================================
# Machile Learning Project | Group 33 | 2024/25 | MSc in Data Science and Advanced Analytics
#        André Silvestre, 20240502 | João Henriques, 20240499 | Simone Genovese, 20241459
#        Steven Carlson, 20240554 | Vinícius Pinto, 20211682 | Zofia Wojcik, 20240654
# =============================================================================
# streamlit run mlproject_group33_streamlit.py

# Import necessary libraries
import streamlit as st
import pandas as pd
import mlproject_dashboard_functions
import plotly.express as px
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---- Streamlit Page Config ----
st.set_page_config(page_title='ML Project | WCB - Group 33 | 2024/25',
                   page_icon='./static/Logo-Nova-IMS-White.png',
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

st.logo(image='static/640px-HD_transparent_picture.png', icon_image='static/Logo-Nova-IMS-Black.png')

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
        <img src='./app/static/WCB_Group33_Banner.png' alt="Banner Image">
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
                    <p class="subtitle-title">Title</p>
                </div>
                <br><br>
                <p style="text-align: justify; max-width: 800px; margin: auto;">
                    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean dictum justo sed fringilla blandit. 
                    In non purus at magna convallis malesuada. Sed ornare velit massa, eu ultricies augue auctor in. 
                    <br><br>
                    Nunc suscipit, arcu non ultrices cursus, ex mauris elementum arcu, sed tempus lectus purus tempor enim. 
                    Sed tincidunt enim lacus, sit amet scelerisque leo imperdiet et. Morbi vel congue ante, vitae commodo magna. 
                    Praesent commodo dolor vel mi fermentum condimentum. Curabitur rutrum massa quis metus mattis scelerisque. 
                    Nunc accumsan tempor est, sed finibus odio egestas vel. Fusce eu dui eget erat volutpat ultricies ut non quam.
                    <br><br>
                    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean dictum justo sed fringilla blandit
                </p>
                <br><br><br>
            </div>
        </div>
    </div>     
    <footer class="footer" style="visibility: visible; margin-top: 40px;">
        <div class="container">
            <div class="row">
                <div class="col-lg-12 text-center">
                    <p class="company-name" style="color: #d3d3d3;">ML Project | Group 33 © 2024/25</p>
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
# Load the data
# data = pd.read_csv('data/FinalData.csv')

# Source: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
# Load the model from disk
filename = './BestModel_11.11.2024.sav'
loaded_model = pickle.load(open(filename, 'rb'))


# =============================================================================
# --------------------------- Sidebar Filters ---------------------------
with st.sidebar:
    st.markdown("""aaaaaaaaaaaaaaaaaaaaaaaa""")



# =============================================================================
# --------------------------- Main Content ---------------------------
# Tabs

tab1, tab2 = st.tabs(["Model Prediction", "Data Analysis"])

with tab1:
    st.write("### Model Prediction")

    # Define input fields for the user
    st.write("#### Input the following details:")

    # Numeric inputs
    c3_date_binary = st.number_input("C-3 Date Binary", min_value=0, max_value=1, step=1)
    first_hearing_date_binary = st.number_input("First Hearing Date Binary", min_value=0, max_value=1, step=1)
    age_at_injury_clean = st.number_input("Age at Injury Clean", min_value=0)
    weekly_wage_reported = st.number_input("Weekly Wage Reported", min_value=0.0)
    ime4_reported = st.number_input("IME-4 Reported", min_value=0.0)
    c2_date_year = st.number_input("C-2 Date Year", min_value=1900, max_value=2100, step=1)

    # Binary input
    attorney_representative_y = st.selectbox("Attorney/Representative", options=[0, 1])

    # Injury cause selection
    wcio_cause_of_injury = st.selectbox("WCIO Cause of Injury", options=[
        '1 - Temp', '2 - Caught', '3 - Cut', '4 - Fall', '5 - Motor Vehicle',
        '6 - Strain_data', '7 - Striking', '8 - Struck', '9 - Rubbed', '10 - Miscellaneous'
    ])

    # Part of body selection
    wcio_part_of_body = st.selectbox("WCIO Part of Body", options=[
        'I - Head', 'II - Neck', 'III - Upper Extremities', 'IV - Trunk',
        'V - Lower Extremities', 'VI - Multiple Body Parts'
    ])

    # Map selections to binary variables
    wcio_cause_of_injury_dict = {
        '1 - Temp': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        '2 - Caught': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        '3 - Cut': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        '4 - Fall': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        '5 - Motor Vehicle': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        '6 - Strain_data': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        '7 - Striking': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        '8 - Struck': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        '9 - Rubbed': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        '10 - Miscellaneous': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }

    wcio_part_of_body_dict = {
        'I - Head': [1, 0, 0, 0, 0, 0],
        'II - Neck': [0, 1, 0, 0, 0, 0],
        'III - Upper Extremities': [0, 0, 1, 0, 0, 0],
        'IV - Trunk': [0, 0, 0, 1, 0, 0],
        'V - Lower Extremities': [0, 0, 0, 0, 1, 0],
        'VI - Multiple Body Parts': [0, 0, 0, 0, 0, 1]
    }

    # Prepare input data for prediction
    input_data = [
        c3_date_binary, first_hearing_date_binary, age_at_injury_clean,
        weekly_wage_reported, ime4_reported, c2_date_year, attorney_representative_y
    ] + wcio_cause_of_injury_dict[wcio_cause_of_injury] + wcio_part_of_body_dict[wcio_part_of_body]

    # Predict button
    if st.button("Predict"):
        prediction = loaded_model.predict([input_data])
        st.write(f"Prediction: {prediction[0]}")



with tab2:
    st.write("### Data Analysis")