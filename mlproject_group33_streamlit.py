# =============================================================================
# Machile Learning Project | Group 33 | 2024/25 | MSc in Data Science and Advanced Analytics
#        André Silvestre, 20240502 | João Henriques, 20240499 | Simone Genovese, 20241459
#        Steven Carlson, 20240554 | Vinícius Pinto, 20211682 | Zofia Wojcik, 20240654
# =============================================================================
# streamlit run mlproject_group33_streamlit.py

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import mlproject_dashboard_functions
import plotly.express as px
import pickle
import gzip

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

                    button[kind="secondary"]:focus {{
                        font-weight: bold;
                        color: #084594;
                        border: 2px solid #084594;
                    }}

                    button[kind="secondary"]:active {{
                        font-weight: bold;
                        color: #fff;
                        background-color: #084594;
                        border: 2px solid #084594;
                    }}  
                
                    button[role="secondary"]:target {{
                        font-weight: bold;
                        color: #084594;
                        border: 2px solid #084594;
                    }}

                    button[role="secondary"]:selected {{
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

# Palette of colors | Blues
palette = ['#002147', '#084594', '#135C9B', '#2171B5', '#4292C6', '#6BAED6', '#9ECAE1', '#C6DBEF', '#DEEBF7', '#F7FBFF']
# Color based on prediction
claim_injury_type_dict_swapped = {1: "1. CANCELLED", 2: "2. NON-COMP", 3: "3. MED ONLY", 4: "4. TEMPORARY",
                                    5: "5. PPD SCH LOSS", 6: "6. PPD NSL", 7: "7. PTD", 8: "8. DEATH"}
claim_injury_type_palette = dict(zip(sorted(claim_injury_type_dict_swapped.keys()), palette))
claim_injury_type_palette_original = dict(zip(sorted(claim_injury_type_dict_swapped.values()), palette))
        
# =============================================================================
# --------------------------- Load Data ---------------------------
# Load the data
@st.cache_data
def load_data():
    train_data = pd.read_parquet("./train_data_cleaned.parquet")
    test_data = pd.read_parquet("./test_data_cleaned.parquet")
    return train_data, test_data

train_data, test_data = load_data()

# Source: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
# Load the model
@st.cache_resource
def init_model():
    with gzip.open('BestModel_Compressed.sav.gz', 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

best_model = init_model()

# =============================================================================
# --------------------------- Sidebar ---------------------------
# Sidebar Title
with st.sidebar:
    
    # Sidebar Image
    st.image('./static/Logo-Nova-IMS-Black.png', width=100)
    
    # Sidebar Title
    st.markdown("<h1 style='color: #084594; font-weight: bold;'>WCB - Group 33</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #4984d2; font-weight: bold;'>2024/25</h2>", unsafe_allow_html=True)
    
    # Disclamer and Instructions to the User 
    st.markdown("""
        <style>
            .sidebar-text { 
                font-size: 14px;
                color: #3d392d;
                font-weight: 600;
                text-align: justify;
            }
        </style>
        
        <div class="sidebar-text">
            <p><b>Disclaimer:</b> This is a Machine Learning project developed by Group 33 for the Workers' Compensation Board (WCB).</p>
            <center><p style="font-size: 14px;"><i>The results presented here are based on the data provided by the WCB and the model developed by the group. <br>
                This result can be used as a reference, but it is not a substitute for professional advice. </i></p> </center> <br><br>
            <p><b>Instructions:</b> Use the tabs below to make a prediction or analyze the data. <br>
                If you have any questions or need to know how to fill in the fields, please read the 
                <a href="https://data.ny.gov/Government-Finance/Assembled-Workers-Compensation-Claims-Beginning-20/jshw-gkgu/about_data">Q&A</a> 
                <br><br>
                To find the correct bucket for the WCIO Cause of Injury, Nature of Injury, and Part of Body, please refer to the link below:
                <a href="https://www.guarantysupport.com/wp-content/uploads/2024/02/WCIO-Legacy.pdf">WCIO Codes</a> 
            </p>
            <br> <br>
            <center><h3 style="color: #084594;">Group 33 Members</h3></center>
            <ul>
                <li>André Silvestre, 20240502</li>
                <li>João Henriques, 20240499</li>
                <li>Simone Genovese, 20241459</li>
                <li>Steven Carlson, 20240554</li>
                <li>Vinícius Pinto, 20211682</li>
                <li>Zofia Wojcik, 20240654</li>
            </ul>
            <br>
            <center><p style="font-size: 14px;"><i>Thank you for using our application!</i></p> </center>
        </div>
    """, unsafe_allow_html=True)

# --------------------------- Main Content ---------------------------
# Tabs

tab1, tab2 = st.tabs(["Model Prediction", "Data Analysis"])

with tab1:
    st.write("### Model Prediction")

    # List of Features 
    features = ['Age at Injury Clean', 'C-2 Date Year', 'C-3 Date Binary', 'First Hearing Date Binary',
                'Attorney/Representative_Y', 'Weekly Wage Reported', 'IME-4 Reported',
                'WCIO Cause of Injury Bucket_3 - Cut', 'WCIO Cause of Injury Bucket_6 - Strain_data',
                'WCIO Nature of Injury Bucket_1 - Specific', 'WCIO Nature of Injury Bucket_2 - Occupational/Cumulative',
                'WCIO Nature of Injury Bucket_3 - Multiple', 'WCIO Part of Body Bucket_III - Upper Extremities',
                'WCIO Part of Body Bucket_IV - Trunk', 'WCIO Part of Body Bucket_V - Lower Extremities']
    
    target = 'Claim Injury Type'
    
    # Define input fields for the user
    st.write("Input the following details to predict the **Claim Injury Type**:")

    # Numeric inputs
    age_at_injury_clean = st.number_input("Age at Injury", min_value=0, max_value=150, step=1, help="Enter the age at the time of injury")
    assembly_date_year = st.number_input("Assembly Date Year", min_value=1900, max_value=2100, step=1, help="Enter the year of the Assembly date")
    c2_date_year = st.number_input("C-2 Date Year", min_value=1900, max_value=2100, step=1, help="Enter the year of the C-2 date")

    # Binary input with "Yes" or "No" options
    accident_date_binary = st.selectbox(label="Accident Date Binary", options=["No", "Yes"], help="Was the accident date reported?",
                                        placeholder="Select an option", label_visibility="visible")

    alternative_dispute_resolution_y = st.selectbox("Alternative Dispute Resolution", options=["No", "Yes"], help="Was an alternative dispute resolution present?",
                                                     placeholder="Select an option", label_visibility="visible")
    attorney_representative_y = st.selectbox("Attorney/Representative", options=["No", "Yes"], help="Was an attorney or representative present?",
                                             placeholder="Select an option", label_visibility="visible")
    
    c2_date_binary = st.selectbox(label="C-2 Date Binary", options=["No", "Yes"], help="Was the C-2 date reported?", 
                                  placeholder="Select an option", label_visibility="visible")
    c3_date_binary = st.selectbox(label="C-3 Date Binary", options=["No", "Yes"],  help="Was the C-3 date reported?", 
                                  placeholder="Select an option", label_visibility="visible")
    
    covid_19_indicator_y = st.selectbox("COVID-19 Indicator", options=["No", "Yes"], help="Was the COVID-19 indicator present?",
                                        placeholder="Select an option", label_visibility="visible")
    district_name_nyc = st.selectbox("District Name NYC", options=["No", "Yes"], help="Was the district name NYC?",
                                      placeholder="Select an option", label_visibility="visible")
    
    first_hearing_date_binary = st.selectbox("First Hearing Date Binary", options=["No", "Yes"], help="Was the first hearing date reported?",
                                             placeholder="Select an option", label_visibility="visible")
    
    gender_m = st.selectbox("Gender M", options=["No", "Yes"], help="Was your gender Male?",
                            placeholder="Select an option", label_visibility="visible")
    weekly_wage_reported = st.selectbox("Weekly Wage Reported", options=["No", "Yes"], help="Was the Average Weekly Wage reported?",
                                        placeholder="Select an option", label_visibility="visible")
    ime4_reported = st.selectbox("IME-4 Reported", options=["No", "Yes"])

    # Convert "Yes" or "No" to binary values
    accident_date_binary = 1 if accident_date_binary == "Yes" else 0
    alternative_dispute_resolution_y = 1 if alternative_dispute_resolution_y == "Yes" else 0
    attorney_representative_y = 1 if attorney_representative_y == "Yes" else 0
    c2_date_binary = 1 if c2_date_binary == "Yes" else 0
    c3_date_binary = 1 if c3_date_binary == "Yes" else 0
    covid_19_indicator_y = 1 if covid_19_indicator_y == "Yes" else 0
    district_name_nyc = 1 if district_name_nyc == "Yes" else 0
    first_hearing_date_binary = 1 if first_hearing_date_binary == "Yes" else 0
    gender_m = 1 if gender_m == "Yes" else 0
    weekly_wage_reported = 1 if weekly_wage_reported == "Yes" else 0
    ime4_reported = 1 if ime4_reported == "Yes" else 0
    
    # Carrier Type Bucket Selection
    carrier_type_bucket = st.selectbox("Carrier Type Bucket", 
                                       options=['1A. PRIVATE', 
                                                '2A. SIF', '3A. SELF PUBLIC', 
                                                '4A. SELF PRIVATE', 
                                                '5A-5C. SPECIAL FUND'], 
                                       help="Select the Carrier Type Bucket")
    
    # Industry Code Description Selection
    industry_code = st.selectbox("Industry Code", 
                                 options=['Not Applicable',
                                          'ACCOMMODATION AND FOOD SERVICES',
                                          'ADMINISTRATIVE AND SUPPORT AND WASTE MANAGEMENT AND REMEDIAT',
                                          'AGRICULTURE, FORESTRY, FISHING AND HUNTING',
                                          'ARTS, ENTERTAINMENT, AND RECREATION',
                                          'CONSTRUCTION',
                                          'EDUCATIONAL SERVICES',
                                          'FINANCE AND INSURANCE',
                                          'HEALTH CARE AND SOCIAL ASSISTANCE',
                                          'INFORMATION',
                                          'MANAGEMENT OF COMPANIES AND ENTERPRISES',
                                          'MANUFACTURING',
                                          'MINING',
                                          'OTHER SERVICES (EXCEPT PUBLIC ADMINISTRATION)',
                                          'PROFESSIONAL, SCIENTIFIC, AND TECHNICAL SERVICES',
                                          'PUBLIC ADMINISTRATION',
                                          'REAL ESTATE AND RENTAL AND LEASING',
                                          'RETAIL TRADE',
                                          'TRANSPORTATION AND WAREHOUSING',
                                          'UTILITIES',
                                          'WHOLESALE TRADE'], 
                                         help="Select the Industry Code")
    

    # WCIO Cause of Injury Selection [Buckets]
    wcio_cause_of_injury = st.selectbox("WCIO Cause of Injury", 
                                        options=['0 - Unknown',
                                                 '1 - Temp',
                                                 '2 - Caught',
                                                 '3 - Cut',
                                                 '4 - Fall',
                                                 '5 - Motor Vehicle',
                                                 '6 - Strain_data',
                                                 '7 - Striking',
                                                 '8 - Struck',
                                                 '9 - Rubbed',
                                                 '10 - Miscellaneous'])

    # WCIO Part of Body Selection [Buckets]
    wcio_part_of_body = st.selectbox("WCIO Part of Body", options=[
        '0 - Unknown', 'I - Head', 'II - Neck', 'III - Upper Extremities',
        'IV - Trunk', 'V - Lower Extremities', 'VI - Multiple Body Parts'])

    # WCIO Nature of Injury Selection [Buckets]
    wcio_nature_of_injury = st.selectbox("WCIO Nature of Injury", options=[
        '0 - Unknown', '1 - Specific', '2 - Occupational/Cumulative', '3 - Multiple'])
    
    # List of features to be used in the prediction
    # 'Age at Injury Clean',
    # 'Assembly Date Year',
    # 'C-2 Date Year',
    # 'Accident Date Binary',
    # 'Alternative Dispute Resolution_Y',
    # 'Attorney/Representative_Y',
    # 'C-2 Date Binary',
    # 'C-3 Date Binary',
    # 'COVID-19 Indicator_Y',
    # 'Carrier Type Bucket_1A. PRIVATE',
    # 'Carrier Type Bucket_2A. SIF',
    # 'Carrier Type Bucket_3A. SELF PUBLIC',
    # 'Carrier Type Bucket_4A. SELF PRIVATE',
    # 'Carrier Type Bucket_5A-5C. SPECIAL FUND',
    # 'District Name_NYC',
    # 'First Hearing Date Binary',
    # 'Gender_M',
    # 'IME-4 Reported',
    # 'Industry Code Description_CONSTRUCTION',
    # 'Industry Code Description_PUBLIC ADMINISTRATION',
    # 'Industry Code Description_Unknown',
    # 'WCIO Cause of Injury Bucket_3 - Cut',
    # 'WCIO Cause of Injury Bucket_6 - Strain_data',
    # 'WCIO Nature of Injury Bucket_1 - Specific',
    # 'WCIO Nature of Injury Bucket_2 - Occupational/Cumulative',
    # 'WCIO Nature of Injury Bucket_3 - Multiple',
    # 'WCIO Part of Body Bucket_III - Upper Extremities',
    # 'WCIO Part of Body Bucket_IV - Trunk',
    # 'WCIO Part of Body Bucket_V - Lower Extremities',
    # 'Weekly Wage Reported'

    # Prepare input data for prediction
    input_data_dict = {
        'Age at Injury Clean': age_at_injury_clean,
        'Assembly Date Year': assembly_date_year,
        'C-2 Date Year': c2_date_year,
        'Accident Date Binary': accident_date_binary,
        'Alternative Dispute Resolution_Y': alternative_dispute_resolution_y,
        'Attorney/Representative_Y': attorney_representative_y,
        'C-2 Date Binary': c2_date_binary,
        'C-3 Date Binary': c3_date_binary,
        'COVID-19 Indicator_Y': covid_19_indicator_y,
        'Carrier Type Bucket_1A. PRIVATE': 1 if carrier_type_bucket == '1A. PRIVATE' else 0,
        'Carrier Type Bucket_2A. SIF': 1 if carrier_type_bucket == '2A. SIF' else 0,
        'Carrier Type Bucket_3A. SELF PUBLIC': 1 if carrier_type_bucket == '3A. SELF PUBLIC' else 0,
        'Carrier Type Bucket_4A. SELF PRIVATE': 1 if carrier_type_bucket == '4A. SELF PRIVATE' else 0,
        'Carrier Type Bucket_5A-5C. SPECIAL FUND': 1 if carrier_type_bucket == '5A-5C. SPECIAL FUND' else 0,
        'District Name_NYC': district_name_nyc,
        'First Hearing Date Binary': first_hearing_date_binary,
        'Gender_M': gender_m,
        'IME-4 Reported': ime4_reported,
        'Industry Code Description_CONSTRUCTION': 1 if industry_code == "CONSTRUCTION" else 0,
        'Industry Code Description_PUBLIC ADMINISTRATION': 1 if industry_code == "PUBLIC ADMINISTRATION" else 0,
        'Industry Code Description_Unknown': 1 if industry_code == "Unknown" else 0,
        'WCIO Cause of Injury Bucket_3 - Cut': 1 if wcio_cause_of_injury == "3 - Cut" else 0,
        'WCIO Cause of Injury Bucket_6 - Strain_data': 1 if wcio_cause_of_injury == "6 - Strain_data" else 0,
        'WCIO Nature of Injury Bucket_1 - Specific': 1 if wcio_nature_of_injury == "1 - Specific" else 0,
        'WCIO Nature of Injury Bucket_2 - Occupational/Cumulative': 1 if wcio_nature_of_injury == "2 - Occupational/Cumulative" else 0,
        'WCIO Nature of Injury Bucket_3 - Multiple': 1 if wcio_nature_of_injury == "3 - Multiple" else 0,
        'WCIO Part of Body Bucket_III - Upper Extremities': 1 if wcio_part_of_body == "III - Upper Extremities" else 0,
        'WCIO Part of Body Bucket_IV - Trunk': 1 if wcio_part_of_body == "IV - Trunk" else 0,
        'WCIO Part of Body Bucket_V - Lower Extremities': 1 if wcio_part_of_body == "V - Lower Extremities" else 0,
        'Weekly Wage Reported': weekly_wage_reported
    }
    
    # Predict button
    if st.button("Predict"):
        
        # Prediction
        prediction = best_model.predict([list(input_data_dict.values())])
        prediction_text = claim_injury_type_dict_swapped[prediction[0]]
        
        # Prediction Score
        probabilities = best_model.predict_proba([list(input_data_dict.values())])
        predicted_class_index = list(claim_injury_type_dict_swapped.keys()).index(prediction[0])
        prediction_score = probabilities[0][predicted_class_index]
        
        # Prediction Color
        prediction_color = claim_injury_type_palette[prediction[0]]  # Get the color for the predicted class

        # Display the prediction result
        st.markdown(f"""
            <!-- Bootstrap CSS -->
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.4.1/dist/css/bootstrap.min.css"
                integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh" crossorigin="anonymous">                
            <h1 style='text-align: center; margin-top: 50px;'>Prediction Result</h1>
            <br>
            <div class="container">
                <div class="row">
                    <div class="card text-center" style="border-color: {prediction_color};width: 50rem; margin: auto;">
                        <div class="card-header" style="background-color: {prediction_color}; color: #FFF;">
                            <h1 style="margin: 0;color: #FFF;">{prediction_text}</h1>
                        </div>
                        <div class="card-body">
                            <p class="card-text" style="color: {prediction_color};">
                                <b>Score:</b> {round(float(prediction_score), 4)}
                            </p>
                        </div>
                    </div>
                </div> 
            </div>
        """, unsafe_allow_html=True)

with tab2:
    st.write("### Data Analysis")
    
    # Filters for the data analysis
    st.write("#### Filter the data:")
    
    # Filter by Claim Injury Type | Assembly Date Year | Average Weekly Wage Reported [3 columns]
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filter by Claim Injury Type
        claim_injury_type = st.multiselect("Claim Injury Type", sorted(train_data["Claim Injury Type"].unique()), 
                                           help="Select the Claim Injury Type")
    
    with col2:
        # Filter by Assembly Date Year
        assembly_date_year = st.multiselect("Assembly Date Year", sorted(train_data["Assembly Date Year"].unique()),
                                             help="Select the Assembly Date Year")
    
    with col3:
        # Filter by Average Weekly Wage Reported
        weekly_wage_reported = st.multiselect("Weekly Wage Reported", 
                                              options=["Yes", "No"],
                                              help="Select the Average Weekly Wage Reported")
        weekly_wage_reported = [1 if x == "Yes" else 0 for x in weekly_wage_reported]
    
    
    # Apply filters to the data
    filtered_data = train_data.copy()
    
    if claim_injury_type:
        filtered_data = filtered_data[filtered_data["Claim Injury Type"].isin(claim_injury_type)]
        
    if assembly_date_year:
        filtered_data = filtered_data[filtered_data["Assembly Date Year"].isin(assembly_date_year)]
        
    if weekly_wage_reported:
        filtered_data = filtered_data[filtered_data["Weekly Wage Reported"].isin(weekly_wage_reported)]
        
    # # Display the filtered data
    # st.write("#### Display the filtered data:")
    # st.write(filtered_data)
    
    
    # --------------------------- Data Analysis ---------------------------
    # Card with the number Assembled Claims, Average Weekly Wage, and Average Age at Injury
    card_col1, card_col2, card_col3 = st.columns(3)
    
    # Card 1 | Number of Assembled Claims
    mlproject_dashboard_functions.create_card(col=card_col1, 
                                              icon_name='fas fa-briefcase-medical', 
                                              color=[171, 186, 223], 
                                              color_text=[255, 255, 255],
                                              title='Assembled Claims', 
                                              value=filtered_data.shape[0])
    
    # Card 2 | Average Weekly Wage
    mlproject_dashboard_functions.create_card(card_col2, 'fas fa-dollar-sign', [15, 66, 142], [255, 255, 255], 
                                              'Average Weekly Wage', round(filtered_data["Weekly Wage Reported"].mean(), 2))                                              
    # Card 3 | Average Age at Injury
    mlproject_dashboard_functions.create_card(card_col3, 'fas fa-user', [0, 33, 64], [255, 255, 255], 
                                              'Average Age at Injury', 
                                              round(filtered_data["Age at Injury Clean"].mean()))
    
    # =================================================================================================================
    # 2 Columns for the plots 
    col1, col2 = st.columns([0.3, 0.7])
        
    # Plot 1 | Distribution of the Claim Injury Type
    fig_claims = px.histogram(filtered_data, 
                              x="Claim Injury Type", 
                              title="Distribution of the Claim Injury Type",
                              color = "Claim Injury Type",
                              color_discrete_map = claim_injury_type_palette_original,
                              labels={"Claim Injury Type": "Claim Injury Type", "count": "Number of Claims"})
    # Update the layout
    fig_claims.update_layout(
        font=dict(family="Arial", size=12, color="black"),
        showlegend=False)  # Hide the legend
    
    # Update the traces
    fig_claims.update_traces(hovertemplate='<b>Claim Injury Type</b>: %{x}<br><b>Number of Claims</b>: %{y}')
    col1.plotly_chart(fig_claims)
    
    # Plot 2 | Time Series of the Accident Date, Assembly Date, C-2 Date, C-3 Date and First Hearing Date

    # Transform the data into long format
    date_columns = ["Accident Date", "Assembly Date", "C-2 Date", "C-3 Date", "First Hearing Date"]
    long_data = filtered_data.melt(
        value_vars=date_columns,
        var_name="Date Type",       # New column for the type of date
        value_name="Date"           # New column for the actual dates
    )

    # Set the index to the date
    long_data["Date"] = pd.to_datetime(long_data["Date"])
    long_data = long_data.set_index("Date")

    # Resample the data to monthly frequency and count the occurrences
    monthly_data = long_data.groupby([pd.Grouper(freq='M'), 'Date Type']).size().reset_index(name='Count')

    # Create the time series plot
    fig_dates = px.line(
        monthly_data,
        x="Date",                                           # X-axis with the dates
        y="Count",                                          # Y-axis with the counts
        color="Date Type",                                  # Differentiate by the type of date
        color_discrete_map={"Accident Date": "#084594", "Assembly Date": "#4292C6", "C-2 Date": "#6BAED6", 
                            "C-3 Date": "#9ECAE1", "First Hearing Date": "#C6DBEF"},  # Colors
        title="Monthly Time Series of the Accident Date, Assembly Date, C-2 Date, C-3 Date and First Hearing Date",
        labels={"Date": "Date", "Count": "Number of Cases", "Date Type": "Date Type"}
    )

    # Customize the layout
    fig_dates.update_layout(
        font=dict(family="Arial", size=12, color="black"),
        showlegend=True,                                    # Show the legend
        legend_title_text="Date Type",                      # Legend title
        xaxis_title="Date",                                 # X-axis title
        yaxis_title="Number of Events",                     # Y-axis title
        template="plotly_white"                             # Clean theme
    )

    # Improve hover information
    fig_dates.update_traces(
        hovertemplate='<b>%{x|%Y-%m}</b><br>Count: %{y}<br>Event Type: %{legendgroup}'
    )

    # Display in Streamlit
    col2.plotly_chart(fig_dates)
    
    # =================================================================================================================
 
    # Create a new dataframe grouped by
    grouped_data = filtered_data.groupby(["Medical Fee Region", "District Name", "County of Injury"]).size().reset_index(name="Number of Claims")
        
    # Treemap Plot | Medical Fee Region, District Name & County of Injury
    fig_treemap = px.treemap(grouped_data, 
                             path=['Medical Fee Region', 'District Name', 'County of Injury'], 
                             title="Medical Fee Region, District Name & County of Injury",
                             values='Number of Claims',
                             color='Number of Claims',
                             color_continuous_scale='Greys',
                             height=1200)
    
    # Update the layout
    fig_treemap.update_layout(
        font=dict(family="Arial", size=12, color="black"),
        showlegend=False)  # Hide the legend
    
    st.plotly_chart(fig_treemap, use_container_width=True)
    
    # =================================================================================================================
    
    # Sunburst Plot | Industry Code Description, Carrier Type Bucket & Gender
    
    # Count number of claims by Industry Code Description, Carrier Type Bucket and Gender
    grouped_data = filtered_data.groupby(["Industry Code Description", "Carrier Type Bucket", "Gender"]).size().reset_index(name="Number of Claims")
    
    fig_sunburst = px.sunburst(grouped_data, 
                               path=['Industry Code Description', 'Carrier Type Bucket', 'Gender'],
                               title="Industry Code Description, Carrier Type Bucket & Gender",
                               values='Number of Claims',
                               color='Number of Claims',
                               color_continuous_scale='Blues',
                               height=1200)
    
    # Update the layout
    fig_sunburst.update_layout(
        font=dict(family="Arial", size=12, color="black"),
        showlegend=False)  # Hide the legend
    
    # Update the traces
    fig_sunburst.update_traces(textinfo='label+percent parent',
                                hoverinfo='label+percent parent',
                                hovertemplate='<b>%{label}</b><br>Number of Claims: %{value}<br>Percentage: %{percentParent:.2%}',
                                branchvalues='total')
    
    st.plotly_chart(fig_sunburst, use_container_width=True)
    