<p align="center">
   <a href="https://mlproject-wcb-group33.streamlit.app/">
        <img src="https://github.com/Silvestre17/ML_WebApp_Group33/blob/main/static/WCB_Group33_Banner.png" alt="WCB Group33 WebApp Banner" width="800">
    </a>
</p>

# 📊 ML Project - WCB Claim Severity Dashboard/Web App 🚀

This repository contains the code for the interactive Streamlit web application developed as part of the **Machine Learning** project for the **Master's in Data Science and Advanced Analytics** at **NOVA IMS**.

> The web application serves as the deployment phase (CRISP-DM) of the project, providing an interface to interact with the developed classification model for predicting the severity of New York Workers' Compensation Board (NWCB) claims and exploring the underlying data.

<p align="center">
    <a href="https://mlproject-wcb-group33.streamlit.app/">
        <img src="https://img.shields.io/badge/Live_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Streamlit App">
    </a>
     <a href="https://github.com/Silvestre17/ML_WebApp_Group33">
        <img src="https://img.shields.io/badge/GitHub_Repo-100000?style=for-the-badge&logo=github&logoColor=white" alt="WebApp Repo">
    </a>
</p>
<p align="center">
    <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" /></a>
    <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" /></a>
    <a href="https://plotly.com/python/"><img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly" /></a>
    <a href="https://catboost.ai/"><img src="https://img.shields.io/badge/CatBoost-00AEEF?style=for-the-badge&logo=yandex&logoColor=white" alt="CatBoost" /></a>
    <a href="https://scikit-learn.org/stable/"><img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" /></a>
     <a href="https://github.com/marcotcr/lime"><img src="https://img.shields.io/badge/LIME-4CAF50?style=for-the-badge&logo=python&logoColor=white" alt="LIME" /></a>
</p>

## 🔗 Relation to Main Project

This web application represents the deployment and visualization component of our comprehensive **Machine Learning project** focused on predicting WCB claim severity. The main project repository contains all the data preprocessing, feature engineering, model training, evaluation notebooks, and detailed analysis reports.

➡️ **Main Project Repository:** [**Silvestre17/ML_24.25_Project_Group33**](https://github.com/Silvestre17/ML_24.25_Project_Group33) ⬅️

<br>

## ✨ WebApp Overview

This interactive dashboard provides two main functionalities:

1.  **🤖 Model Prediction:** Allows users to input claim details and receive a prediction for the `Claim Injury Type` based on the best-performing model developed in the main project (CatBoost). It also integrates LIME for explaining individual predictions.
2.  **🔍 Data Exploration:** Enables interactive exploration of the cleaned dataset used for model training. Users can visualize distributions, trends, and relationships between different claim features.

## 👥 Team (Group 33)

*   André Silvestre, 20240502
*   João Henriques, 20240499
*   Simone Genovese, 20241459
*   Steven Carlson, 20240554
*   Vinícius Pinto, 20211682
*   Zofia Wojcik, 20240654

## ⚙️ Dashboard Setup Locally

To run this Streamlit dashboard on your local machine, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Silvestre17/ML_WebApp_Group33.git
    cd ML_WebApp_Group33
    ```

2.  **Install the required libraries:**
    *(It's recommended to use a virtual environment)*
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit application:**
    ```bash
    streamlit run mlproject_group33_streamlit.py
    ```

4.  **Access the dashboard:** Open your web browser and navigate to the local URL provided in the terminal (usually `http://localhost:8501`).

5.  Enjoy interacting with the model and data! 🚀🔎
