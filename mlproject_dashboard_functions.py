import pandas as pd
import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# Function for sidebar launch selection dates
# Source: https://github.com/korenkaplan/Admin-dashboard/blob/main/sidebar.py
def init_sidebar_dates_pickers(data_frame_datatime):
    # Convert the order_date column to datetime
    data_frame_datatime = pd.to_datetime(data_frame_datatime).dt.date
    # Convert the order_date column to datetime for manipulation and find the min and max value
    min_date = data_frame_datatime.min()
    max_date = data_frame_datatime.max()
    # Initialize the sidebar date pickers and define the min and max value to choose from
    start_date = st.sidebar.date_input('Data de Início', min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.sidebar.date_input('Data de Fim', min_value=min_date, max_value=max_date, value=max_date)
    # Return the values
    return start_date, end_date


# Function for boot selection dates on page
def init_page_dates_pickers(data_frame_datatime, col1=None, col2=None):
    # Convert the order_date column to datetime
    data_frame_datatime = pd.to_datetime(data_frame_datatime).dt.date
    # Convert the order_date column to datetime for manipulation and find the min and max value
    min_date = data_frame_datatime.min()
    max_date = data_frame_datatime.max()
    # Initialize the sidebar date pickers and define the min and max value to choose from
    if col1 is not None and col2 is not None:
        start_date = col1.date_input('Data de Início', min_value=min_date, max_value=max_date, value=min_date)
        end_date = col2.date_input('Data de Fim', min_value=min_date, max_value=max_date, value=max_date)
    else:
        start_date = st.date_input('Data de Início', min_value=min_date, max_value=max_date, value=min_date)
        end_date = st.date_input('Data de Fim', min_value=min_date, max_value=max_date, value=max_date)
    # Return the values
    return start_date, end_date


# Function to create a custom HTML card
def create_card(col, icon_name, color, color_text, title, value):
    htmlstr = f"""
    <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.12.1/css/all.css" crossorigin="anonymous">
    <p style="background-color: rgb({color[0]}, {color[1]}, {color[2]});
            color: rgb({color_text[0]}, {color_text[1]}, {color_text[2]});
            font-weight: 700;
            font-size: 30px;
            border-radius: 7px;
            padding-left: 20px; 
            padding-top: 18px; 
            padding-bottom: 18px;
            line-height: 25px;">
        <i class='{icon_name} fa-xs' style='margin-right: 5px;'></i>{value}</style>
        <br>
        <span style='font-size: 18px; margin-top: 0; font-weight: 100;margin-left: 30px;'>{title}</style></span></p>
    """
    col.markdown(htmlstr, unsafe_allow_html=True)
