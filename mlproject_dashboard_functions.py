import pandas as pd
import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


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
