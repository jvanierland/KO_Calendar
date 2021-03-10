import os
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

st.sidebar.title('App Mode')
app_mode=st.sidebar.selectbox(
"Input Forms",
("Kick Off Compressor Form", "Well Protection Trailer Form")
)

if app_mode=="Kick Off Compressor Form":
    st.write("""
    # Kick Off Compressor Request Form
    Demo Data Web app, project idea by ***Brandt Tucker***
    """)

    df=pd.read_csv(r"C:/Calendar_Files_Brandt/KO_Compressor_Data.csv")
    df=df.sort_values('Start_Date')
    df

    st.write("""
    ## Foreman/ Engineer
    """)
    foreman_name=st.text_input('Enter Name')

    st.write("""
    ## Location
    """)
    location=st.text_input('Enter Location')

    st.write("""
    ## Start Date
    """)
    start_date=st.date_input('Enter Start Date')

    st.write("""
    ## End Date
    """)
    end_date=st.date_input('Enter End Date')

    st.write("""
    ## Expected Maximum Discharge Pressure (psi)
    """)
    max_discharge_psi=st.number_input('Enter Max Discharge Pressure', min_value=0, step=50, value=1000)

    st.write("""
    ## Additional Comments
    """)
    comments=st.text_input('Enter Comments', 'None')
    

    dd1 = {'Foreman/Engineer': [foreman_name],
        'Location': [location],
        'Start_Date': [start_date],
        'End_Date': [end_date],
        'Max_Discharge_Psi': [max_discharge_psi],
        'Comments': [comments]
        }

    df_new=pd.DataFrame(dd1)

    if st.button('Form Submission'):
        df_result=pd.concat([df,df_new], ignore_index=True)
        df_result.to_csv(r"C:/Calendar_Files_Brandt/KO_Compressor_Data.csv", index=False)
        df_result

elif app_mode=="Well Protection Trailer Form":
    st.write("""
    # Well Protection Trailer Form
    Demo Data Web app, project idea by ***Brandt Tucker***
    """)

    df2=pd.read_csv(r"C:/Calendar_Files_Brandt/Well_Protection_Trailer_Data.csv")
    df2=df2.sort_values('Start_Date')
    df2

    st.write("""
    ## Foreman/ Engineer
    """)
    foreman_name=st.text_input('Enter Name')

    st.write("""
    ## Location
    """)
    location=st.text_input('Enter Location')

    st.write("""
    ## Start Date
    """)
    start_date=st.date_input('Enter Start Date')

    st.write("""
    ## End Date
    """)
    end_date=st.date_input('Enter End Date')

    st.write("""
    ## Number of wells that will be pumped
    """)
    max_discharge_psi=st.number_input('Enter # of Wells', min_value=1, step=1, value=1)

    st.write("""
    ## Additional Comments
    """)
    comments=st.text_input('Enter Comments', 'None')
    

    dd2 = {'Foreman/Engineer': [foreman_name],
        'Location': [location],
        'Start_Date': [start_date],
        'End_Date': [end_date],
        'Max_Discharge_Psi': [max_discharge_psi],
        'Comments': [comments]
        }

    df_new2=pd.DataFrame(dd2)

    if st.button('Form Submission'):
        df_result2=pd.concat([df2,df_new2], ignore_index=True)
        df_result2.to_csv(r"C:/Calendar_Files_Brandt/Well_Protection_Trailer_Data.csv", index=False)
        df_result2


