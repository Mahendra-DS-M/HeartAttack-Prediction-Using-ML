######################################## Required Modules ###############################
# For Data Load & Study
import numpy as np
import pandas as pd

# For Model Files
import joblib

# For UI           
import streamlit as st # pip install streamlit
from streamlit_option_menu import option_menu # pip install streamlit_optione_menu

import time

######################## Reading Model & Its Supporting files ####################
ohe = joblib.load("Pickles/ohe.pkl")
model = joblib.load("Pickles/dt.pkl")

########################### Input Data Reference ############################
inpdata = pd.read_csv("HeartInput.csv")

######################## Helper functions for Inputs #####################
if 'sbutton' not in st.session_state:
    st.session_state['sbutton'] = False

if 'fbutton' not in st.session_state:
    st.session_state['fbutton'] = False

def switch_sbutton_state():
    st.session_state['sbutton'] = True
    st.session_state['fbutton'] = False

def switch_fbutton_state():
    st.session_state['sbutton'] = False
    st.session_state['fbutton'] = True

############################### UI & Logic ################################
# Streamlit Doc Help: https://docs.streamlit.io/develop/api-reference
st.markdown(
    """
    <style>
    /* Reduce space at the top */
    .block-container {
        padding-top: 2rem;   /* adjust this value */
        padding-bottom: 1rem;
    }
    [data-testid="stSidebar"] {
        width: 250px !important;  /* Adjust Sidebar Width */
        min-width: 250px !important;
    }
    /* Justify main content */
    .stMarkdown p {
        text-align: justify !important;
    }
    .Analyze {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    selected = option_menu("🩺Health", ["Home", "Project Details", "Input & Predict"], 
        icons=['house', 'info-square-fill', 'cloud-upload'], 
        menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "black"},
            "icon": {"color": "white", "font-size": "15px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "black"},
            "nav-link-selected": {"background-color": "green"},
        }
    )

if selected == 'Home':
    st.set_page_config(page_title="ML", layout="wide")
    st.subheader("🎯 Heart Attack Identification:")
    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
    st.write("For the taken persons data, an ML model trained for future prediction of Heart Disease..")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image('https://miro.medium.com/v2/1*hoXNj7fBYuNCaZIfF2h8vg.jpeg')

elif selected == 'Project Details':
    st.markdown("#### :blue[📋 Data:]")
    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)    
    st.write("Model Built on below X Data:")
    st.dataframe(inpdata.head())
    st.write("Classification models built on to study and predict heart attack for a person can be occured or not...")

elif selected == 'Input & Predict':
    # Userinputs -> DataFrame -> Data Pre-Processing Steps -> Prediction
    st.subheader(":green[📝 Enter Person's Health Data for Prediction:]")
    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
    # Prediction Buttons
    cola, colb = st.columns(2)
    with cola:
        sbutton = st.button(":yellow[Predict Single]", on_click=switch_sbutton_state, icon=':material/input:')
        if st.session_state['sbutton'] == True:
            st.write("Enter Details of a Person....")
            # User inputs
            x1, x2 = st.columns(2)
            with x1:
                gen = st.selectbox("Select Gender:", inpdata.Gender.unique())
            with x2:
                agecat = st.selectbox("Select Age Category:", inpdata.AgeCategory.unique())

            x3, x4 = st.columns(2)
            with x3:
                bmi = st.number_input("Enter BMI value:", min_value=inpdata.BMI.min(), max_value=inpdata.BMI.max())
            with x4:
                ph = st.number_input("Enter Physical Health value:", min_value=inpdata.PhysicalHealth.min(), max_value=inpdata.PhysicalHealth.max())

            x5, x6 = st.columns(2)
            with x5:
                mh = st.number_input("Enter Mental Health value:", min_value=inpdata.MentalHealth.min(), max_value=inpdata.MentalHealth.max())
            with x6:
                smoke = st.selectbox("Select Smoking Status:", inpdata.Smoking.unique())
                
            if st.button(":red[Predict]"):
                row = pd.DataFrame([[gen,agecat,bmi,ph,mh,smoke]], columns=inpdata.columns)
                st.write(":green[Given User Data:]")
                st.dataframe(row)

                # Pre-Processing
                row.Gender.replace({'Male':1, 'Female':0}, inplace=True)
                row.Smoking.replace({'Yes':1, 'No':0}, inplace=True)

                ohedata = ohe.transform(row[['AgeCategory']]).toarray()
                ohedata = pd.DataFrame(ohedata, columns=ohe.get_feature_names_out())
                row = pd.concat([row.drop("AgeCategory",axis=1), ohedata], axis=1)

                # Predictions
                probs = model.predict_proba(row)[0] # will predict 12 classes probabilities
                classes = {
                            0: "No 😊",
                            1: "Yes 🤒👨🏻‍⚕️",
                        }
                
                # Pick Top
                indices = probs.argsort()[::-1]
                top_preds = [(classes[i], probs[i]) for i in indices]

                # Create dataframe
                df = pd.DataFrame(top_preds, columns=["Prediction", "Probability"])
                df["Probability"] = df["Probability"].apply(lambda x: f"{x*100:.2f}%")
                st.write(f"##### :blue[Heart Attack Possibility:]")
                st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
                st.table(df)
                st.balloons()

    with colb:
        fbutton = st.button(":yellow[Predict Multiple]", on_click=switch_fbutton_state, icon=':material/table:')
        if st.session_state['fbutton'] == True:
            file = st.file_uploader("Upload Data File Containing Multiple Persons Details:", type=['csv','xlsx'])
            st.write("Ref to Data given in Project Particulars..")
            if file!=None:
                try:
                    df = pd.read_csv(file)
                except:
                    df = pd.read_excel(file)

                st.write(":green[Uploaded Data....]")
                st.dataframe(df.head())

                if st.button(":red[Predict]"):

                    # Taking Copy of Uploaded Data
                    data = df.copy()

                    ############### Using above saved encoded files transforming text columns to numeric ###################

                    # Ordinal Encoding
                    data.Gender.replace({'Male':1, 'Female':0}, inplace=True)
                    data.Smoking.replace({'Yes':1, 'No':0}, inplace=True)

                    # One-Hot Encoding
                    data_ohe = ohe.transform(data[['AgeCategory']]).toarray()
                    data_ohe = pd.DataFrame(data_ohe, columns=ohe.get_feature_names_out())

                    data = pd.concat([data.drop('AgeCategory', axis=1), data_ohe], axis=1)
                    
                    with st.spinner('Predicting...'):
                        # Predictions
                        ypred = model.predict(data)
                        time.sleep(2)
                        st.success(":green[Done!]")

                        # Taking output column & adding predictions
                        df['Heart Disease'] = ypred

                        st.write(":blue[Predictions....]")
                        st.dataframe(df)
                        st.write("Total Inputs given:", len(df))
                        st.write("Predicted Yes:",len(df[df['Heart Disease']=='Yes']))
                        st.write("Predicted No:",len(df[df['Heart Disease']=='No']))
                        st.balloons()
                        
                        csv = df.to_csv(index=False)
                        st.download_button(label="Download Above Predictions as CSV",data=csv,file_name="predictions.csv",mime="text/csv")
