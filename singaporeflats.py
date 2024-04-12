import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
import joblib
from streamlit_option_menu import option_menu

# Define unique values for select boxes
flat_model_options = ['IMPROVED', 'NEW GENERATION', 'MODEL A', 'STANDARD', 'SIMPLIFIED',
                      'MODEL A-MAISONETTE', 'APARTMENT', 'MAISONETTE', 'TERRACE', '2-ROOM',
                      'IMPROVED-MAISONETTE', 'MULTI GENERATION', 'PREMIUM APARTMENT',
                      'ADJOINED FLAT', 'PREMIUM MAISONETTE', 'MODEL A2', 'DBSS', 'TYPE S1',
                      'TYPE S2', 'PREMIUM APARTMENT LOFT', '3GEN']
flat_type_options = ['1 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE', 'MULTI GENERATION']
town_options = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT TIMAH',
                'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG',
                'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE',
                'QUEENSTOWN', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS',
                'YISHUN', 'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS', 'PUNGGOL']
storey_range_options = ['10 TO 12', '04 TO 06', '07 TO 09', '01 TO 03', '13 TO 15', '19 TO 21',
                        '16 TO 18', '25 TO 27', '22 TO 24', '28 TO 30', '31 TO 33', '40 TO 42',
                        '37 TO 39', '34 TO 36', '46 TO 48', '43 TO 45', '49 TO 51']


# Load the saved model
model_filename = r'c:\\Users\\KAAY\\Downloads\\resale_price_prediction_decision_tree.joblib'
pipeline = joblib.load(model_filename)

# Streamlit app title
st.title("Singapore Resale Flat Prices Prediction ")

# Create a Streamlit sidebar with input fields
with st.sidebar:
    selected = option_menu("Main Menu", ["About Project", "Predictions"],
                           icons=["house", "gear"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "centre"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "#0072b1"},
                                   "icon": {"font-size": "20px"}
                                   }
                           )
    
if selected == "About Project":
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("### :blue[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, Data Wrangling, "
                "Model Deployment")
    st.markdown("### :blue[Overview :] This project aims to construct a machine learning model and implement "
                "it as a user-friendly online application in order to provide accurate predictions about the "
                "resale values of apartments in Singapore. This prediction model will be based on past transactions "
                "involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the "
                "worth of a flat after it has been previously resold. Resale prices are influenced by a wide variety "
                "of criteria, including location, the kind of apartment, the total square footage, and the length "
                "of the lease. The provision of customers with an expected resale price based on these criteria is "
                "one of the ways in which a predictive model may assist in the overcoming of these obstacles.")
    st.markdown("### :blue[Domain :] Real Estate")

if selected == "Predictions":
    town = st.selectbox("Town", options=town_options)
    flat_type = st.selectbox("Flat Type", options=flat_type_options)
    flat_model = st.selectbox("Flat Model", options=flat_model_options)
    storey_range = st.selectbox("Storey Range", options=storey_range_options)
    floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=0.0, max_value=500.0, value=100.0)
    current_remaining_lease = st.number_input("Current Remaining Lease", min_value=0.0, max_value=99.0, value=20.0)
    year = 2024
    lease_commence_date = current_remaining_lease + year - 99
    years_holding = 99 - current_remaining_lease

# Create a button to trigger the prediction
if st.button("Predict Resale Price"):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'town': [town],
        'flat_type': [flat_type],
        'flat_model': [flat_model],
        'storey_range': [storey_range],
        'floor_area_sqm': [floor_area_sqm],
        'current_remaining_lease': [current_remaining_lease],
        'lease_commence_date': [lease_commence_date],
        'years_holding': [years_holding],
        'remaining_lease': [current_remaining_lease],
        'year': [year]
    })

    # Make a prediction using the model
    prediction = pipeline.predict(input_data)

    # Display the prediction
    st.write("Predicted Resale Price:", prediction)