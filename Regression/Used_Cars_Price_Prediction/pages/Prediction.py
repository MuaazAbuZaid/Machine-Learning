# Importing libraries
import streamlit as st
import pandas as pd
import pickle as pickle
import sklearn 
from datetime import datetime

# reading data
df = pd.read_csv('./Machine-Learning/Regression/Used Car Price Prediction/Sources/Cars_cleaned.csv')

# titles
st.markdown(" <center>  <h1> Prediction Page </h1> </font> </center> </h1> ",
            unsafe_allow_html=True)
st.markdown(" <center>  <h2> Enter Your Data </h2> </font> </center> </h2> ",
            unsafe_allow_html=True)

# storing data to use in selectboxes & radio choices
brands = tuple(df.Brand.unique())
brand = st.radio('Select Brand', brands, horizontal= True)
brand_models = tuple(df[df['Brand'] == brand]['Model'].unique())
model = st.selectbox('Select Model', brand_models, key= 'model_box')
bodies = list(df.Body.unique())
transmissions = list(df.Transmission.unique())
engine_sizes = tuple(df.Engine.unique())
fuel_type = tuple(df.Fuel.unique())
colors = tuple(df.Color.unique())
kilometers = tuple(df.Kilometers.unique())
govs = tuple(df.Gov.unique())

# page content
body_engine_cont = st.container()
body_col, trans_col = body_engine_cont.columns(2)
with body_col:
    body = st.radio(
                "Select body type",
                bodies,
                horizontal= True,
            )
with trans_col:
    trans = st.radio(
                "Select transmission type",
                transmissions,
                horizontal= True,
            )
    
engine = st.radio('Select engine size', engine_sizes, horizontal= True)

fuel = st.radio('Select fuel type', fuel_type, horizontal= True)

color = st.selectbox('Select color', colors, key= 'color_box')

kilometer = st.selectbox('Select range', kilometers, key= 'kilo_box')

gov = st.selectbox('Select governorate', govs, key= 'gov_box')

year = st.slider('Enter a year', 1970, (datetime.today().year - 1))
age = datetime.today().year - year # transfering year into age to use in prediction
# st.write(age)

# create dataframe for entries
pred_sample = {'Brand': brand, 'Model': model, 'Body': body, 'Transmission': trans, 'Engine': engine,
               'Fuel': fuel, 'Color': color, 'Kilometers': kilometer, 'Age': age, 'Gov': gov
               } 
pred_sample_df = pd.DataFrame(pred_sample, index= [0])

# mapping kilometers & engine sizes
kilometer_map = {'0 to 9999': 1, '10000 to 19999': 2, '100000 to 119999': 3, '20000 to 29999': 4, '30000 to 39999': 5,
                 '40000 to 49999': 6, '50000 to 59999': 7, '60000 to 69999': 8, '70000 to 79999': 9,  '80000 to 89999': 10, 
                   '90000 to 99999': 11,  '100000 to 119999': 12,  '120000 to 139999': 13,  '140000 to 159999': 14,
                   '160000 to 179999': 15,  '180000 to 199999': 16, 'More than 200000': 17
                 }
engine_map = {'1600 CC': 1, '1400 - 1500 CC': 2, '1000 - 1300 CC': 3}
pred_sample_df.Kilometers = df.Kilometers.map(kilometer_map)
pred_sample_df.Engine = df.Engine.map(engine_map)

# providing cat_cols & num_cols to be able to use transformer
cat_cols = ['Brand', 'Model', 'Body', 'Color', 'Fuel', 'Transmission', 'Gov']
num_cols = ['Kilometers', 'Engine', 'Age']

# loading transformer & model
t = pickle.load(open('./Machine-Learning/Regression/Used_Car_Price_Prediction/transf.pkl', 'rb'))
m = pickle.load(open('./Machine-Learning/Regression/Used_Car_Price_Prediction/RF.pkl', 'rb'))

# show the prediction when pressing the button
if st.button('Predict'):
    res_smpl = t.transform(pred_sample_df[cat_cols+num_cols])
    prediction = m.predict(res_smpl)[0]
    pred_shape = str(f'{prediction:,.0f}')
    st.metric("Price", str(pred_shape)) 
    
