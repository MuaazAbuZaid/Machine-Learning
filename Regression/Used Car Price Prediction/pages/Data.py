# importing neede libraries
import streamlit as st
import pandas as pd

# Title
st.markdown(" <center>  <h1> Used Car Dataset </h1> </font> </center> </h1> ",
            unsafe_allow_html=True)

# Link of Data
st.markdown('<a href="https://www.kaggle.com/datasets/abdo977/used-car-price-in-egypt"> <center> Link to Dataset  </center> </a> ', unsafe_allow_html=True)

# Load data
df = pd.read_csv('Sources/Cars.csv')

# Show data
st.write('Data collected from a popular website for buying and selling in Egypt. This data contains +14K cars includes 5692 Hyundai, 5033 Fiat and 4016 Chevrolet cars in Egypt.')
st.write(df)
st.write(df.shape)