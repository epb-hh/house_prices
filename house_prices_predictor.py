import streamlit as st
import pandas as pd
import numpy as np
import pickle
from math import exp
from sklearn.ensemble import RandomForestRegressor
import warnings
from treeinterpreter import treeinterpreter
from waterfall_chart import plot as waterfall

st.write('''
# House Pricing Prediction App

This app predicts the **Housing Prices** given a number of features.
''')

st.write("""
----
""")

# Simulate data
st.sidebar.header("House Features")

data = {'home_type': 'house',
        'home_age': 'new',
        'state': 'aguascalientes',
        'parking': 2,
        'presale': False,
        'rooms': 4,
        'm2': 120
        }

features = pd.DataFrame(data, index=[0])

# Process data

home_type_classes = ['#na#', 'apartment', 'house']
home_age_classes = ['#na#', 'new', 'used']
state_classes = ['#na#', 'aguascalientes', 'baja-california', 'baja-california-sur', 'campeche', 'chiapas', 'chihuahua',
                 'coahuila', 'colima', 'distrito-federal', 'durango', 'estado-de-mexico', 'guanajuato', 'guerrero',
                 'hidalgo', 'jalisco', 'michoacan', 'morelos', 'nayarit', 'nuevo-leon', 'oaxaca', 'puebla', 'queretaro',
                 'quintana-roo', 'san-luis-potosi', 'sinaloa', 'sonora', 'tabasco', 'tamaulipas', 'tlaxcala',
                 'veracruz', 'yucatan', 'zacatecas']
home_age_classes = ['#na#', 'new', 'used']


def user_input_features():
    home_type = st.sidebar.radio('HomeType', ['apartment', 'house'])
    home_age = st.sidebar.radio('HomeAge', ['new', 'used'])
    state = st.sidebar.selectbox('Region', state_classes, 9)
    parking = st.sidebar.slider('Parking', min_value=0, max_value=4, value=2)
    presale = st.sidebar.radio('Presale', [True, False], 1)
    rooms = st.sidebar.slider('Rooms', min_value=1, max_value=10, value=4)
    m2 = st.sidebar.number_input('Construction in m2', value=120)
    data = {'home_type': home_type,
            'home_age': home_age,
            'state': state,
            'parking': parking,
            'presale': presale,
            'rooms': rooms,
            'm2': m2
            }
    return data


user_data = user_input_features()


def encode_var(data_point, type_class):
    for item in type_class:
        if item == data_point:
            return type_class.index(item)
    return '#na#'


def encode_data(input_data):
    encoded = {'home_type': encode_var(input_data['home_type'], home_type_classes),
               'home_age': encode_var(input_data['home_age'], home_age_classes),
               'state': encode_var(input_data['state'], state_classes),
               'parking': input_data['parking'],
               'presale': input_data['presale'],
               'rooms': input_data['rooms'],
               'm2': input_data['m2']
               }
    return pd.DataFrame(encoded, index=[0])


def format_number(number):
    return "${:,.2f}".format(exp(number))


# Price Prediction

m = pickle.load(open('m.pkl', 'rb'))
prediction = m.predict(encode_data(user_data))

st.subheader('Predicted house price:')
st.header(format_number(prediction))

st.write("""
----
""")

st.subheader('Features:')
st.write(pd.DataFrame(user_data, index=[0]))


st.write("""
----
""")

# Price interpretation

warnings.simplefilter('ignore', FutureWarning)

prediction, bias, contributions = treeinterpreter.predict(m, encode_data(user_data))

f = waterfall(encode_data(data).columns, contributions[0], threshold=0.08,
              rotation_value=45, formatting='{:,.3f}')

st.subheader('Price appreciation from the house mean price (in logs per feature):')
st.pyplot(f)

st.write("""
----
""")
