import streamlit as st
import time
import joblib
import sklearn
import catboost
import pandas
import numpy
import category_encoders

st.title("Airline Price")
st.write("This is a web app to predict the price of airline tickets.")

model = joblib.load('model.h5')
inputs = joblib.load('inputs.h5')

def predict(Airline, Time_Period, Source, Destination, Duration, Total_Stops, Route_Status, Additional_Info):
    test_df = pandas.DataFrame(columns=inputs)
    test_df.at[0, 'Airline'] = Airline
    test_df.at[0, 'Time_Period'] = Time_Period
    test_df.at[0, 'Source'] = Source
    test_df.at[0, 'Destination'] = Destination
    test_df.at[0, 'Duration'] = Duration
    test_df.at[0, 'Total_Stops'] = Total_Stops
    test_df.at[0, 'Route_Status'] = Route_Status
    test_df.at[0, 'Additional_Info'] = Additional_Info
    prediction = model.predict(test_df)
    return prediction[0]

def main():
    airline = st.selectbox('Airline', ['Vistara', 'Jet Airways', 'Air India', 'GoAir', 'IndiGo', 'Multiple carriers', 'SpiceJet', 'Other', 'Air Asia', 'Jet Airways Business', 'Multiple carriers Premium economy'])
    time_period = st.selectbox('Time Period', ['Morning', 'Afternoon', 'Evening', 'Night'])
    source = st.selectbox('Source', ['Delhi', 'Kolkata', 'Banglore', 'Mumbai', 'Chennai'])
    destination = st.selectbox('Destination', ['Cochin', 'Banglore', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata'])
    duration = st.number_input('Duration', min_value=0, max_value=10000, value=0)
    total_stops = st.slider('Total Stops', min_value=0, max_value=4, value=0, step=1)
    route_status = st.selectbox('Route Status', ['Direct', 'Indirect'])
    additional_info = st.selectbox('Additional Info', ['No info', 'In-flight meal not included', 'No check-in baggage included', '1 Short layover', 'No Info', '1 Long layover', 'Change airports', 'Business class', 'Red-eye flight', '2 Long layover'])
    results = predict(airline, time_period, source, destination, duration, total_stops, route_status, additional_info)
    
    if st.button('Predict'):
        with st.spinner('Wait for it...'):
            time.sleep(1)
        st.write('The price of the ticket is {:.2f}'.format(results))
    
if __name__ == '__main__':
    main()