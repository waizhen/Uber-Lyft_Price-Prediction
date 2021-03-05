#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from catboost import CatBoostRegressor, Pool
import streamlit as st
import pandas as pd
import numpy as np

model = CatBoostRegressor()
model.load_model("catboost_uberlyft_05032021")

sources = pd.Series(['Back Bay', 'Beacon Hill', 'Boston University', 'Fenway', 'Financial District', 'Haymarket Square',
                     'North End', 'North Station', 'Northeastern University', 'South Station', 'Theatre District', 
                     'West End'])
destinations = pd.Series(['Back Bay', 'Beacon Hill', 'Boston University', 'Fenway', 'Financial District', 'Haymarket Square',
                          'North End', 'North Station', 'Northeastern University', 'South Station', 'Theatre District', 
                          'West End'])
cab_names = pd.DataFrame({"cab_type": ["Uber", "Uber", "Uber", "Uber", "Uber", "Uber", 
                                       "Lyft", "Lyft", "Lyft", "Lyft", "Lyft", "Lyft"], 
                          "name": ['UberXL', 'Black', 'UberX', 'WAV', 'Black SUV', 'UberPool', 
                                   'Shared', 'Lux', 'Lyft', 'Lux Black XL', 'Lyft XL', 'Lux Black']})
weather = pd.DataFrame({"short_summary": [' Clear ', ' Clear ', ' Clear ', ' Clear ', ' Clear ', ' Clear ', ' Drizzle ',
                                          ' Drizzle ', ' Foggy ', ' Foggy ', ' Foggy ', ' Light Rain ', ' Light Rain ', 
                                          ' Light Rain ', ' Light Rain ', ' Light Rain ', ' Mostly Cloudy ', ' Mostly Cloudy ',
                                          ' Mostly Cloudy ', ' Mostly Cloudy ', ' Mostly Cloudy ', ' Mostly Cloudy ', 
                                          ' Mostly Cloudy ', ' Overcast ', ' Overcast ', ' Overcast ', ' Overcast ', ' Overcast ', 
                                          ' Overcast ', ' Overcast ', ' Overcast ', ' Overcast ', ' Overcast ', ' Overcast ',
                                          ' Partly Cloudy ', ' Partly Cloudy ', ' Partly Cloudy ', ' Partly Cloudy ', ' Partly Cloudy ',
                                          ' Partly Cloudy ', ' Possible Drizzle ', ' Possible Drizzle ', ' Possible Drizzle ',
                                          ' Possible Drizzle ', ' Rain ', ' Rain ', ' Rain '], 
                        "long_summary": [' Foggy in the morning. ', ' Light rain in the morning and overnight. ',
                                         ' Light rain in the morning. ', ' Mostly cloudy throughout the day. ',
                                         ' Partly cloudy throughout the day. ', ' Rain throughout the day. ',
                                         ' Light rain in the morning. ', ' Rain until morning, starting again in the evening. ',
                                         ' Foggy in the morning. ', ' Rain in the morning and afternoon. ',
                                         ' Rain until morning, starting again in the evening. ', ' Light rain in the morning and overnight. ',
                                         ' Light rain in the morning. ', ' Light rain until evening. ', ' Rain throughout the day. ',
                                         ' Rain until morning, starting again in the evening. ', ' Foggy in the morning. ',
                                         ' Light rain in the morning and overnight. ', ' Light rain in the morning. ',
                                         ' Mostly cloudy throughout the day. ', ' Overcast throughout the day. ',
                                         ' Partly cloudy throughout the day. ', ' Rain throughout the day. ', ' Foggy in the morning. ',
                                         ' Light rain in the morning and overnight. ', ' Light rain in the morning. ',
                                         ' Light rain until evening. ', ' Mostly cloudy throughout the day. ', ' Overcast throughout the day. ',
                                         ' Partly cloudy throughout the day. ', ' Possible drizzle in the morning. ',
                                         ' Rain in the morning and afternoon. ', ' Rain throughout the day. ',
                                         ' Rain until morning, starting again in the evening. ', ' Foggy in the morning. ',
                                         ' Light rain in the morning and overnight. ', ' Light rain in the morning. ',
                                         ' Mostly cloudy throughout the day. ', ' Partly cloudy throughout the day. ',
                                         ' Rain throughout the day. ', ' Light rain in the morning. ', ' Light rain until evening. ',
                                         ' Rain throughout the day. ', ' Rain until morning, starting again in the evening. ',
                                         ' Light rain in the morning. ', ' Rain throughout the day. ', ' Rain until morning, starting again in the evening. ']
})

@st.cache
def predict(model, input_df):
    predictions_arr = model.predict(input_df)
    predictions = predictions_arr[0]
    return predictions

def run():
    
    from PIL import Image
    image_cab = Image.open("Lyft-Uber.jpg")
    image_phone = Image.open("Phone-interface.jpg")
    
    #st.image(image_cab, use_column_width=False)
    st.markdown(
    """
    <style>
    .reportview-container .markdown-text-container {
        font-family: monospace;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    .Widget>label {
        color: white;
        font-family: monospace;
    }
    [class^="st-b"]  {
        color: #EA738DFF;
        font-family: monospace;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #e2e9f3;
    }
    footer {
        font-family: monospace;
    }
    .reportview-container .main footer, .reportview-container .main footer a {
        color: #e2e9f3;
    }
    header .decoration {
        background-image: none;
    }

    </style>
    """,
        unsafe_allow_html=True,
    )
    
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", 
        ("Online", "Batch")
    )
    
    st.sidebar.info("This app is built to predict the price rate & fare of Uber or Lyft rides in Boston area in Nov & Dec")
    st.sidebar.success("https://catboost.ai/")
    
    st.sidebar.image(image_cab)
    
    #st.title("Uber & Lyft Ride Price Rate Prediction App")

    st.write("""
        ## Uber and Lyft Ride Price Rate Prediction App

        Key in the inputs to predict the fare rate for your ride!

    """)
    
    with st.beta_expander("View App Overview"):
        col1, col2 = st.beta_columns([1, 2])
        col1.image(image_phone)
        col2.write("""
            The price rates of rides offered by ride-sharing companies like Uber and Lyft change dynamically and 
            fluctuate without you knowing it. You often hear about frequent complaints about how ridiculous Uber 
            charged their customers at a certain time period or when they travelled to a certain region, which really 
            pissed customers off! Do you want to get the best value out of your ride and plan your ride expenses 
            before you take a ride? This app is your lifesaver!
        """)
        
    
    if add_selectbox == "Online":
        
        hour = st.number_input("Hour (24-hour clock)", min_value=0, max_value=23, value=12, step=1)
        day = st.number_input("Day", min_value=1, max_value=30, value=1, step=1)
        month = st.selectbox("Month", [11, 12])
        day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        source = st.selectbox("Your Location", list(sources))
        destination = st.selectbox("Your Destination", list(destinations[destinations != source]))
        path = source + "-" + destination
        cab_type = st.radio("Cab Type", ["Uber", "Lyft"])
        
        name = st.selectbox("Choose your cab product:", list(cab_names.loc[cab_names['cab_type'] == cab_type, "name"]))
        distance = st.number_input("Distance (mile)", min_value=0.5, max_value=10.0, step=0.5)
        surge_multiplier = st.selectbox("Surge Multiplier", [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
        
        temperature = st.number_input("Temperature (Degree Celcius)", min_value=18, max_value=60, value=25, step=1)
        short_summary = st.selectbox("Short Summary of Weather", weather['short_summary'].unique())
        long_summary = st.selectbox("Long Summary of Weather", list(weather.loc[weather['short_summary'] == short_summary, "long_summary"]))
        humidity = 0.74
        windSpeed = 6.19
        windGust = 8.47
        visibility = 8.47
        pressure = 1010.09
        cloudCover = 0.69
        moonPhase = 0.58
        
        if short_summary in [" Rain ", " Light Rain "]:
            precipIntensityMax = st.slider("Max Rain Intensity", min_value=0.2, max_value=0.4, value=0.2, step=0.02)
        elif short_summary in [" Drizzle ", " Possible Drizzle ", " Foggy "]:
            precipIntensityMax = st.slider("Max Rain Intensity", min_value=0.1, max_value=0.2, value=0.1, step=0.02)
        else:
            precipIntensityMax = st.slider("Max Rain Intensity", min_value=0.0, max_value=0.1, value=0.0, step=0.02)
        
        output = ""
        
        input_dict = {'hour':hour, 'day':day, 'month':month, 'source':source, 'destination':destination, 'cab_type':cab_type, 
                      'name':name, 'distance':distance, 'surge_multiplier':surge_multiplier, 'temperature':temperature, 
                      'short_summary':short_summary, 'long_summary':long_summary, 'humidity':humidity, 'windSpeed':windSpeed,
                      'windGust':windGust, 'visibility':visibility, 'pressure':pressure, 'cloudCover':cloudCover, 
                      'moonPhase':moonPhase, 'precipIntensityMax':precipIntensityMax, 'day_of_week':day_of_week, 'path':path}
        input_df = pd.DataFrame([input_dict])
        
        if st.button("Predict Price Rate & Fare"):
            output = predict(model=model, input_df=input_df)
            print_output = "$" + str(round(output, 2))
            fare = output * distance
            print_fare = "$" + str(round(fare, 2))
            st.success("The predicted price rate for the ride is {} per mile. Your total fare is {}.".format(print_output, print_fare))
        
        #st.success("The predicted price rate for the ride is {} per mile.".format(output))
        
        
    if add_selectbox == "Batch":
        
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = model.predict(data)
            pred_df = data.copy()
            pred_df['prediction'] = predictions
            st.write(pred_df)
            
if __name__ == "__main__":
    run()


# In[ ]:




