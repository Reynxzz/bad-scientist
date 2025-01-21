from custom_cortex_llm.snowflake_mistral_agents import CrewSnowflakeLLM
from config import MODEL_NAME, MODEL_TEMPERATURE

def summarize_cortex(session, prompt):
    response = session.sql(
        f"SELECT snowflake.cortex.summarize('{str(prompt)}')",
    ).collect()[0][0]
    return response


def custom_table_summarizer(session, prompt):
    llm = CrewSnowflakeLLM(
        session=session,
        model_name=MODEL_NAME, 
        temperature=MODEL_TEMPERATURE
    )

    response = llm.call(f"""
                    You are a summarizer that summarize requirements and guide people to use a Snowflake tables and their columns.
                    Summarize text between the <content> below in less than 50 words, 
                    - If there is name, column/table/dataset name, keep it as it is.
                    - if there is a code, keep the code as it is but select the most important code to show.
                        
                    <content>
                    {prompt}
                    <content>

                    """)
    
    return response


# # test
# session = Session.builder.configs(connection_params).create()

# prompt = """
# Most Relevant Tables for the Query
# DRIVERS

# Description: Contains driver profile information.
# Key Columns:
# DRIVER_ID (TEXT)
# FIRST_NAME (TEXT)
# LAST_NAME (TEXT)
# EMAIL (TEXT)
# PHONE (TEXT)
# LICENSE_NUMBER (TEXT)
# RATING (FLOAT)
# VEHICLE_ID (TEXT)
# STATUS (TEXT)
# CREATED_AT (TIMESTAMP_NTZ)
# RIDES

# Description: Contains ride history information.
# Key Columns:
# RIDE_ID (TEXT)
# RIDER_ID (TEXT)
# DRIVER_ID (TEXT)
# PICKUP_LOCATION_LAT (FLOAT)
# PICKUP_LOCATION_LONG (FLOAT)
# DROPOFF_LOCATION_LAT (FLOAT)
# DROPOFF_LOCATION_LONG (FLOAT)
# REQUEST_TIME (TIMESTAMP_NTZ)
# PICKUP_TIME (TIMESTAMP_NTZ)
# DROPOFF_TIME (TIMESTAMP_NTZ)
# STATUS (TEXT)
# FARE (FLOAT)
# DISTANCE (FLOAT)
# PAYMENTS

# Description: Contains payment data.
# Key Columns:
# PAYMENT_ID (TEXT)
# RIDE_ID (TEXT)
# AMOUNT (FLOAT)
# PAYMENT_METHOD (TEXT)
# STATUS (TEXT)
# TRANSACTION_TIME (TIMESTAMP_NTZ)
# DRIVER_LOCATIONS

# Description: Contains real-time location data of drivers.
# Key Columns:
# LOCATION_ID (TEXT)
# DRIVER_ID (TEXT)
# LATITUDE (FLOAT)
# LONGITUDE (FLOAT)
# TIMESTAMP (TIMESTAMP_NTZ)
# Brief Explanation and Example Code
# To use these tables in a Streamlit app with a Snowflake connector, you can follow these steps:

# Install Required Libraries:

# pip install streamlit snowflake-connector-python

# Streamlit App Code:

# import streamlit as st
# import snowflake.connector

# # Establish Snowflake connection
# conn = snowflake.connector.connect(**conn_params)
# cursor = conn.cursor()

# # Query to fetch driver profiles
# cursor.execute("SELECT * FROM DRIVERS")
# drivers = cursor.fetchall()

# # Query to fetch ride history
# cursor.execute("SELECT * FROM RIDES")
# rides = cursor.fetchall()

# # Query to fetch payment data
# cursor.execute("SELECT * FROM PAYMENTS")
# payments = cursor.fetchall()

# # Query to fetch real-time location data
# cursor.execute("SELECT * FROM DRIVER_LOCATIONS")
# locations = cursor.fetchall()

# # Display data in Streamlit
# st.title("Driver Profiles")
# st.write(drivers)

# st.title("Ride History")
# st.write(rides)

# st.title("Payment Data")
# st.write(payments)

# st.title("Real-Time Location Data")
# st.write(locations)

# # Close the connection
# cursor.close()
# conn.close()

# This code connects to a Snowflake database, retrieves data from the relevant tables, and displays it in a Streamlit app. Adjust the connection parameters and queries as needed for your specific use case.


# """

# response = custom_table_summarizer(session, prompt)
# print(response)