import os
from dotenv import load_dotenv

load_dotenv()
CONNECTION_PARAMETER = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_USER_PASSWORD"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE")
}

DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
MODEL_NAME = "mistral-large2"
MODEL_TEMPERATURE = 0.9

STREAMLIT_TEMPLATE = """
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import col
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Snowflake connection
@st.cache_resource
def get_snowflake_session():
    return Session.builder.configs({
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA")
    }).create()

# Data loading function
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data(session):
    try:
        # Example query - modify based on your needs
        query = "SELECT * FROM YOUR_TABLE WHERE DATE_COLUMN >= DATEADD(month, -12, CURRENT_DATE())"
        df = session.sql(query).to_pandas()
        df.columns = df.columns.str.lower()  # Convert column names to lowercase
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def main():
    # Title and description
    st.title("📊 Interactive Data Analysis Dashboard")
    st.markdown("Analyze your data with interactive visualizations and filters")

    # Initialize session
    session = get_snowflake_session()

    # Load data
    with st.spinner('Loading data...'):
        df = load_data(session)

    if df is not None:
        # Sidebar filters
        st.sidebar.header("📎 Filters")

        # Date range filter
        st.sidebar.subheader("Date Range")
        date_col = 'date'  # Replace with your date column name
        min_date = pd.to_datetime(df[date_col]).min()
        max_date = pd.to_datetime(df[date_col]).max()
        
        date_range = st.sidebar.date_input(
            "Select date range",
            value=(min_date, max_date),
        )

        # Category filter (example)
        if 'category' in df.columns:
            categories = sorted(df['category'].unique())
            selected_categories = st.sidebar.multiselect(
                "Select Categories",
                options=categories,
                default=categories[:3]
            )

        # Metric selector
        available_metrics = [col for col in df.select_dtypes(include=['float64', 'int64']).columns]
        selected_metric = st.sidebar.selectbox("Select Metric", available_metrics)

        # Apply filters (make sure to prevent any data type error)
        mask = (pd.to_datetime(df[date_col]) >= pd.Timestamp(date_range[0])) & (pd.to_datetime(df[date_col]) <= pd.Timestamp(date_range[1]))

        if 'category' in df.columns:
            mask &= df['category'].isin(selected_categories)
        filtered_df = df[mask]

        # Main content area
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Trend Analysis")
            
            # Time series plot
            fig_time = px.line(
                filtered_df,
                x=date_col,
                y=selected_metric,
                title=f"{selected_metric} Over Time"
            )
            st.plotly_chart(fig_time, use_container_width=True)

            # Additional statistics
            st.metric(
                label=f"Average {selected_metric}",
                value=f"{filtered_df[selected_metric].mean():.2f}",
                delta=f"{filtered_df[selected_metric].mean() - df[selected_metric].mean():.2f}"
            )

        with col2:
            st.subheader("📊 Distribution Analysis")
            
            # Plot type selector
            plot_type = st.radio(
                "Select Plot Type",
                ["Histogram", "Box Plot", "Violin Plot"]
            )

            fig_dist = plt.figure(figsize=(10, 6))
            if plot_type == "Histogram":
                sns.histplot(data=filtered_df, x=selected_metric, kde=True)
            elif plot_type == "Box Plot":
                sns.boxplot(data=filtered_df, y=selected_metric)
            else:
                sns.violinplot(data=filtered_df, y=selected_metric)
            
            plt.title(f"{plot_type} of {selected_metric}")
            st.pyplot(fig_dist)

        # Data table with sorting and filtering
        st.subheader("🔍 Detailed Data View")
        
        # Number of rows selector
        n_rows = st.slider("Number of rows to display", 5, 50, 10)
        
        # Column selector
        selected_columns = st.multiselect(
            "Select columns to display",
            options=df.columns,
            default=list(df.columns[:5])
        )

        # Display interactive table
        st.dataframe(
            filtered_df[selected_columns].head(n_rows),
            use_container_width=True,
            hide_index=True
        )

        # Download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name="filtered_data.csv",
            mime="text/csv"
        )

        # Footer
        st.markdown("---")
        st.markdown("Dashboard created with Streamlit and Snowflake")

if __name__ == "__main__":
    main()
"""