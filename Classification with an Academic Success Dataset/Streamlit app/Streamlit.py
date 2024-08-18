import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context = 'notebook', palette = 'pastel', style = 'whitegrid')

# Streamlit app
st.title(' predict academic risk of students')

# Load the pre-trained model and encoder
model = joblib.load('best_model.pkl')
cat_encode = joblib.load('encode.pkl')

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)

    # Display the head of the data
    st.subheader('Data Preview')
    st.write(data.head())

    # Apply preprocessing (drop ID column)
    X_test = data.drop('id', axis=1)

    # Apply the model to make predictions
    y_pred = model.predict(X_test)

    # Create a DataFrame with the predictions
    output = pd.DataFrame({
        'id': data['id'],
        'Target': y_pred
    })

    # Inverse transform the target column
    output["Target"] = cat_encode.inverse_transform(output["Target"])

    # Display the head of the prediction output
    st.subheader('Prediction Output')
    st.write(output.head())

    # Plot the distribution of the predictions
    st.subheader('Prediction Distribution')

    fig, axes = plt.subplots(1, 2, figsize=(25, 5))
    sns.countplot(x='Target', data=output, palette='pastel', ax = axes[0])

    # Add labels to each bar in the plot
    for p in axes[0].patches:
        axes[0].text(p.get_x() + p.get_width() / 2, p.get_height() + 3, f'{int(p.get_height())}', ha="center")
    axes[0].set_title('Target Distribution')

    palette_color = sns.color_palette('pastel')
    explode = [0.1 for _ in range(output['Target'].nunique())]

    # Plotting
    output.groupby('Target')['Target'].count().plot.pie(
        colors=palette_color,
        explode=explode,
        autopct="%1.1f%%",
        shadow=True,  # Adding shadow for better visibility
        startangle=140,  # Start angle for better alignment
        textprops={'fontsize': 14},  # Adjust text size
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}, # Adding edge color and width
        ax = axes[1]
    )

    # Adding a title
    axes[1].set_title('Target Distribution')

    # Equal aspect ratio ensures that pie is drawn as a circle.
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    st.pyplot(plt)

# Create a download button
    st.subheader('Download Predictions')
    csv = output.to_csv(index=False)
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv',
    )