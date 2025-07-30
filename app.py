import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os
import gdown

model_path = './savedModels/plant_deficiency_model.h5'
# file_id = '1v-P2KD916TAevWcQHrOUeNm4KzhSnuZp'
# url = f'https://drive.google.com/uc?id={file_id}'


# if not os.path.exists(model_path):
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     gdown.download(url, model_path, quiet=False, fuzzy=True)

model = load_model(model_path)

class_labels =[
    "ashgourd_fresh",
    "ashgourd_nitrogen",
    "ashgourd_potassium",
    "bittergourd_fresh",
    "bittergourd_nitrogen",
    "bittergourd_potassium",
    "snakegourd_fresh",
    "snakegourd_nitrogen",
    "snakegourd_potassium",

]

# Streamlit UI
st.title("🌿 Plant Nutrient Deficiency Detection")
st.write("Upload a leaf image to detect possible nutrient deficiency.")

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Leaf Image", use_container_width=True)

    # Preprocess
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    probabilities = prediction[0]

    # ✅ Print debug lengths
    st.write(f"🔢 Model returned {len(probabilities)} probabilities")
    st.write(f"🏷 You have {len(class_labels)} class labels")

    # Check if lengths match
    if len(probabilities) == len(class_labels):
            predicted_index = np.argmax(probabilities)
            predicted_class = class_labels[predicted_index]
            confidence = probabilities[predicted_index] * 100

            # Create chart DataFrame
            df_probs = pd.DataFrame({
                'Class': class_labels,
                'Probability': probabilities

            })

            # Show results
            st.subheader("📊 Prediction Probabilities")
            st.bar_chart(df_probs.set_index('Class'))

            st.success(f"✅ Predicted Class: {predicted_class} ({confidence:.2f}% confidence)")

            # Download report
            report_text = f"""
            Plant Nutrient Deficiency Detection Report
            ------------------------------------------
            Predicted Class: {predicted_class}
            Confidence: {confidence:.2f}%
            # Button to download report
            st.download_button(
                label="📄 Download Report",
                data=report_text,
                file_name="plant_deficiency_report.txt",
                mime="text/plain"
            )

            All Class Probabilities:
            {df_probs.to_string(index=False)}
            """

            st.download_button(
                "📄 Download Report",
                data=report_text,
                file_name="prediction_report.txt",
                mime="text/plain"
            )