# import streamlit as st
# import PIL
# import tensorflow as tf
# import tensorflow_hub as hub
# import numpy as np
# import pandas as pd
# from geopy.geocoders import Nominatim
# import os
# import sqlite3


# model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
# # model_url = 'on_device_vision_classifier_landmarks_classifier_asia_V1_1'

# # label_url = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_label_map.csv'
# labels = 'landmarks_classifier_asia_V1_label_map.csv'
# df = pd.read_csv(labels)
# labels = dict(zip(df.id, df.name))

# def image_processing(image):
#     img_shape = (321, 321)
#     classifier = tf.keras.Sequential(
#         [hub.KerasLayer(model_url, input_shape=img_shape + (3,), output_key="predictions:logits")])
#     img = PIL.Image.open(image)
#     img = img.resize(img_shape)
#     img1 = img
#     img = np.array(img) / 255.0
#     img = img[np.newaxis]
#     result = classifier.predict(img)
#     return labels[np.argmax(result)],img1

# def get_map(loc):
#     geolocator = Nominatim(user_agent="Your_Name")
#     location = geolocator.geocode(loc)
#     return location.address,location.latitude, location.longitude

# def run():
#     # st.title("Landmark Recognition")
#     st.title("Heritage Identification of Monuments using Deep Learning")
#     img = PIL.Image.open('logo.png')
    
#     img = img.resize((256,256))
#     st.image(img)
#     img_file = st.file_uploader("Choose your Image", type=['png', 'jpg'])
#     if img_file is not None:
#         save_image_path = './Uploaded_Images/' + img_file.name
#         with open(save_image_path, "wb") as f:
#             f.write(img_file.getbuffer())
#         prediction,image = image_processing(save_image_path)
#         st.image(image)
#         st.header("📍 **Predicted Landmark is: " + prediction + '**')
#         try:
#             address, latitude, longitude = get_map(prediction)
#             st.success('Address: '+address )
#             loc_dict = {'Latitude':latitude,'Longitude':longitude}
#             st.subheader('✅ **Latitude & Longitude of '+prediction+'**')
#             st.json(loc_dict)
#             data = [[latitude,longitude]]
#             df = pd.DataFrame(data, columns=['lat', 'lon'])
#             st.subheader('✅ **'+prediction +' on the Map**'+'🗺️')
#             st.map(df)
#         except Exception as e:
#             st.warning("No address found!!")
# run()


import streamlit as st
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import os
import sqlite3

model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
# model_url = 'on_device_vision_classifier_landmarks_classifier_asia_V1_1'

# label_url = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_label_map.csv'
labels = 'landmarks_classifier_asia_V1_label_map.csv'
df = pd.read_csv(labels)
labels = dict(zip(df.id, df.name))

def image_processing(image):
    img_shape = (321, 321)
    classifier = tf.keras.Sequential(
        [hub.KerasLayer(model_url, input_shape=img_shape + (3,), output_key="predictions:logits")])
    img = PIL.Image.open(image)
    img = img.resize(img_shape)
    img1 = img
    img = np.array(img) / 255.0
    img = img[np.newaxis]
    result = classifier.predict(img)
    return labels[np.argmax(result)],img1

def get_map(loc):
    geolocator = Nominatim(user_agent="Your_Name")
    location = geolocator.geocode(loc)
    return location.address, location.latitude, location.longitude

def run():
    st.title("Heritage Identification of Monuments using Deep Learning")
    
    img = PIL.Image.open('logo.png')
    img = img.resize((256,256))
    st.image(img)
    
    img_file = st.file_uploader("Choose your Image", type=['png', 'jpg'])
    if img_file is not None:
        save_image_path = './Uploaded_Images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        prediction, image = image_processing(save_image_path)
        st.image(image)
        st.header("📍 **Predicted Landmark is: " + prediction + '**')
        try:
            address, latitude, longitude = get_map(prediction)
            st.success('Address: ' + address)
            loc_dict = {'Latitude': latitude, 'Longitude': longitude}
            st.subheader('✅ **Latitude & Longitude of ' + prediction + '**')
            st.json(loc_dict)
            data = [[latitude, longitude]]
            df = pd.DataFrame(data, columns=['lat', 'lon'])
            st.subheader('✅ **' + prediction + ' on the Map**' + '🗺️')
            st.map(df)
        except Exception as e:
            st.warning("No address found!!")

    # Contribution Section
    st.subheader("💡 **Contribute to Our Project**")
    st.write("If you find this tool useful and want to contribute, here are a few ways you can help:")
    
    st.write("1. **Feedback**: Share your feedback on how we can improve this tool.")
    st.markdown("""
        <style>
        .feedback-form {
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            border-radius: 8px;
            background-color: #f0f4f8; /* Light background color for the form */
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .feedback-form label {
            display: block;
            margin-bottom: 10px;
            font-weight: bold;
            color: #333; /* Dark text color for better readability */
        }
        .feedback-form input[type="email"],
        .feedback-form textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ccc; /* Light grey border */
            border-radius: 4px;
            box-sizing: border-box;
            color: #333; /* Dark text color for input fields */
            background-color: #fff; /* White background for input fields */
        }
        .feedback-form textarea {
            height: 100px;
        }
        .feedback-form button {
            padding: 10px 20px;
            font-size: 16px;
            color: #fff;
            background-color: #007bff; /* Primary button color */
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 10px;
        }
        .feedback-form button:hover {
            background-color: #0056b3; /* Darker shade for button hover */
        }
        </style>
        <div class="feedback-form">
            <form action="https://formspree.io/f/xrbzdozv" method="POST">
                <label>
                    Your email:
                    <input type="email" name="email" required>
                </label>
                <label>
                    Your message:
                    <textarea name="message" required></textarea>
                </label>
                <button type="submit">Send</button>
            </form>
        </div>
    """, unsafe_allow_html=True)

    st.write("2. **Help Us Improve**: Have landmark images or data to contribute? Share your suggestions or uploads with us! [Submit Your Data](https://forms.gle/AUC9grqtY1kYkBR58)")
    st.write("3. **Spread the Word**: Share this tool with your friends and colleagues.")

run()
