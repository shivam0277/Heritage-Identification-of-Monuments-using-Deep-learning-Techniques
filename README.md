# Heritage Identification of Monuments using Deep learning Techniques

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)                 
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)   


## Source
- Trained model [`landmarks_classifier_asia_V1/1`](https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1) is taken from the Tensorflow-Hub
- There are total `98961` classes supported, in which Asia's most of the famous Landmark is covered.
- This model was trained on [`Google Landmarks Dataset V2`](https://ai.googleblog.com/2019/05/announcing-google-landmarks-v2-improved.html). 

## Features
- Simple repsonsive UI.
- It will give you the full address of Landmark
- It will provide you the `Latitude` & `Longitude` of predicted landmark.
- It will plot the predicted landmark on the Map.

## Usage

- Clone my repository.
- Open CMD in working directory.
- Run following command.

  ```
  pip install -r requirements.txt
  ```
- `LM_Detection.py` is the main Python file of Streamlit Web-Application. 
- To run app, write following command in CMD. or use any IDE.

  ```
  streamlit run LM_Detection.py
  ```

- For more explanation of this project see the tutorial on Machine Learning Hub YouTube channel.

