from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the saved model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

app = Flask(__name__)

# Function to get image filename based on crop name
def get_crop_image(crop_name):
    crop_images = {
        'Rice': 'rice.jpg',
        'Maize': 'maize.jpg',
        'Jute': 'jute.jpg',
        'Cotton': 'cotton.jpg',
        'Coconut': 'coconut.jpg',
        'Papaya': 'papaya.jpg',
        'Orange': 'orange.jpg',
        'Apple': 'apple.jpg',
        'Muskmelon': 'muskmelon.jpg',
        'Watermelon': 'watermelon.jpg',
        'Grapes': 'grapes.jpg',
        'Mango': 'mango.jpg',
        'Banana': 'banana.jpg',
        'Pomegranate': 'pomegranate.jpg',
        'Lentil': 'lentil.jpg',
        'Blackgram': 'blackgram.jpg',
        'Mungbean': 'mungbean.jpg',
        'Mothbeans': 'mothbeans.jpg',
        'Pigeonpeas': 'pigeonpeas.jpg',
        'Kidneybeans': 'kidneybeans.jpg',
        'Chickpea': 'chickpea.jpg',
        'Coffee': 'coffee.jpg'
    }
    return crop_images.get(crop_name, 'default.jpg')

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Retrieve form data and convert to float
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Prepare feature list and prediction
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Apply scaling
        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)
        prediction = model.predict(sc_mx_features)

        # Debug: Print feature list and prediction
        print(f"Features: {feature_list}")
        print(f"Scaled Features: {sc_mx_features}")
        print(f"Prediction: {prediction}")

        # Map prediction to crop
        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        # Get crop name and result
        crop = crop_dict.get(prediction[0], 'Unknown')
        result = "{} is the best crop to grow in your region.".format(crop)
    except Exception as e:
        result = "Error occurred: {}".format(str(e))
        crop = 'Unknown'  # Set default crop in case of error

    # Render the template with result and crop
    return render_template('index.html', result=result, crop=crop, get_crop_image=get_crop_image)

if __name__ == "__main__":
    app.run(debug=True)
