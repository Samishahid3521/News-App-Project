from importlib.resources import read_text
from flask import Flask, render_template, request
import pickle
from news_checker import check_news_on_internet

app = Flask(__name__, )
           

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

API_KEY = "5023f90d19e34618b1a995200e2d638e"  # ← Replace with your actual NewsAPI key

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    input_data = [news]
    vectorized_input = vectorizer.transform(input_data)
    prediction = model.predict(vectorized_input)

    prediction_label = "Real" if prediction[0] == 1 else "Fake"

    # Check from internet
    internet_results = check_news_on_internet(news, API_KEY)
    found_online = bool(internet_results)

    if prediction_label == "Real" and found_online:
        final_result = "✅ Real News (Verified on Internet)"
    elif prediction_label == "Real" and not found_online:
        final_result = "⚠️ Possibly Fake (Not found online)"
    elif prediction_label == "Fake" and found_online:
        final_result = "⚠️ Confused Case (Check sources manually)"
    else:
        final_result = "❌ Fake News (No match on internet)"

    return render_template('index.html', prediction_text=final_result)

if __name__ == "__main__":
    import os
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug_mode)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
try:
    api_result = check_news_on_internet(read_text, API_KEY)
except Exception as e:
    api_result = {"error": str(e)}