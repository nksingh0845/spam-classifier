from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    message_vector = vectorizer.transform([message])
    result = model.predict(message_vector)[0]

    if result == 1:
        prediction = "🚫 SPAM MESSAGE"
    else:
        prediction = "✅ NOT SPAM"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
