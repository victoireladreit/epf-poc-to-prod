from flask import Flask, request, jsonify, render_template_string
from run import TextPredictionModel

app = Flask(__name__)

artefacts_path = "../../train/data/artefacts/"
model = TextPredictionModel.from_artefacts(artefacts_path)


@app.route('/')
def home():
    return 'Hello World :))'


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = request.form.get('text')
        predictions = model.predict([text])

        # Ensure each element in predictions is serializable
        serializable_predictions = [str(prediction) for prediction in predictions]

        return jsonify(predictions=serializable_predictions, text=text)
    return '''
    <!DOCTYPE html>
    <html>
    <head>
    <title>Prediction Text</title>
    </head>
    <body>
        <h2>Enter text for prediction</h2>
        <form method="post" action="/predict">
            <textarea name="text" rows="4" cols="50"></textarea>
            <br><br>
            <input type="submit" value="Predict">
        </form>
    </body>
    </html>
    '''


if __name__ == '__main__':
    app.run(debug=True)