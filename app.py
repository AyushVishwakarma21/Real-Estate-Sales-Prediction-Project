from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            data = CustomData(
                List_Year=int(request.form['List_Year']),
                Assessed_Value=float(request.form['Assessed_Value']),
                Sales_Ratio=float(request.form['Sales_Ratio']),
                Property_Type=request.form['Property_Type'],
                Residential_Type=request.form['Residential_Type'],
                Town=request.form['Town']
            )
            df = data.get_data_as_dataframe()
            pipeline = PredictPipeline()
            prediction_value = pipeline.predict(df)[0]
            prediction = f"${prediction_value:,.2f}"
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('home.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

    #  http://127.0.0.1:5000/. 