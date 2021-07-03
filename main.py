
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Pregnancies =float(request.form['Pregnancies'])
            Glucose = float(request.form['Glucose'])
            BloodPressure = float(request.form['BloodPressure'])
            SkinThickness = float(request.form['SkinThickness'])
            Insulin = float(request.form['Insulin'])
            BMI = float(request.form['BMI'])
            DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
            Age = float(request.form['Age'])

            filename_scaler = 'standardScaler.sav'
            filename = 'DiabetesModelForPrediction.sav'

            # loading the model file from the storage
            scaler_model= pickle.load(open(filename_scaler, 'rb'))
            loaded_model = pickle.load(open(filename, 'rb'))

            # predictions using the loaded model file
            scaled_data=scaler_model.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            prediction=loaded_model.predict(scaled_data)
            print('prediction is', prediction[0])
            if prediction[0]==1:
                result='You have chances to Diabetic'
            else:
                result='You have not chances to Diabetic'

            # showing the prediction results in a UI
            return render_template('results.html',prediction=result)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
	#app.run(debug=True) # running the app