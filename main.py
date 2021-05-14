from flask import Flask, request, render_template,jsonify # Import flask libraries
import pickle
import numpy as np
# Initialize the flask class and specify the templates directory
app = Flask(__name__,template_folder="templates")

with open('IrisLogModel.pkl', 'rb') as f:
    model = pickle.load(f)
@app.route("/")
def hello():
    return "Hello World!"
# Default route set as 'home'
@app.route('/home')
def home():
    return render_template('home.html') # Render home.html

# Route 'classify' accepts GET request
@app.route('/classify',methods=['POST','GET'])
def classify_type():
    try:
        sepal_len = request.args.get('slen') # Get parameters for sepal length
        sepal_wid = request.args.get('swid') # Get parameters for sepal width
        petal_len = request.args.get('plen') # Get parameters for petal length
        petal_wid = request.args.get('pwid') # Get parameters for petal width
        arr = np.array([sepal_len, sepal_wid, petal_len, petal_wid]) # Convert to numpy array
        arr = arr.astype(np.float64) # Change the data type to float
        query = arr.reshape(1, -1) # Reshape the array
        variety_mappings = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        prediction = variety_mappings[model.predict(query)[0]] 

        # Render the output in new HTML page
        return render_template('output.html', variety=prediction)
    except:
        return 'Error'

# Run the Flask server
if(__name__=='__main__'):
    app.run(host='0.0.0.0')    