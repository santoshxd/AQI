
### Prerequisites
You must have Scikit Learn, Pandas (for Machine Learning Model) and Flask (for API) installed.

### Project Structure
This project has four major parts :
1. model.py - This contains code fot our Machine Learning model to predict Air Quality Index absed on training data in 'RealCombile.csv' file.
2. app.py - This contains Flask APIs that receives Air Quality details through API calls, computes the precited value based on our model and returns it.
3. templates - This folder contains the HTML template to allow user to enter details regarding atmosphere and displays the predicted Air Quality Index.

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python model.py ( model can be anything like Linear Regression,Random Forest,Decision Tree,XgBoost etc)
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

You should be able to view the homepage as below :
![alt text](Screenshots/Homepage.png)
Enter valid numerical values in all 8 input boxes.
![alt text](Screenshots/Enter_Details.png)
After Enterting valid inputs into 8 input boxes hit Predict.
![alt text](Screenshots/Entered.png)
If everything goes well, you should  be able to see the predicted AQI vaule on the HTML page!
![alt text](Screenshots/Result.png)


