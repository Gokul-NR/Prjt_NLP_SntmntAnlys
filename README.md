PROJECT: NLP - Sentiment Analysis & Prediction

REQUIREMENTS:

Please refer the pdf document named “NLP_Prjrqrmnt” attached above.

APPROACH:
- Cleaned data by looking for null, missing values and unknown values like Nan
- Converted object data type columns to lower case
- Removed white spaces, special characters, stop words, punctuation and repeated words in required columns
- Carried out spell check 
- Tokenized words in the review column using Tf-idf vectorizer
- Prepared training data and test data
- Mapped rating values to sentiments such as positive, neutral and negative
- Built a ML model for predicting the sentiment for user reviews using logistic regression  
- Evaluated the models using test data set and the results are shared in the insight&infrnc file
- Saved the model using pickle
- Loaded model in app.py file and designed Streamlit app that enables business team to enter the reviews and predict the sentiment for further enhancements

Streamlit App Deployment Instructions:
- Open a new notebook in VS code or in Google colab
- Upload the given pickle file of the NLP model in the same folder where the notebook is saved
- Run the cells to Install Streamlit and app.py in the notebook
- Use or click the link given in the terminal; Streamlit app appears in the web browser 
- Select from the options in the left pane; text box appears on the right side 
- Enter the user review and click Predict to see the outcome

