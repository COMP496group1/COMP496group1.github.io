import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, classification_report

#Load the dataset into pandas and drop any duplicates found in the csv file
df = pd.read_csv("emails.csv")
df.drop_duplicates(inplace = True)

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["spam"], test_size=0.2, random_state=42)

#Create CountVectorizer object
vectorizer = CountVectorizer()

#Transform the training and testing data
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

#Create a MultinomialNB object and fit it to the training data
nb = MultinomialNB()
nb.fit(X_train_counts, y_train)

#Make predictions on the testing data
y_pred = nb.predict(X_test_counts)

#Obtain User Input and transform it 
email = input("Enter Subject Line Content: ")
email_counts = vectorizer.transform([email])

#Set up the prediction to compared to the user input 
prediction = nb.predict(email_counts)[0]
if prediction == 1:
    print("This email is spam.")
else:
    print("This email is not spam.")

#Evaluate the performance of the algorithm
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

#Print classification report
print(classification_report(y_test, y_pred))

#Non-spam Examples: "Amplifon USA is hiring for 2023 Summer Internship Program"
#Spam Examples: "Congrats you won some money!"