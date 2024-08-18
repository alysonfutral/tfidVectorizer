
from sklearn import svm
# TfidVectorizer is used with the purpose of converting a string into a matrix of numbers that will be able to be used in our artificial intelligence program. The first step is to create our vectorizer object. Once the object is created, we will use the method fit_transform on our sentences that we have collected to create our input data.
from sklearn.feature_extraction.text import TfidfVectorizer

input_data = []
sentences = ["There was a decent amount of meat though and the noodles were great.", "The food is slightly worse than the other two locations.", "The beef noodle soup taste so bad and the male waiter(with glasses) even took a sip on it to prove that it's ok.", "My son says good food." , "The fish smells terrible."]

# When working with your input data, your data should be in the form of a list [] with each element inside the list being another sentence or string. Sentences and strings are not able to be processed with sklearn. To go around this issue, we will be using TfidVectorizer.

vectorizer = TfidfVectorizer()
input_data = vectorizer.fit_transform(sentences)

# Once you have your input data setup, you need to create your output data and have it match your input data based on what you have collected. The data will match whether the input sentences are positive or negative. If the sentence is positive, then the output data will be 1, and if the sentence is negative, then the output data will be 0.

output_data = [1, 0, 0, 1, 0]

#With both the input data and output data completed, you will need to create your model and algorithm. In this program, we will be once again using svm as the model and SVC as the algorithm.


model = svm.SVC()

#The next step in our program is to fit the input data and output data we have collected into the model.


model.fit(input_data, output_data)

# The only missing aspect of the program is to create test data and predict using the model we have created. The test sentences we use must also be in the same format as the input data, a list within a list.


test_sentences = [
 "The food I ate is worse than the old store.",
 "My dad said good food." ,
 "Why is the food terrible."
]

# When predicting using our model on our test sentences, we must again utilize the vectorizer to transform the test cases into a form that can be processed by sklearn.


print(model.predict(vectorizer.transform(test_sentences)))

# With the prediction step finished, you should be able to see the result of the machine predicting whether or not the test sentences were positive or negative.