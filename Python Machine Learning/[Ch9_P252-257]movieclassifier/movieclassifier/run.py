# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:22:33 2017
"Python Machine Learning"
Chapter 9
Page: 252-257
    movieclassifier
@author: vincchen
"""


import pickle
import re
import os
from vectorizer import vect
clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))

"""
After we have successfully loaded the vectorizer and unpickled the classifier, we can now use these objects 
to pre-process document samples and make predictions about their sentiment:
"""
import numpy as np
label = {0:'negative', 1:'positive'}
example = ['I love and like this movie happy sad']
X = vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%' % (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))

"""
Since our classifier returns the class labels as integers, we defined a simple Python dictionary to map those integers to their sentiment. We then used the HashingVectorizer to transform the simple example document into a word vector X. Finally, we used the predict method of the logistic regression classifier to predict the class label as well as the predict_proba method to return the corresponding probability of our prediction. Note that the predict_proba method call returns an array with a probability value for each unique class label. Since the class label with the largest probability corresponds to the class label that is returned by the predict call, we used the np.max function to return the probability of the predicted class.
"""

####################################################################################################
# Setting up a SQLite database for data storage
####################################################################################################
"""
By executing the following code, we will create a new SQLite database inside the movieclassifier directory 
and store two example movie reviews:
"""
import sqlite3
conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
c.execute('CREATE TABLE review_db' ' (review TEXT, sentiment INTEGER, date TEXT)')
example1 = 'I love this movie'
c.execute("INSERT INTO review_db" " (review, sentiment, date) VALUES" " (?, ?, DATETIME('now'))", (example1, 1))
example2 = 'I disliked this movie'
c.execute("INSERT INTO review_db" " (review, sentiment, date) VALUES" " (?, ?, DATETIME('now'))", (example2, 0))
conn.commit()
conn.close()
"""
Following the preceding code example, we created a connection (conn) to an SQLite database file by calling sqlite3's connect method, which created the new database file reviews.sqlite in the movieclassifier directory if it didn't already exist. Please note that SQLite doesn't implement a replace function for existing tables; you need to delete the database file manually from your file browser if you want to execute the code a second time. Next, we created a cursor via the cursor method, which allows us to traverse over the database records using the powerful SQL syntax. Via the first execute call, we then created a new database table, review_db. We used this to store and access database entries. Along with review_db, we also created three columns in this database table: review, sentiment, and date. We used these to store two example movie reviews and respective class labels (sentiments). Using the SQL command DATETIME('now'), we also added date-and timestamps to our entries. In addition to the timestamps, we used the question mark symbols (?) to pass the movie review texts (example1 and example2) and the corresponding class labels (1 and 0) as positional arguments to the execute method as members of a tuple. Lastly, we called the commit method to save the changes that we made to the database and closed the connection via the close method.
"""

"""
To check if the entries have been stored in the database table correctly, we will now reopen the connection to the database and use the SQL SELECT command to fetch all rows in the database table that have been committed between the beginning of the year 2015 and today:
"""
conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
c.execute("SELECT * FROM review_db WHERE date" " BETWEEN '2015-01-01 00:00:00' AND DATETIME('now')")
results = c.fetchall()
conn.close()
print(results)

