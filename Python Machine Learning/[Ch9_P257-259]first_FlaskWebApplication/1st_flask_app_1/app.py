# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:53:02 2017
"Python Machine Learning"
Chapter 9
Page: 257-259
Our first Flask web application

@author: vincchen
"""
"""
we create a directory tree
1st_flask_app_1/
    app.py
    templates/
        first_app.html

The app.py file will contain the main code that will be executed by the Python interpreter to run the Flask web application.
The templates directory is the directory in which Flask will look for static HTML files for rendering in the web browser
"""

from flask import Flask, render_template
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('first_app.html')
if __name__ == '__main__':
    app.run()

"""
let's start our web application by executing the command from the terminal inside the 1st_flask_app_1 directory:
We should now see a line such as the following displayed in the terminal:
* Running on http://127.0.0.1:5000/

This line contains the address of our local server. We can now enter this address in our web browser to see the web application in action. If everything has executed correctly, we should now see a simple website with the content: Hi, this is my first Flask web app!.
"""