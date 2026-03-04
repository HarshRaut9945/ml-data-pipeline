from flask import Flask
from src.logger import logging
from src.exception import CustmeException

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    try:
         logging.info("We are testing our second method of logging")  
         return "Welcome to Enginering wale bhaiya"


if __name__=="__main__":
    app.run(debug=True)