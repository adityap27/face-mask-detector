import requests
import json

def sendSMS(msg,numbers):
    headers = {
    "authkey": "place AUTH-KEY here",
    "Content-Type": "application/json"
    }

    data = "{ \"sender\": \"GTURES\", \"route\": \"4\", \"country\": \"91\", \"sms\": [ { \"message\": \""+msg+"\", \"to\": "+json.dumps(numbers)+" } ] }"

    requests.post("https://api.msg91.com/api/v2/sendsms", headers=headers, data=data)
#sendSMS("demo",[7041677471])