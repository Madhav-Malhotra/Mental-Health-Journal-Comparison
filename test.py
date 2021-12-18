import requests

url = 'http://mdvmlhtr.pythonanywhere.com/compare'
myobj = {'entries': ['This is some text.', 'This is a test.']}

x = requests.post(url, data = myobj)
print(x.text)