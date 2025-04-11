from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return 'POST received!'
    return 'GET received!'

if __name__ == '__main__':
    app.run()
