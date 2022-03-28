from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('card2.html')

if __name__=="__main__":
    app.run("0.0.0.0")