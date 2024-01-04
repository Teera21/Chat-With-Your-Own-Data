from flask import Flask
# from flask_cors import CORS
from flask import render_template
import config
from flask import  jsonify, request
import json
from chat_retriever import question_answering

app = Flask(__name__)

@app.route('/generate_answer', methods=['POST'])
def predict():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        data = json.loads(request.data)
        text = data["question"]
    
    model = question_answering()
    answer = model.QA_Retriever(text)
    source,data = model.get_relevant()
    return jsonify({'Answer': str(answer),
                    'Resource':str(source),
                    'Data Reference':str(data)})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8401)
