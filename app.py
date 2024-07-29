from flask import Flask

from flask import Flask, jsonify, request
import json
import pandas as pd
import os  # os 모듈 추가
from konlpy.tag import Okt
okt = Okt()

app = Flask(__name__)
@app.route('/tokenize', methods=['POST'])
def tokenize():
    data = request.get_json()
    text = data.get('text')
    print(text)
    if text:
        tokens = okt.morphs(text)
        print(tokens)
        totalPositive, totalNegative = makeToken(tokens)
        combined_result = f"{totalPositive} {totalNegative}"
        print(combined_result)
        return combined_result
    else:
        return jsonify({'error': 'No text provided'}), 400

def makeToken(tokens):
    totalPositive = 0
    totalNegative = 0

    # Initialize Okt tokenizer
    okt = Okt()

    # Tokenize the text into morphemes
    morphemes = []
    for token in tokens:
        morphemes.extend(okt.morphs(token))

    # Load SentiWord_info.json
    with open('SentiWord_info.json', encoding='utf-8-sig', mode='r') as f:
        SentiWord_info = json.load(f)

    # Create DataFrame from SentiWord_info
    sentiword_dic = pd.DataFrame(SentiWord_info)

    # Initialize DataFrame for sentiment results
    df = pd.DataFrame(columns=("content", "sentiment"))
    idx = 0

    for morpheme in morphemes:
        sentiment = 0

        # Check each word in sentiword_dic
        for i in range(len(sentiword_dic)):
            # Ensure exact match between morpheme and word in sentiword_dic
            if sentiword_dic['word'][i] == morpheme:
                if int(sentiword_dic['polarity'][i]) > 0:
                    totalPositive += int(sentiword_dic['polarity'][i])
                else:
                    totalNegative += int(sentiword_dic['polarity'][i])

    return totalPositive, totalNegative

        #df.loc[idx] = [morpheme, sentiment]
        #idx += 1


if __name__ == '__main__':
    app.run()
