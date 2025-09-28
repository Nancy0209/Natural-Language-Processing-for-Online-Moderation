import gradio as gr
import plotly.graph_objects as go

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import os

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


MODEL = f"{os.getcwd()}/ckpts"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def plot_results(prediction):
    labels = list(prediction.keys())
    values = list(prediction.values())

    fig = go.Figure([go.Bar(x=labels, y=values)])
    fig.update_layout(title='Prediction Probabilities', xaxis_title='Classes', yaxis_title='Probability')

    return fig


def predict(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    o = {}
    for i in range(scores.shape[0]):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]]
        o[l] = float(s)
    
    print(o)

    if 'negative' in o:
        result = {"hate": 0., "non-hate": 0.}
        result['hate'] += o['negative']
        result['non-hate'] += (o['positive'] + o['neutral'] )
    elif 'LABEL_2' in o:
        result = {"hate": 0., "non-hate": 0.}
        result['hate'] += o['LABEL_1']
        result['non-hate'] += (o['LABEL_0'] + o['LABEL_2'] )
    else:
        result = {"hate": 0., "non-hate": 0.}
        result['hate'] += (o['0'] + o['1'])
        result['non-hate'] += o['2']
    
    return result
    
    
interface = gr.Interface(
    fn=lambda x: plot_results(predict(x)),
    inputs=gr.Textbox(lines=2, placeholder="Text Here...", label='Please input your tweet'),
    outputs="plot" 
).launch(share=True)