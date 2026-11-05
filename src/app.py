"""Aplicação Gradio para modelo final, sem alegar previsão hídrica."""
from pathlib import Path
import numpy as np
import tensorflow as tf
import gradio as gr

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'outputs'/'best.keras'
if not MODEL.exists(): raise SystemExit('Execute src/train.py antes de iniciar a aplicação: outputs/best.keras não existe.')
model=tf.keras.models.load_model(MODEL)
classes=['Forest','Industrial','Residential','River']
def classify(image):
    if image is None: raise gr.Error('Selecione uma imagem antes de classificar.')
    image=image.convert('RGB').resize((64,64))
    x=np.asarray(image,dtype='float32')[None,...]/255.0
    probabilities=model.predict(x,verbose=0)[0]
    return {label:float(probabilities[i]) for i,label in enumerate(classes)}
with gr.Blocks(title='Territory Vision') as app:
    gr.Markdown('# Territory Vision\nClassificação experimental de imagens semelhantes às do EuroSAT. O resultado não mede escassez hídrica nem substitui análise especializada.')
    inp=gr.Image(type='pil',label='Imagem de satélite')
    out=gr.Label(num_top_classes=4,label='Classe prevista e probabilidades')
    gr.Button('Classificar').click(classify,inp,out)
if __name__=='__main__': app.launch()
