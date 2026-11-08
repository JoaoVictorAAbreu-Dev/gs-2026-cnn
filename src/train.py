"""Experimentos EuroSAT: seleção exclusivamente por validação e teste final único."""
import json, os, random, zipfile, urllib.request
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
SEED=42
random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
try: tf.config.experimental.enable_op_determinism()
except (AttributeError, RuntimeError): pass
CLASSES=['Forest','Industrial','Residential','River']
ROOT=Path(os.environ.get('CNN_ROOT',Path.cwd())).resolve()
DATA=ROOT/'data'; OUT=ROOT/'outputs'; DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
archive=DATA/'EuroSAT.zip'
if not archive.exists():
    urllib.request.urlretrieve('https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1', archive)
folder=DATA/'2750'
if not folder.exists() and not (DATA/'EuroSAT_RGB').exists():
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            p=Path(info.filename)
            if p.is_absolute() or '..' in p.parts: raise ValueError('Caminho inseguro no arquivo zip')
        z.extractall(DATA)
if not folder.exists(): folder=DATA/'EuroSAT_RGB'
paths=[]; labels=[]
for label,name in enumerate(CLASSES):
    files=sorted((folder/name).glob('*.jpg'))
    if len(files)<200: raise RuntimeError(f'{name}: esperadas ao menos 200 imagens, encontradas {len(files)}')
    chosen=np.random.default_rng(SEED+label).choice(files,200,replace=False)
    paths.extend(map(str,chosen)); labels.extend([label]*200)
paths=np.array(paths); labels=np.array(labels)
trainval, test=train_test_split(np.arange(len(labels)),test_size=.2,stratify=labels,random_state=SEED)
train, val=train_test_split(trainval,test_size=.25,stratify=labels[trainval],random_state=SEED)
assert not (set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
# Converter uma vez mantém as mesmas imagens e normalização para os três experimentos.
def load(ids):
    imgs=[tf.keras.utils.img_to_array(tf.keras.utils.load_img(paths[i],target_size=(64,64)))/255.0 for i in ids]
    return np.asarray(imgs,dtype='float32'),labels[ids]
xtr,ytr=load(train); xv,yv=load(val); xt,yt=load(test)
(OUT/'split.json').write_text(json.dumps({k:[str(Path(paths[i]).relative_to(DATA)) for i in ids] for k,ids in [('train',train),('val',val),('test',test)]},indent=2))
# Experimentos 1–3: primeira rodada. Experimentos 4–6 mudam UMA coisa por vez em relação à CNN 2.
configs=[
    {'name':'cnn_1','filters':[16,32],'dense':64,'dropout':0.0,'lr':0.001,'batchnorm':False,'augment':False,'epochs':12},
    {'name':'cnn_2','filters':[32,64],'dense':128,'dropout':0.25,'lr':0.001,'batchnorm':False,'augment':False,'epochs':12},
    {'name':'cnn_3','filters':[32,64,128],'dense':128,'dropout':0.35,'lr':0.0005,'batchnorm':False,'augment':False,'epochs':12},
    {'name':'cnn_4','filters':[32,64],'dense':128,'dropout':0.25,'lr':0.001,'batchnorm':True,'augment':False,'epochs':12},
    {'name':'cnn_5','filters':[32,64],'dense':128,'dropout':0.25,'lr':0.001,'batchnorm':False,'augment':True,'epochs':12},
    {'name':'cnn_6','filters':[32,64],'dense':128,'dropout':0.25,'lr':0.001,'batchnorm':False,'augment':False,'epochs':30},
]
def make_model(c):
    layers=[tf.keras.layers.Input((64,64,3))]
    if c['augment']:
        # Imagens de satélite não têm "lado de cima": espelhar e girar gera exemplos válidos (só no treino).
        layers += [tf.keras.layers.RandomFlip('horizontal_and_vertical'),tf.keras.layers.RandomRotation(0.25)]
    for filters in c['filters']:
        layers.append(tf.keras.layers.Conv2D(filters,3,padding='same',use_bias=not c['batchnorm']))
        if c['batchnorm']: layers.append(tf.keras.layers.BatchNormalization())
        layers += [tf.keras.layers.Activation('relu'),tf.keras.layers.MaxPooling2D(2)]
    layers += [tf.keras.layers.Flatten(),tf.keras.layers.Dense(c['dense'],activation='relu')]
    if c['dropout']: layers.append(tf.keras.layers.Dropout(c['dropout']))
    layers.append(tf.keras.layers.Dense(4,activation='softmax'))
    model=tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(c['lr']),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model
results=[]
for c in configs:
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED)
    model=make_model(c)
    callback=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3 if c['epochs']<=12 else 5,restore_best_weights=True)
    history=model.fit(xtr,ytr,validation_data=(xv,yv),epochs=c['epochs'],batch_size=32,callbacks=[callback],verbose=2)
    val_prob=model.predict(xv,verbose=0); val_pred=val_prob.argmax(axis=1)
    model.save(OUT/(c['name']+'.keras'))
    item={'config':c,'epochs_run':len(history.history['loss']),'val_accuracy':float(accuracy_score(yv,val_pred)),
          'val_report':classification_report(yv,val_pred,target_names=CLASSES,output_dict=True,zero_division=0),
          'val_confusion_matrix':confusion_matrix(yv,val_pred,labels=range(4)).tolist(),
          'history':{k:[float(x) for x in v] for k,v in history.history.items()},'parameters':model.count_params()}
    results.append(item)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key in zip(axes,['loss','accuracy']):
        ax.plot(history.history[key],label='treino'); ax.plot(history.history['val_'+key],label='validação'); ax.set_title(key); ax.legend(); ax.set_xlabel('época')
    fig.tight_layout(); fig.savefig(OUT/(c['name']+'_curvas.png'),dpi=150); plt.close(fig)
# Acurácia de validação decide; empate por menor loss de validação.
best=max(results,key=lambda x:(x['val_accuracy'],-min(x['history']['val_loss'])))
model=tf.keras.models.load_model(OUT/(best['config']['name']+'.keras'))
prob=model.predict(xt,verbose=0); pred=prob.argmax(axis=1)
summary={'classes':CLASSES,'seed':SEED,'counts':{'train':len(train),'validation':len(val),'test':len(test)},
         'selection':'maior acurácia de validação; desempate por menor perda de validação',
         'experiments':results,'best':best['config']['name'],
         'test_report':classification_report(yt,pred,target_names=CLASSES,output_dict=True,zero_division=0),
         'test_confusion_matrix':confusion_matrix(yt,pred,labels=range(4)).tolist()}
(OUT/'results.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
model.save(OUT/'best.keras')
fig,ax=plt.subplots(figsize=(6,5)); cm=np.array(summary['test_confusion_matrix']); im=ax.imshow(cm,cmap='Blues'); fig.colorbar(im,ax=ax)
ax.set(xticks=range(4),yticks=range(4),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Previsto',ylabel='Real',title='Teste: matriz de confusão')
plt.setp(ax.get_xticklabels(),rotation=30,ha='right')
for i in range(4):
    for j in range(4): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
fig.tight_layout(); fig.savefig(OUT/'matriz_teste.png',dpi=150); plt.close(fig)
print('Modelo selecionado:',summary['best']); print(classification_report(yt,pred,target_names=CLASSES,zero_division=0))
