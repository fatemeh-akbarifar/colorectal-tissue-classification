"""Bounded PathMNIST comparison. Run from repository root; downloads are explicit."""
from pathlib import Path
import argparse,hashlib,json,time,random,platform,warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
from skimage.filters import sobel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score,confusion_matrix,recall_score,log_loss
from sklearn.exceptions import ConvergenceWarning
import joblib
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
LABELS=['Adipose','Background','Debris','Lymphocytes','Mucus','Smooth muscle','Normal mucosa','Stroma','Tumour epithelium']
URL='https://zenodo.org/records/10519652/files/pathmnist_64.npz?download=1'
MD5='55aa9c1e0525abe5a6b9d8343a507616'

def digest(path,algorithm):
 h=hashlib.new(algorithm)
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def features(images):
 rows=[]
 for im in images:
  x=im.astype(float)/255;gray=img_as_ubyte(rgb2gray(x))
  lbp=local_binary_pattern(gray,8,1,method='uniform')
  hist=np.bincount(lbp.astype(int).ravel(),minlength=10)/gray.size
  rows.append(np.r_[x.mean((0,1)),x.std((0,1)),hist,shannon_entropy(gray),sobel(gray.astype(float)/255).mean()])
 return np.asarray(rows)

class Patches(Dataset):
 def __init__(self,x,y):self.x=x;self.y=y
 def __len__(self):return len(self.y)
 def __getitem__(self,i):return torch.from_numpy(self.x[i].copy()).permute(2,0,1).float()/255,int(self.y[i])

class TinyCNN(nn.Module):
 def __init__(self):
  super().__init__();self.layers=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,9))
 def forward(self,x):return self.layers(x)

def probabilities(model,x,y):
 model.eval();outputs=[]
 with torch.no_grad():
  for xb,_ in DataLoader(Patches(x,y),batch_size=128,shuffle=False):outputs.append(model(xb).softmax(1).numpy())
 return np.concatenate(outputs)

def main(download=False):
 OUT.mkdir(exist_ok=True);(ROOT/'data').mkdir(exist_ok=True)
 path=ROOT/'data/pathmnist_64.npz'
 if not path.exists():
  if not download:raise FileNotFoundError('Run python analysis.py --download to acquire PathMNIST')
  import urllib.request
  urllib.request.urlretrieve(URL,path)
 if digest(path,'md5')!=MD5:raise ValueError('Dataset checksum mismatch')
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 archive=np.load(path);images={s:archive[s+'_images'] for s in ('train','val','test')};labels={s:archive[s+'_labels'].ravel() for s in images}
 for s,n in [('train',89996),('val',10004),('test',7180)]:
  assert images[s].shape==(n,64,64,3) and images[s].dtype==np.uint8
  assert set(np.unique(labels[s]))==set(range(9))
 hashes={s:[hashlib.sha256(im.tobytes()).hexdigest() for im in images[s]] for s in images}
 testhash=set(hashes['test']);trainhash=set(hashes['train']);indices={};exclusions=[];rng=np.random.default_rng(42)
 for s in ('train','val'):
  seen=set();valid=[]
  for i,h in enumerate(hashes[s]):
   reason='matches_test' if h in testhash else ('matches_training' if s=='val' and h in trainhash else ('within_split_duplicate' if h in seen else None))
   seen.add(h)
   if reason:exclusions.append({'split':s,'index':i,'reason':reason})
   else:valid.append(i)
  valid=np.array(valid);chosen=[]
  for c in range(9):
   candidates=valid[labels[s][valid]==c];target=1000 if s=='train' else min(200,len(candidates))
   if len(candidates)<target:raise RuntimeError('Insufficient training coverage after duplicate audit')
   chosen.extend(rng.choice(candidates,target,replace=False))
  indices[s]=np.array(sorted(chosen))
 indices['test']=np.arange(len(labels['test']))
 pd.DataFrame(exclusions,columns=['split','index','reason']).to_csv(OUT/'duplicate-exclusions.csv',index=False)
 pd.DataFrame([{'split':s,'original_index':int(i),'label':int(labels[s][i])} for s in indices for i in indices[s]]).to_csv(OUT/'split-manifest.csv',index=False)
 x={s:images[s][indices[s]] for s in indices};y={s:labels[s][indices[s]] for s in indices};del images,archive
 atlas_rng=np.random.default_rng(123);fig,axes=plt.subplots(9,4,figsize=(6,12));atlas=[]
 for c in range(9):
  chosen=atlas_rng.choice(np.flatnonzero(y['train']==c),4,replace=False)
  for j,i in enumerate(chosen):
   axes[c,j].imshow(x['train'][i]);axes[c,j].set_xticks([]);axes[c,j].set_yticks([])
   if j==0:axes[c,j].set_ylabel(LABELS[c],fontsize=9)
   atlas.append({'label':c,'original_index':int(indices['train'][i])})
 fig.suptitle('Nine tissue categories · PathMNIST 64',fontsize=13);fig.text(.5,.005,'Images: PathMNIST / Kather et al. · CC BY 4.0',ha='center',fontsize=8);fig.tight_layout(rect=(0,.02,1,.98));fig.savefig(OUT/'tissue-atlas.png',dpi=160);plt.close(fig)
 pd.DataFrame(atlas).to_csv(OUT/'atlas-indices.csv',index=False)
 f={s:features(x[s]) for s in ('train','val')}
 dummy=DummyClassifier(strategy='most_frequent').fit(f['train'],y['train'])
 feature_model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=42))
 with warnings.catch_warnings():
  warnings.simplefilter('error',ConvergenceWarning);feature_model.fit(f['train'],y['train'])
 joblib.dump(feature_model,OUT/'feature-model.joblib')
 validation=[]
 for name,model in [('dummy',dummy),('features',feature_model)]:
  pr=model.predict_proba(f['val']);validation.append({'model':name,'macro_f1':f1_score(y['val'],pr.argmax(1),average='macro',zero_division=0)})
 model=TinyCNN();optimizer=torch.optim.Adam(model.parameters(),lr=.001);criterion=nn.CrossEntropyLoss()
 loader=DataLoader(Patches(x['train'],y['train']),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(42),num_workers=0)
 start=time.monotonic();best=-1;history=[]
 before=next(model.parameters()).detach().clone()
 for epoch in range(1,9):
  model.train();total=0
  for xb,yb in loader:
   optimizer.zero_grad();loss=criterion(model(xb),yb)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
   loss.backward();optimizer.step();total+=loss.item()*len(yb)
  pr=probabilities(model,x['val'],y['val']);score=f1_score(y['val'],pr.argmax(1),average='macro',zero_division=0)
  history.append({'epoch':epoch,'train_loss':total/len(y['train']),'validation_loss':log_loss(y['val'],pr,labels=range(9)),'validation_macro_f1':score,'elapsed_seconds':time.monotonic()-start})
  if score>best:best=score;best_epoch=epoch;torch.save(model.state_dict(),OUT/'cnn.pt')
  print(json.dumps(history[-1]),flush=True)
  if time.monotonic()-start>=3600:break
 assert not torch.equal(before,next(model.parameters()).detach())
 pd.DataFrame(history).to_csv(OUT/'training-history.csv',index=False)
 model.load_state_dict(torch.load(OUT/'cnn.pt',map_location='cpu',weights_only=True));validation.append({'model':'cnn','macro_f1':best})
 pd.DataFrame(validation).to_csv(OUT/'validation-metrics.csv',index=False)
 # All model choices are frozen at this point. Only now compute test predictions.
 f['test']=features(x['test'])
 probs={'dummy':dummy.predict_proba(f['test']),'features':feature_model.predict_proba(f['test']),'cnn':probabilities(model,x['test'],y['test'])}
 metrics=[];predrows=[];confusions={}
 for name,pr in probs.items():
  pred=pr.argmax(1);cm=confusion_matrix(y['test'],pred,labels=range(9));confusions[name]=cm
  metrics.append({'model':name,'macro_f1':f1_score(y['test'],pred,average='macro',zero_division=0),'balanced_accuracy':balanced_accuracy_score(y['test'],pred),'accuracy':accuracy_score(y['test'],pred)})
  pd.DataFrame(cm,index=LABELS,columns=LABELS).to_csv(OUT/(name+'-confusion.csv'))
  pd.DataFrame({'label':range(9),'class':LABELS,'count':np.bincount(y['test'],minlength=9),'recall':recall_score(y['test'],pred,labels=range(9),average=None,zero_division=0)}).to_csv(OUT/(name+'-recall.csv'),index=False)
  for i in range(len(pred)):predrows.append({'model':name,'test_index':i,'true_label':int(y['test'][i]),'prediction':int(pred[i]),**{'p'+str(c):float(pr[i,c]) for c in range(9)}})
 pd.DataFrame(metrics).to_csv(OUT/'test-metrics.csv',index=False);pd.DataFrame(predrows).to_csv(OUT/'test-predictions.csv',index=False)
 fig,axes=plt.subplots(1,3,figsize=(14,4.5))
 for ax,(name,cm) in zip(axes,confusions.items()):
  norm=cm/cm.sum(axis=1,keepdims=True);ax.imshow(norm,vmin=0,vmax=1,cmap='Purples')
  ax.set(xticks=range(9),yticks=range(9),xlabel='Predicted class',title=name)
  for a in range(9):
   for b in range(9):ax.text(b,a,f'{norm[a,b]:.1f}',ha='center',va='center',fontsize=7,color='white' if norm[a,b]>.5 else 'black')
 axes[0].set_ylabel('Reference class');fig.suptitle('Test recall and confusion by tissue class');fig.tight_layout();fig.savefig(OUT/'confusion-comparison.png',dpi=170);plt.close(fig)
 cm=confusions['cnn'];pairs=sorted([(int(cm[a,b]),a,b) for a in range(9) for b in range(9) if a!=b],key=lambda z:(-z[0],z[1],z[2]))[:2]
 errors=[];rng=np.random.default_rng(123);pred=probs['cnn'].argmax(1)
 fig,axes=plt.subplots(2,12,figsize=(15,4))
 for row,(_,a,b) in enumerate(pairs):
  candidates=np.flatnonzero((y['test']==a)&(pred==b));wrong=rng.choice(candidates,min(6,len(candidates)),replace=False)
  groups=[('error',wrong)]
  for c in (a,b):
   good=np.flatnonzero((y['test']==c)&(pred==c));groups.append(('correct',rng.choice(good,min(3,len(good)),replace=False)))
  for ax in axes[row]:ax.axis('off')
  col=0
  for kind,ids in groups:
   for i in ids:
    axes[row,col].imshow(x['test'][i]);axes[row,col].set_title(f'{y["test"][i]} → {pred[i]}',fontsize=8)
    errors.append({'kind':kind,'test_index':int(i),'true_label':int(y['test'][i]),'predicted_label':int(pred[i])});col+=1
 axes[0,0].text(0,1.5,'Reference → prediction; first six per row are errors',transform=axes[0,0].transAxes,fontsize=12)
 fig.text(.5,.02,'PathMNIST / Kather et al. · CC BY 4.0 · 64-pixel patches; no clinical relabelling',ha='center',fontsize=9);fig.tight_layout(rect=(0,.08,1,.93));fig.savefig(OUT/'error-atlas.png',dpi=170);plt.close(fig)
 pd.DataFrame(errors).to_csv(OUT/'error-indices.csv',index=False)
 opposite=np.flatnonzero((probs['features'].argmax(1)==y['test'])&(pred!=y['test']))
 if len(opposite):
  i=int(opposite[0]);fig,ax=plt.subplots(figsize=(3,3));ax.imshow(x['test'][i]);ax.axis('off');ax.set_title(f'Test {i}: label {y["test"][i]}, CNN {pred[i]}\nFeature model correct');fig.tight_layout();fig.savefig(OUT/'feature-wins-example.png',dpi=150);plt.close(fig)
 manifest={'dataset_url':URL,'publisher_md5':MD5,'sha256':digest(path,'sha256'),'licence':'CC BY 4.0','resolution':64,'train_count':len(y['train']),'validation_count':len(y['val']),'test_count':len(y['test']),'excluded_train_val_images':len(exclusions),'test_duplicate_count':len(hashes['test'])-len(testhash),'selected_epoch':best_epoch,'epochs_run':len(history),'seed':42,'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'source_sha256':digest(ROOT/'analysis.py','sha256'),'test_use':'Once after model freeze','top_confusions':[{'count':n,'true':LABELS[a],'predicted':LABELS[b]} for n,a,b in pairs],'feature_correct_cnn_wrong_count':len(opposite)}
 (OUT/'run.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(metrics,indent=2));return pd.DataFrame(metrics)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');main(parser.parse_args().download)
