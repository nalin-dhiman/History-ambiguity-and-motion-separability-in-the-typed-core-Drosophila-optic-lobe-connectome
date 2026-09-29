from pathlib import Path
from types import SimpleNamespace
import sys,json,time,hashlib,warnings,argparse
import numpy as np,pandas as pd
from scipy import sparse
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score,log_loss,confusion_matrix
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','confirmation'],default='development');args=p.parse_args();root=Path.cwd();base=root/'analysis_pipeline/core';out=base/'results'/args.phase;out.mkdir(exist_ok=True,parents=True);cache=root/'analysis_pipeline/diagnostics';sys.path.insert(0,str(cache))
from diagnostic_helpers.synthetic_tasks import generate_task_data
from diagnostic_helpers.plasticity_metrics import split_train_test
cfg=json.loads((base/'configs/probe_protocol.json').read_text());node=pd.read_csv(cache/'data/spatial_metadata.csv').set_index('bodyId');rows=[];cvrows=[];cms={};spectra=[];traces=[];predictions=[];hashes=[];t0=time.time()
def model(C,scale,maxiter=2000):
 lr=LogisticRegression(C=C,max_iter=maxiter,tol=cfg['tolerance'],solver='lbfgs')
 return make_pipeline(StandardScaler(),lr) if scale else lr
def fit(m,x,y):
 with warnings.catch_warnings(record=True) as ws:m.fit(x,y)
 return '; '.join(str(w.message) for w in ws)
def cal(y,p):
 conf=p.max(1);corr=p.argmax(1)==y;s=0
 for lo in np.arange(0,1,.1):
  ix=(conf>=lo)&(conf<(lo+.1+1e-12 if lo>.89 else lo+.1))
  if ix.any():s+=ix.mean()*abs(conf[ix].mean()-corr[ix].mean())
 return s
for sample in cfg['graphs']:
 e=pd.read_csv(cache/f'data/motion_edges_{sample}.csv.gz');src=pd.unique(e.src);dst=pd.unique(e.dst);si=pd.Index(src).get_indexer(e.src);di=pd.Index(dst).get_indexer(e.dst);rng=np.random.default_rng(cfg['graph_seed_base']+sample)
 def W(s,d,w):
  m=sparse.coo_matrix((w,(s,d)),shape=(len(src),len(dst))).tocsr();den=np.asarray(m.sum(0)).ravel();return m@sparse.diags(np.divide(1.,den,out=np.zeros_like(den,dtype=float),where=den>0))
 w=e.weight.to_numpy(float);real=W(si,di,w)
 mats={'real':real,'source_assignment_shuffle':real[rng.permutation(len(src))],'configuration':W(si,rng.permutation(di),w),'weight_shuffle':W(si,di,rng.permutation(w)),'uniform':W(si,di,np.ones(len(e)))}
 seed=(cfg['development_seed_base'] if args.phase=='development' else cfg['confirmation_seed_base'])+1000*sample
 g=SimpleNamespace(source_nodes=src,target_nodes=dst,source_types=node.loc[src,'type'].tolist());task=generate_task_data(g,None,'motion_direction',cfg['samples'],cfg['noise'],seed);X=task['X_source'];y=task['y'];tr,te,yt,ye=split_train_test(X,y,seed)
 hashes.append(dict(sample_id=sample,seed=seed,X_sha256=hashlib.sha256(X.tobytes()).hexdigest(),y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),train_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),test_sha256=hashlib.sha256(te.tobytes()).hexdigest()))
 for name in cfg['conditions']:
  a,b=(tr.astype(float),te.astype(float)) if name=='source' else (np.asarray(tr@mats[name]),np.asarray(te@mats[name]))
  if args.phase=='development':
   sv=np.linalg.svd(a-a.mean(0),compute_uv=False);ss=sv**2;ss/=ss.sum();erank=np.exp(-np.sum(ss[ss>0]*np.log(ss[ss>0])));rank=int((sv>sv[0]*max(a.shape)*np.finfo(float).eps).sum())
   spectra.extend(dict(sample_id=sample,condition=name,index=i,singular_value=float(v),numerical_rank=rank,effective_rank=erank) for i,v in enumerate(sv))
  cvs=[]
  for C in cfg['C_grid']:
   ls=[]
   for fold,(ii,jj) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed+1).split(a,yt)):
    m=model(C,True);ws=fit(m,a[ii],yt[ii]);loss=log_loss(yt[jj],m.predict_proba(a[jj]),labels=[0,1,2,3]);ls.append(loss);cvrows.append(dict(sample_id=sample,condition=name,C=C,fold=fold,loss=loss,warnings=ws))
   cvs.append(np.mean(ls))
  chosen=cfg['C_grid'][int(np.argmin(cvs))]
  for profile,C,scale in [('unscaled_C1',1,False),('scaled_C1',1,True),('scaled_trainCV',chosen,True)]:
   m=model(C,scale);ws=fit(m,a,yt);pp=m.predict_proba(b);pred=pp.argmax(1);lr=m[-1] if scale else m;key=f'{sample}_{name}_{profile}';cms[key]=confusion_matrix(ye,pred,labels=[0,1,2,3]).tolist()
   row=dict(phase=args.phase,sample_id=sample,condition=name,profile=profile,C=C,accuracy=accuracy_score(ye,pred),balanced_accuracy=balanced_accuracy_score(ye,pred),log_loss=log_loss(ye,pp,labels=[0,1,2,3]),brier=float(np.square(pp-np.eye(4)[ye]).sum(1).mean()),ece_10=cal(ye,pp),train_accuracy=m.score(a,yt),train_log_loss=log_loss(yt,m.predict_proba(a),labels=[0,1,2,3]),coef_norm=float(np.linalg.norm(lr.coef_)),iterations=int(lr.n_iter_.max()),warnings=ws,n_test=len(ye));rows.append(row)
   if profile=='scaled_trainCV':predictions.extend(dict(sample_id=sample,condition=name,test_index=i,y=int(ye[i]),**{f'p{k}':float(pp[i,k]) for k in range(4)}) for i in range(len(ye)))
  if args.phase=='development' and name in ['source','real']:
   for steps in [20,80,320,2000]:
    m=model(1,False,steps);ws=fit(m,a,yt);traces.append(dict(sample_id=sample,condition=name,max_iter=steps,actual_iter=int(m.n_iter_.max()),train_log_loss=log_loss(yt,m.predict_proba(a)),test_log_loss=log_loss(ye,m.predict_proba(b)),accuracy=m.score(b,ye),warnings=ws))
  print(args.phase,sample,name,'tuned accuracy',rows[-1]['accuracy'],'seconds',round(time.time()-t0,1),flush=True)
  pd.DataFrame(rows).to_csv(out/'probe_scores.csv',index=False)
 # Shuffled TRAINING labels as a negative control, with unchanged held-out true labels.
 m=model(1,True);ws=fit(m,np.asarray(tr@real),np.random.default_rng(seed+44).permutation(yt));pp=m.predict_proba(np.asarray(te@real));rows.append(dict(phase=args.phase,sample_id=sample,condition='training_label_shuffle',profile='scaled_C1',C=1,accuracy=accuracy_score(ye,pp.argmax(1)),balanced_accuracy=balanced_accuracy_score(ye,pp.argmax(1)),log_loss=log_loss(ye,pp),warnings=ws))
pd.DataFrame(rows).to_csv(out/'probe_scores.csv',index=False);pd.DataFrame(cvrows).to_csv(out/'probe_train_cv.csv',index=False);pd.DataFrame(predictions).to_csv(out/'probe_predictions.csv.gz',index=False);pd.DataFrame(spectra).to_csv(out/'probe_spectra.csv.gz',index=False);pd.DataFrame(traces).to_csv(out/'probe_convergence.csv',index=False);(out/'probe_confusions.json').write_text(json.dumps(cms,indent=2));(out/'probe_hashes.json').write_text(json.dumps(hashes,indent=2));(out/'probe_runtime.json').write_text(json.dumps({'elapsed_seconds':time.time()-t0,'phase':args.phase},indent=2))
