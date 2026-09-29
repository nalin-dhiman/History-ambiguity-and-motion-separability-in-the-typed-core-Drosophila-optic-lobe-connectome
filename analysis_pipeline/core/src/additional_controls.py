from pathlib import Path
import numpy as np,pandas as pd,json
from scipy.stats import spearmanr
from decomposition import decompose,matrix_metrics
root=Path.cwd();out=root/'analysis_pipeline/core/results/development';t=pd.read_csv(out/'type_decomposition.csv');supported=set(t.loc[(t.bodies>=20)&(t.weight>=1000),'dst_type']);comparisons=[]
for name in ['clear_labels','traced','roughly_traced','equal_body','threshold_3','threshold_5','threshold_10']:
 q=pd.read_csv(out/f'type_{name}.csv');m=t[t.dst_type.isin(supported)].merge(q,on='dst_type',suffixes=('_full','_sensitivity'))
 comparisons.append(dict(sensitivity=name,shared_types=len(m),spearman_R=float(spearmanr(m.R_full,m.R_sensitivity).statistic),median_delta_R=float(np.median(m.R_sensitivity-m.R_full))))
pd.DataFrame(comparisons).to_csv(out/'sensitivity_rank_checks.csv',index=False)
# Within-type selected-input fragmentation, deterministic algebraic counterexample.
fixtures=[]
for label,m in [('same_margins_segregated',[[1,0],[0,1]]),('same_margins_mixed',[[.5,.5],[.5,.5]]),('one_intact_body',[[2],[2]]),('input_selective_split',[[2,0],[0,2]])]:fixtures.append(dict(fixture=label,**matrix_metrics(m)))
pd.DataFrame(fixtures).to_csv(out/'exact_fixtures.csv',index=False)
# At fixed type channel, heterogeneity and a common-mode/contrast task need not be ordered together.
# Source activity X_c=(1,0) or (0,1); identity vs averaging gives perfect discrimination vs none.
# X_m=(1,1) or (0,0); both preserve discrimination. After losing the first target, identity
# fails a task carried only by source 1 while the averaging circuit retains a difference.
# This is an analytical demonstration about readout/lesion dependence, not a physiological simulation.
(out/'task_counterexamples.json').write_text(json.dumps({'high_R':{'W':[[1,0],[0,1]],'A':1,'C':0,'R':1},'low_R':{'W':[[0.5,0.5],[0.5,0.5]],'A':1,'C':1,'R':0},'contrast_task':{'inputs':[[1,0],[0,1]],'intact_high_R_separable':True,'intact_low_R_separable':False},'localized_task_target1_removed':{'inputs':[[1,0],[0,0]],'high_R_remaining_target_separable':False,'low_R_remaining_target_separable':True},'message':'The same structural ordering reverses functional preference when task and perturbation change.'},indent=2))
print(pd.DataFrame(comparisons).to_string(index=False))
