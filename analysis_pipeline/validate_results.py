"""Check generated numerical identities and saved summaries without dataset-specific targets."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
PIPE=Path(__file__).resolve().parent

def validate(base=PIPE):
    checks=[];stages=[]
    def check(label,ok):
        if not bool(ok):raise AssertionError(label)
        checks.append(label)
    def close(a,b):return np.allclose(a,b,atol=1e-10,rtol=1e-10,equal_nan=True)
    core=base/'core/results/development';u=pd.read_csv(core/'universe_decomposition.csv');t=pd.read_csv(core/'type_decomposition.csv');body=pd.read_parquet(core/'body_decomposition.parquet');inputs=pd.read_parquet(core/'source_type_body_inputs.parquet')
    check('Core universe identities',close(u.A,u.C+u.R));check('Core type identities',close(t.A,t.C+t.R));full=u[u.universe=='full'].iloc[0]
    check('Core input, body and type mass',close([inputs.weight.sum(),body.weight.sum(),t.weight.sum()],[full.weight]*3))
    for k in ['A','C','R']:check('Core weighted '+k,close(np.average(t[k],weights=t.weight),full[k]))
    check('Core denominators',len(body)==full.target_bodies and len(t)==full.target_types)
    if (core/'fixed_margin_summary.csv').exists():
        settings=json.loads((base/'core/configs/gate1.json').read_text());s=pd.read_csv(core/'fixed_margin_summary.csv').set_index('dst_type');d=pd.read_csv(core/'fixed_margin_draws.csv.gz')
        eligible=t[(t.bodies>=settings['support_rule']['minimum_target_bodies'])&(t.weight>=settings['support_rule']['minimum_retained_contacts'])]
        check('Core reference support',set(s.index)==set(eligible.dst_type)==set(d.dst_type))
        for typ,q in d.groupby('dst_type'):
            row=s.loc[typ];check('Core reference '+typ,len(q)==settings['fixed_margin_null']['draws'] and close(q.R.mean(),row.null_R_mean) and close(q.R.quantile(.975),row.null_R_q975))
        rare=pd.read_csv(core/'count_matched_draws.csv.gz');check('Core rarefaction identities',close(rare.A,rare.C+rare.R))
        check('Core rarefaction draw counts',rare.groupby('dst_type').size().eq(settings['input_count_matching']['draws']).all())
        sample=pd.read_csv(core/'sampling_sensitivity.csv');setting=json.loads((base/'core/configs/sampling_protocol.json').read_text());check('Sampling identities',close(sample.A,sample.C+sample.R))
        check('Sampling design',len(sample)==len(setting['fractions'])*len(setting['methods'])*setting['draws'] and sample.groupby(['fraction','method']).size().eq(setting['draws']).all())
    stages.append('core');r=base/'cross_specimen/results';cfg=json.loads((base/'cross_specimen/config.json').read_text())
    if (r/'replication_global.csv').exists():
        g=pd.read_csv(r/'replication_global.csv');check('Replication global identities',close(g.A,g.C+g.R))
        for row in g.itertuples():
            if row.universe=='male_original_audited':continue
            x=pd.read_csv(r/(row.universe+'_types.csv'));check(row.universe+' identities and mass',close(x.A,x.C+x.R) and close(x.weight.sum(),row.weight))
            for k in ['A','C','R']:check(row.universe+' weighted '+k,close(np.average(x[k],weights=x.weight),getattr(row,k)))
        alph=json.loads((r/'matched_primary_alphabets.json').read_text())
        for name in ['male_matched_right','female_matched_right']:
            x=pd.read_parquet(r/(name+'_inputs.parquet'));check(name+' common source roles',set(x.src_type)==set(alph['source_types']));check(name+' common target roles',set(x.dst_type)==set(alph['target_types']))
        for row in pd.read_csv(r/'type_agreement.csv').itertuples():
            p=pd.read_csv(r/(row.comparison+'_paired_types.csv'));a=p[row.metric+'_a'];b=p[row.metric+'_b'];valid=a.notna()&b.notna();a=a[valid];b=b[valid]
            check(row.comparison+' '+row.metric+' agreement',len(a)==row.types and close(spearmanr(a,b).statistic,row.spearman) and close(abs(a-b).mean(),row.mean_absolute_difference))
        stages.append('replication')
        for path in r.glob('*_fixed_margin_summary.csv'):
            name=path.name.removesuffix('_fixed_margin_summary.csv');s=pd.read_csv(path).set_index('dst_type');d=pd.read_csv(r/(name+'_fixed_margin_draws.csv.gz'))
            check(name+' reference type coverage',set(s.index)==set(d.dst_type))
            for typ,q in d.groupby('dst_type'):
                row=s.loc[typ];check(name+' '+typ+' reference summary',len(q)==cfg['fixed_margin_draws'] and close(q.R.mean(),row.null_R_mean) and close(q.R.quantile(.975),row.null_R_q975))
        if (r/'spatial_type_decomposition.csv').exists():
            sp=pd.read_csv(r/'spatial_type_decomposition.csv');sg=pd.read_csv(r/'spatial_global.csv');check('All spatial identities',close(sp.R,sp.R_spatial+sp.R_residual) and close(sp.A,sp.C+sp.R))
            for row in sg.itertuples():
                q=sp[(sp.dataset==row.dataset)&(sp.width==row.width)&(sp['shift']==row.shift)]
                check(f'{row.dataset} {row.width}/{row.shift} aggregation',len(q)==row.types and q.weight.sum()==row.weight and all(close(np.average(q[k],weights=q.weight),getattr(row,k)) for k in ['A','C','R','R_spatial','R_residual']))
            bc=pd.read_parquet(r/'spatial_prediction_bodies.parquet');f=pd.read_csv(r/'spatial_prediction_folds.csv');check('Each evaluated body tested once',not bc.duplicated(['dataset','dst']).any());check('Per-body loss identities',close(bc.gain_bits,bc.baseline_ce_bits-bc.spatial_ce_bits))
            for row in f[f.status=='evaluated'].itertuples():
                q=bc[(bc.dataset==row.dataset)&(bc.dst_type==row.dst_type)&(bc.fold==row.fold)]
                check(f'{row.dataset} {row.dst_type} fold {row.fold} losses',len(q)==row.test_bodies and q.contacts.sum()==row.test_contacts and all(close(np.average(q[k],weights=q.contacts),getattr(row,k)) for k in ['baseline_ce_bits','spatial_ce_bits','gain_bits']))
            if (r/'spatial_reference_summary.csv').exists():
                s=pd.read_csv(r/'spatial_reference_summary.csv').set_index(['dataset','dst_type']);d=pd.read_csv(r/'spatial_reference_draws.csv.gz')
                for key,q in d.groupby(['dataset','dst_type']):
                    row=s.loc[key];check('Spatial reference '+str(key),len(q)==cfg['fixed_margin_draws'] and close(q.R_residual.mean(),row.reference_mean) and close(q.R_residual.quantile(.975),row.reference_q975))
            stages.append('spatial')
    out=base/'runs/validation.json';out.parent.mkdir(parents=True,exist_ok=True);report={'status':'PASS','stages_present':stages,'checks':checks,'count':len(checks)};out.write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
    report=validate();print('PASS',report['count'],'checks for',', '.join(report['stages_present']))
