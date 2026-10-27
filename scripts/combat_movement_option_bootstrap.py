"""Recover uncalled decision values from observed next features and causal history."""
from ppo_combat import torch


def recover(rows,context,prepared,actor,options):
    present={(c['seed'],c['frame']) for c in context}
    wanted=[]
    for i,row in enumerate(rows):
        seed=row['seed']
        following=rows[i+1] if i+1<len(rows) else None
        if following and following['seed']==seed and following['frame']==row['next_frame']:continue
        event=row['sample']['movement_option']
        if row['bootstrap_zero'] or (seed,row['next_frame']) in present:continue
        if row['next_frame']!=event['started_frame']+5:continue
        assert row['next_frame']==row['frame']+1 and not row['next_reset']
        assert len(row['next_features'])==1554
        wanted.append(row)
    if not wanted:return {},dict(reconstructed=0)
    x,indices,inverse,segments=prepared
    endings={(segment[-1]['seed'],segment[-1]['frame']):(b,len(segment)) for b,segment in enumerate(segments)}
    extended=torch.zeros((x.shape[0],x.shape[1]+1,x.shape[2]),device='cuda',dtype=x.dtype)
    extended[:,:x.shape[1]]=x
    locations=[]
    for row in wanted:
        b,t=endings[(row['seed'],row['frame'])]
        extended[b,t]=torch.tensor(row['next_features'],device='cuda',dtype=x.dtype)
        locations.append((b,t))
    with torch.no_grad():
        original,_=actor(x);raw,_=actor(extended)
        # A causal append must not change any observed provider output.
        error=float((original[indices[:,0],indices[:,1]]-raw[indices[:,0],indices[:,1]]).abs().max())
        assert error<2e-5,('Causal bootstrap changed recorded prefix',error)
        b=torch.tensor([p[0] for p in locations],device='cuda',dtype=torch.long)
        t=torch.tensor([p[1] for p in locations],device='cuda',dtype=torch.long)
        _,values=options(extended[b,t],raw[b,t])
        assert bool(torch.isfinite(values).all())
    result={(row['seed'],row['next_frame']):float(value) for row,value in zip(wanted,values)}
    return result,dict(reconstructed=len(result),prefix_max_error=error,states=[
        dict(seed=row['seed'],frame=row['next_frame'],value=result[row['seed'],row['next_frame']],
            source='observed_next_features_and_frozen_causal_actor') for row in wanted])
