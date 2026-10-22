"""TRAIN-only enemy-motion auxiliary gradient for the existing PPO actor.

The decoder is training-only. Its own optimizer sees detached trunk features;
its parameters are frozen during guarded actor proposals.
"""
import math
import pathlib

from evaluate_registered_combat_models import read,sha


class FutureAuxiliary:
    def __init__(self,actor,prepared,data,decoder,weight,behavior_sha,previous=None):
        import numpy as np
        import torch
        self.torch=torch
        assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
        assert math.isfinite(weight) and weight>=0
        self.weight=weight;self.pins={};self.steps=0;self.last_grad_norm=None
        def pin(path,expected=None):
            path=pathlib.Path(path).resolve();h=sha(path);assert expected is None or expected==h,str(path)
            self.pins[str(path)]=h;return path
        data=pathlib.Path(data);decoder=pathlib.Path(decoder)
        meta=read(pin(data/'complete.json'));assert meta['state']=='complete' and meta['version']=='combat_observed_future_dataset_v1'
        assert meta['current_features_width']==1147 and meta['dt_game_seconds']==.1 and meta['displacement_scale']==64
        for item in meta['inputs']:pin(item['path'],item['sha256'])
        closed=read(pin(decoder/'complete.json'));assert closed['state']=='complete' and closed['device']=='cuda'
        p=read(pin(decoder/'protocol.json',closed['protocol_sha256']))
        assert p['version']=='combat_frozen_actor_future_readout_v1' and p['seed']==20261022
        assert p.get('normalization','none')=='none' and p['outputs']==35 and not p['selection_on_development']
        self.spec=dict(version='combat_enemy_future_auxiliary_v1',weight=weight,known_velocity_only=True,
            initial_decoder_sha256=sha(decoder/'complete.json'),initial_decoder_source_sha256=p['source_weights_sha256'],
            dataset_sha256=sha(data/'complete.json'),head_learning_rate=.001,head_weight_decay=.0001,
            loss='masked enemy-only SmoothL1 beta0.1 displacement/64 residual to observed CV',
            source_behavior_sha256=behavior_sha,initialization='fixed seed22 unnormalized combined readout')
        if previous is None:assert p['source_weights_sha256']==behavior_sha,'Fresh decoder initialization requires matching source actor'
        else:
            for key in ('version','weight','known_velocity_only','initial_decoder_sha256','head_learning_rate','head_weight_decay','loss'):
                assert previous['spec'][key]==self.spec[key],('Explicit future objective fork required',key)
        with np.load(pin(meta['splits']['train']['path'],meta['splits']['train']['sha256']),allow_pickle=False) as arrays:
            keys=list(zip(arrays['seed'].tolist(),arrays['frame'].tolist()))
            features=torch.as_tensor(arrays['x'][:,:1147],device='cuda')
            self.action=torch.as_tensor(arrays['x'][:,-6:],device='cuda')
            self.cv=torch.as_tensor(arrays['cv'][:,1:],device='cuda')
            self.target=torch.as_tensor(arrays['target'][:,1:],device='cuda')
            self.mask=torch.as_tensor(arrays['motion_mask'][:,1:]&arrays['velocity_known'][:,1:],device='cuda')
        assert len(keys)==len(set(keys)) and bool(self.mask.any())
        self.x=prepared[0];locations={}
        for b,segment in enumerate(prepared[3]):
            for t,c in enumerate(segment):
                key=(c['seed'],c['frame']);assert key not in locations;locations[key]=(b,t)
        assert set(keys)<=set(locations),'Auxiliary data must belong to current full TRAIN histories'
        self.ix=torch.tensor([locations[k] for k in keys],device='cuda',dtype=torch.long)
        assert torch.equal(self.x[self.ix[:,0],self.ix[:,1]],features),'Canonical feature mismatch'
        assert actor.base.mha.embed_dim==64 and actor.base.window==32
        q=torch.arange(self.x.shape[1],device='cuda');self.attention_mask=(q[None,:]>q[:,None])|(q[None,:]<q[:,None]-31)
        # Initial decoder construction must not perturb restored PPO RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            self.decoder=torch.nn.Sequential(torch.nn.Linear(134,128,device='cuda'),torch.nn.ReLU(),
                torch.nn.Linear(128,128,device='cuda'),torch.nn.ReLU(),torch.nn.Linear(128,35,device='cuda'))
        initial=torch.load(pin(decoder/'checkpoint.pt',closed['checkpoint_sha256']),map_location='cuda',weights_only=True)
        self.decoder.load_state_dict(initial['models']['combined'])
        self.optimizer=torch.optim.AdamW(self.decoder.parameters(),lr=.001,weight_decay=.0001)
        if previous is not None:
            self.decoder.load_state_dict(previous['decoder']);self.optimizer.load_state_dict(previous['optimizer']);self.steps=previous['decoder_steps']
        self._freeze()
        self.spec.update(rows=len(keys),enemy_labels=int(self.mask.sum()),development_fitted=False)

    def _freeze(self):
        for p in self.decoder.parameters():p.requires_grad_(False)

    def inputs(self,actor):
        torch=self.torch;core=actor.base
        encoded=core.encoder(self.x);_,tokens=core(self.x)
        attended,_=core.mha(tokens,tokens,tokens,attn_mask=self.attention_mask,need_weights=False)
        i=self.ix
        return torch.cat((encoded[i[:,0],i[:,1]],attended[i[:,0],i[:,1]],self.action),-1)

    def prediction_loss(self,inputs):
        prediction=self.cv+self.decoder(inputs)[:,:27].reshape(-1,9,3)[:,1:]
        return self.torch.nn.functional.smooth_l1_loss(prediction[self.mask],self.target[self.mask],beta=.1)

    def actor_loss(self,actor):
        if self.weight==0:return self.x.new_zeros(())
        return self.weight*self.prediction_loss(self.inputs(actor))

    def head_step(self,actor):
        torch=self.torch
        with torch.no_grad():inputs=self.inputs(actor).detach()
        for p in self.decoder.parameters():p.requires_grad_(True)
        try:
            self.optimizer.zero_grad();loss=self.prediction_loss(inputs);assert bool(torch.isfinite(loss))
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.decoder.parameters(),1.,error_if_nonfinite=True)
            self.optimizer.step();self.last_grad_norm=float(norm);self.steps+=1
        finally:self._freeze()

    def stats(self,actor):
        torch=self.torch
        with torch.no_grad():
            inputs=self.inputs(actor);prediction=self.cv+self.decoder(inputs)[:,:27].reshape(-1,9,3)[:,1:]
            return dict(enemy_motion_loss=float(self.prediction_loss(inputs)),
                enemy_axis_rmse=float(((prediction-self.target)[self.mask]*64).square().mean().sqrt()),
                decoder_steps=self.steps,last_decoder_grad_norm=self.last_grad_norm)

    def checkpoint(self):
        return dict(spec=self.spec,decoder=self.decoder.state_dict(),optimizer=self.optimizer.state_dict(),decoder_steps=self.steps)

    def verify(self):
        for path,h in self.pins.items():assert sha(path)==h,path
