"""Observation-only v6 inventory tail, independent parity oracle for Go."""
VERSION='combat_features_v6'
PREFIX=814
MASK_OFFSET=833
WIDTH=845
WEAPONS=('', 'Blaster','Shotgun','Super Shotgun','Machinegun','Chaingun','Grenade Launcher','Rocket Launcher','HyperBlaster','Railgun','BFG10K','Grenades')
AMMO=('Bullets','Shells','Cells','Grenades','Rockets','Slugs')
SCALE=(200,100,200,50,50,50)
REQUIRES=('', '', 'Shells','Shells','Bullets','Bullets','Grenades','Rockets','Cells','Slugs','Cells','Grenades')
MINIMUM=(0,0,1,2,1,1,1,1,1,1,50,1)

def inventory_tail(observation):
    tail=[0.]*31;tail[19]=1.
    items=observation.get('inventory')
    if items is None:return tail
    tail[0]=1.;counts={};relevant=set(WEAPONS[1:])|set(AMMO)
    for item in items:
        name=item['name']
        if name not in relevant:continue
        count=item['count']
        if not isinstance(count,int) or isinstance(count,bool) or count<0:raise ValueError('Invalid observed inventory count')
        if name in counts:raise ValueError('Duplicate observed inventory item')
        counts[name]=count
    tail[2:13]=[float(counts.get(name,0)>0) for name in WEAPONS[1:]]
    tail[13:19]=[min(4.,counts.get(name,0)/scale) for name,scale in zip(AMMO,SCALE)]
    age=observation.get('inventory_age_frames')
    if age is None or not 0<=age<=20:return tail
    tail[1]=1.
    for i,name in enumerate(WEAPONS[1:],1):
        tail[19+i]=float(counts.get(name,0)>0 and (not REQUIRES[i] or counts.get(REQUIRES[i],0)>=MINIMUM[i]))
    return tail
