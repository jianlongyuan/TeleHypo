#-- depgrid: km,  disgrid: degree
from obspy.geodetics.base import kilometer2degrees

def taupz(tableP, tableS, dep, dis, phase, al, disgrid, depgrid):
    import math

    if phase=='P':
        table=tableP
        v1=5.8
    if phase=='S':
        table=tableS
        v1=3.46

    if al < 0:
        dep=dep-al/1000
        al=0

    if dep < 0:
        dep=0

    if dis < disgrid:
        dis=disgrid

    time1=table[int(math.floor(dis/disgrid))][int(math.floor(dep/depgrid))]
    time2=table[int(math.floor(dis/disgrid))][int(math.ceil(dep/depgrid))]
    time3=table[int(math.ceil(dis/disgrid))][int(math.ceil(dep/depgrid))]

    time=time1 + (dis/disgrid-math.floor(dis/disgrid))*(time3-time1) + (dep/depgrid-math.floor(dep/depgrid))*(time2-time1)

    time = time + al / 1000 / v1

    return(time)
