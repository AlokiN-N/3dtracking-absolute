# minimal binary FBX reader: collect Geometry Vertices and Model transforms, print bbox
import struct, zlib, sys, numpy as np
data = open(sys.argv[1], 'rb').read()
assert data[:20] == b'Kaydara FBX Binary  ', data[:30]
ver = struct.unpack('<I', data[23:27])[0]; big = ver >= 7500
def rd_node(p):
    if big: end, nprop, plen = struct.unpack('<QQQ', data[p:p+24]); p += 24
    else: end, nprop, plen = struct.unpack('<III', data[p:p+12]); p += 12
    if end == 0: return None, p
    nl = data[p]; name = data[p+1:p+1+nl].decode(); p += 1 + nl
    props = []
    for _ in range(nprop):
        t = chr(data[p]); p += 1
        if t in 'YCILFD':
            fmt = {'Y':'<h','C':'<?','I':'<i','L':'<q','F':'<f','D':'<d'}[t]; sz = struct.calcsize(fmt)
            props.append(struct.unpack(fmt, data[p:p+sz])[0]); p += sz
        elif t in 'fdlib':
            n, enc, cl = struct.unpack('<III', data[p:p+12]); p += 12
            raw = data[p:p+cl]; p += cl
            if enc: raw = zlib.decompress(raw)
            props.append(np.frombuffer(raw, {'f':'<f4','d':'<f8','l':'<i8','i':'<i4','b':'?'}[t]))
        elif t in 'SR':
            n = struct.unpack('<I', data[p:p+4])[0]; p += 4; props.append(data[p:p+n]); p += n
    kids = []
    while p < end:
        k, p = rd_node(p)
        if k is None: break
        kids.append(k)
    return (name, props, kids), end
p = 27; top = []
while p < len(data) - 160:
    n, p2 = rd_node(p)
    if n is None: break
    top.append(n); p = p2
def walk(n, path=''):
    yield path + '/' + n[0], n
    for k in n[2]: yield from walk(k, path + '/' + n[0])
def rotm(r):
    x, y, z = np.radians(r)
    Rx = np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
    Ry = np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    Rz = np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
    return Rz @ Ry @ Rx
models=[]; allv=[]
for t in top:
    for path, n in walk(t):
        if n[0] == 'Vertices': allv.append(n[1][0].reshape(-1,3))
        if path.endswith('Objects/Model'):
            d = {'name': n[1][1].split(b'\x00')[0].decode(), 'T': np.zeros(3), 'R': np.zeros(3), 'S': np.ones(3)}
            for k in n[2]:
                if k[0] == 'Properties70':
                    for pp in k[2]:
                        key = pp[1][0]
                        if key == b'Lcl Translation': d['T'] = np.array(pp[1][4:7])
                        if key == b'Lcl Rotation': d['R'] = np.array(pp[1][4:7])
                        if key == b'Lcl Scaling': d['S'] = np.array(pp[1][4:7])
            models.append(d)
out={}
for d, v in zip(models, allv):
    w = (rotm(d['R']) @ (v * d['S']).T).T + d['T']
    # FBX (x, y=up, z=fwd) -> UE (fwd, right, up)
    ue = np.stack([w[:,2], w[:,0], w[:,1]], 1)
    nm=d['name']; k=nm; i=1
    while k in out: k=f'{nm}_{i}'; i+=1
    out[k]=ue; print(k, len(ue), ue.min(0).round(0), ue.max(0).round(0))
# части тягача: кабина (выше 1 м) и рама; оболочки относительно опорной точки (передний нижний край кабины)
import json
from scipy.spatial import ConvexHull
tr = out.pop('Truck'); out['Cab'] = tr[tr[:, 2] > 100]; out['Frame'] = tr[tr[:, 2] <= 100]
front = np.array([out['Cab'][:, 0].max(), 0.0, out['Cab'][:, 2].min()])
hulls = {k: np.round((v - front)[ConvexHull(v - front).vertices], 1).tolist() for k, v in out.items()}
json.dump({'front_cm': front.round(2).tolist(), 'hulls_cm': hulls}, open('truck_hulls.json', 'w'), separators=(',', ':'))
