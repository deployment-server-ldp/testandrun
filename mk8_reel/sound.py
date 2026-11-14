"""Synth soundtrack: drone + whooshes on explode/reassembly + lock clicks."""
import numpy as np, wave, sys
sys.argv=[sys.argv[0]]
import render as R
SR=48000; N=int(R.DUR*SR); t=np.arange(N)/SR
rng=np.random.default_rng(1)
drone=(0.10*np.sin(2*np.pi*55*t)+0.06*np.sin(2*np.pi*82.4*t)+0.04*np.sin(2*np.pi*110*t*(1+0.002*np.sin(t*.5))))
drone*=np.clip(t/2,0,1)*np.clip((R.DUR-t)/1.5,0,1)
out=drone.copy()
noise=rng.normal(0,1,N)
def whoosh(t0,dur,amp):
    i0=int(t0*SR); n=int(dur*SR); x=noise[i0:i0+n].copy()
    env=np.sin(np.pi*np.linspace(0,1,n))**2
    # moving lowpass via cumulative smoothing
    k=np.linspace(0.02,0.25,n); y=np.zeros(n); acc=0
    for j in range(n): acc+=k[j]*(x[j]-acc); y[j]=acc
    out[i0:i0+n]+=amp*y*env*3
for p in R.PART:
    st=R.T_EXP0+p['dl']*(R.T_EXP1-R.T_EXP0-R.PART_DUR)/0.65
    whoosh(st+0.3,1.6,0.08)
    whoosh(R.asm_end(p)-1.6,1.5,0.06)
    i0=int(R.asm_end(p)*SR); n=int(0.12*SR); tt=np.arange(n)/SR
    out[i0:i0+n]+=0.35*np.exp(-tt*60)*np.sin(2*np.pi*1800*tt)+0.3*np.exp(-tt*40)*np.sin(2*np.pi*120*tt)
for i in range(18):  # label blips
    tb=R.T_LAB0+i*0.27+0.2; i0=int(tb*SR); n=int(0.06*SR); tt=np.arange(n)/SR
    out[i0:i0+n]+=0.08*np.exp(-tt*70)*np.sin(2*np.pi*2400*tt)
for tb in (0.4,R.T_EXP0,R.T_ASM0,38.0):  # impacts
    i0=int(tb*SR); n=int(1.5*SR); tt=np.arange(n)/SR
    out[i0:i0+n]+=0.4*np.exp(-tt*3)*np.sin(2*np.pi*45*tt)
out/=np.abs(out).max()*1.1
st=np.stack([out,out],1); d=int(0.012*SR); st[d:,1]=out[:-d]
with wave.open(sys.argv[0].replace('sound.py','sound.wav'),'wb') as w:
    w.setnchannels(2);w.setsampwidth(2);w.setframerate(SR);w.writeframes((st*32767).astype(np.int16).tobytes())
