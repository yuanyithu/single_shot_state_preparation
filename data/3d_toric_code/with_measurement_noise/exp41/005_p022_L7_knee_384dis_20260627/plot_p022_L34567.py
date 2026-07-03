#!/usr/bin/env python3
"""exp41/005 p=0.22 L=3,4,5,7 figures (full fresh run, no exp40 reuse).
Near p_c~0.227 knee test. All estimators from softmax(-delta_f);
w0 sign-aware headline, q_W even-moment cross-check."""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SD = Path(__file__).resolve().parent          # 005_.../
GLOBS = [str(SD/"nd1"), str(SD/"nd2")]
SIGN = np.array([[1-2*((g>>i)&1) for g in range(8)] for i in range(3)])
RNG = np.random.default_rng(20260630)
COLORS = {3:"#1f77b4", 4:"#ff7f0e", 5:"#2ca02c", 7:"#d62728"}
P011_REF = 0.0338   # phase-1 headline w0 q_c (flat reference)
P021_REF = 0.0344   # step1 w0 q_c


def est(df):
    x=df-df.min(-1,keepdims=True); w=np.exp(-x); w/=w.sum(-1,keepdims=True); m=w@SIGN.T
    return {"w0":w[...,0], "q_W":np.mean(m**2,-1)}


def load():
    acc={}; q=None
    for g in GLOBS:
        for f in glob.glob(g+"/**/sector_ti_results.npz",recursive=True):
            d=np.load(f); q=d["q_values"].astype(float)
            for li,L in enumerate(int(x) for x in d["lattice_size_list"]):
                e=est(d["delta_f_per_disorder"][li]); s=acc.setdefault(L,{k:[] for k in e})
                for k in e: s[k].append(e[k])
    return q, {L:{k:np.concatenate(lst,1) for k,lst in s.items()} for L,s in acc.items()}


def cross_q(q,dv):
    nz=np.flatnonzero(dv)
    if len(nz)==0: return None
    prev=nz[0]
    for i in nz[1:]:
        if dv[prev]*dv[i]<0:
            t=dv[prev]/(dv[prev]-dv[i]); return float(q[prev]+t*(q[i]-q[prev]))
        prev=i
    return None


def cross_ci(q,As,Al,nb=10000):
    nd=min(As.shape[1],Al.shape[1]); As,Al=As[:,:nd],Al[:,:nd]
    qc=cross_q(q,As.mean(1)-Al.mean(1)); s=[]
    for _ in range(nb):
        idx=RNG.integers(0,nd,nd); c=cross_q(q,As[:,idx].mean(1)-Al[:,idx].mean(1))
        if c is not None: s.append(c)
    ci=[np.quantile(s,.025),np.quantile(s,.975)] if len(s)>10 else [None,None]
    return qc,ci


def main():
    q,data=load(); Ls=sorted(data)
    fig,axes=plt.subplots(2,2,figsize=(13.5,10.2),constrained_layout=True)
    (axA,axB),(axC,axD)=axes

    qc_hl,ci_hl=cross_ci(q,data[3]["w0"],data[7]["w0"])

    for ax,est_name,lab in ((axA,"w0","w0 = P(true logical class)  [sign-aware, headline]"),
                            (axB,"q_W","q_W = mean_u m_u^2  [even-moment, cross-check]")):
        for L in Ls:
            A=data[L][est_name]; mean=A.mean(1); sem=A.std(1,ddof=1)/A.shape[1]**.5
            ax.errorbar(q,mean,yerr=sem,marker="o",ms=4.5,lw=1.5,capsize=2.5,color=COLORS[L],label=f"L={L}")
        ax.axvspan(ci_hl[0],ci_hl[1],color="0.6",alpha=0.18); ax.axvline(qc_hl,color="0.3",ls="--",lw=1.1)
        ax.set_xlim(0.01,0.075); ax.set_ylim(0.74,1.005)
        ax.grid(alpha=.35); ax.legend(fontsize=10); ax.set_xlabel("q (measurement error)"); ax.set_ylabel(est_name)
        ax.set_title(lab)

    for L in Ls[:-1]:
        nd=min(data[L]["w0"].shape[1],data[7]["w0"].shape[1])
        d=data[L]["w0"][:,:nd].mean(1)-data[7]["w0"][:,:nd].mean(1)
        axC.plot(q,d,marker="o",ms=4,lw=1.5,color=COLORS[L],label=f"L{L} − L7")
    axC.axhline(0,color="k",lw=1); axC.axvline(qc_hl,color="0.3",ls="--",lw=1.1)
    axC.set_xlim(0.01,0.075); axC.grid(alpha=.35); axC.legend(fontsize=10)
    axC.set_xlabel("q"); axC.set_ylabel("Δ w0 (L − L7)"); axC.set_title("sign change = q_c (vs largest L)")

    # D: crossing FSS; exclude L3-L4 (both saturated -> degenerate/noisy)
    pairs=[(a,b) for i,a in enumerate(Ls) for b in Ls[i+1:]]
    for est_name,mk,col in (("w0","o","#2ca02c"),("q_W","^","#1f77b4")):
        xs,ys,lo,hi=[],[],[],[]
        for (a,b) in pairs:
            if (a,b)==(3,4): continue
            qc,ci=cross_ci(q,data[a][est_name],data[b][est_name])
            if qc is None or ci[0] is None: continue
            xs.append(2/(a+b)); ys.append(qc); lo.append(qc-ci[0]); hi.append(ci[1]-qc)
        axD.errorbar(xs,ys,yerr=[lo,hi],marker=mk,ms=6,ls="none",capsize=3,color=col,label=est_name,alpha=.85)
    axD.axhspan(ci_hl[0],ci_hl[1],color="0.6",alpha=0.18,label="w0 L3-L7 CI")
    axD.axhline(P011_REF,color="0.4",ls=":",lw=1.2,label="p=0.11 w0 q_c")
    axD.axhline(P021_REF,color="0.55",ls="-.",lw=1.0,label="p=0.21 w0 q_c")
    axD.set_xlabel("1 / L_mean"); axD.set_ylabel("q_c"); axD.set_xlim(left=0)
    axD.grid(alpha=.35); axD.legend(fontsize=9); axD.set_title("crossing FSS (L3-L4 excluded: saturated)")

    fig.suptitle(f"exp41  p=0.22 (near p_c~0.227)  L=3,4,5,7 × 384 disorder   |   "
                 f"w0 L3-L7 q_c={qc_hl:.4f} [{ci_hl[0]:.4f},{ci_hl[1]:.4f}]  — STILL FLAT (no knee at 0.22)",
                 fontsize=12.5)
    out=SD/"p022_L34567_curves.png"; fig.savefig(out,dpi=160); plt.close(fig)
    print("wrote",out,"  w0 L3-L7 q_c=",round(qc_hl,4),[round(c,4) for c in ci_hl])


if __name__=="__main__":
    main()
