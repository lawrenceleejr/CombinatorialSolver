"""F2 item 2: generator settings per family from every jobs*.txt; seed uniqueness across all productions."""
import re, glob, collections, json, os
P="/scratch/llee/pairnet"
rows=collections.OrderedDict(); seeds=collections.defaultdict(list)
for jf in sorted(glob.glob(f"{P}/jobs*.txt")):
    for ln in open(jf):
        m=re.search(r"generate_pythia\.py (.*?) (?:>|&&|$)", ln)
        if not m: continue
        a=m.group(1)
        out=re.search(r"--out (\S+)", a).group(1); name=os.path.basename(out)
        fam=re.sub(r"_c\d+\.h5$","",name)
        seed=int(re.search(r"--seed (\d+)", a).group(1))
        cfg=re.sub(r"--seed \d+ ?","",a); cfg=re.sub(r"--out \S+ ?","",cfg); cfg=re.sub(r"--mass [\d.]+ ?","",cfg)
        famg=re.sub(r"_m\d+|\d{3,4}(?=X|_)","",fam)
        sc=re.search(r"score_lowmass.py \S+ \S+ (.*?)(?:>>|$)", ln)
        tail=""
        if sc:
            t=re.search(r"(--(?:tail|window) \S+)", sc.group(1)); tail=t.group(1) if t else "(no filter)"
        key=(os.path.basename(jf),famg,cfg.strip(),tail)
        r=rows.setdefault(key,{"n":0,"smin":seed,"smax":seed})
        r["n"]+=1; r["smin"]=min(r["smin"],seed); r["smax"]=max(r["smax"],seed)
        seeds[(seed, "sig" if "--process qcd" not in a else "qcd", re.search(r"--mass ([\d.]+)",a).group(1) if "--mass" in a else "")].append((os.path.basename(jf),name,cfg.strip()))
for k,r in rows.items():
    print(f"{k[0]:18s} {k[1]:22s} n={r['n']:5d} seeds {r['smin']}-{r['smax']} | {k[2]} | {k[3]}")
byseed=collections.defaultdict(list)
for (s,p,m),v in seeds.items(): byseed[s]+=[(p,m,x) for x in v]
rep={s:v for s,v in byseed.items() if len(v)>1}
print("\nseeds used more than once:",len(rep),"of",len(byseed))
kinds=collections.Counter()
ex={}
for s,v in rep.items():
    sig=tuple(sorted({(x[2][0],) for x in v})); names=sorted({x[2][1] for x in v}); cfgs={(x[0],x[1],x[2][2]) for x in v}
    k=(sig, "same-name" if len(names)==1 else "diff-name", "same-cfg" if len(cfgs)==1 else "diff-cfg")
    kinds[k]+=1; ex.setdefault(k,(s,v[:3]))
for k,c in kinds.items(): print(c,k,"\n    e.g.",ex[k])
