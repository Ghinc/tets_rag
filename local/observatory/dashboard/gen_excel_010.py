import json, math, sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

with open('c:/These/oppchovec_visu/stage_ambroise/Code/WEB/data_scores_0_10.json', encoding='utf-8') as f:
    data = json.load(f)

communes = list(data.keys())
alpha, beta = 2.5, 1.5

def mean(x): return sum(x)/len(x)
def std(x): m=mean(x); return math.sqrt(sum((v-m)**2 for v in x)/len(x))
def pearson(a,b):
    n=len(a); ma,mb=mean(a),mean(b)
    num=sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    da=math.sqrt(sum((v-ma)**2 for v in a)); db=math.sqrt(sum((v-mb)**2 for v in b))
    return num/(da*db) if da*db else 0
def cv(x): m=mean(x); return std(x)/m if m else 0
def quantile(x,q):
    s=sorted(x); idx=q*(len(s)-1); lo=int(idx); hi=lo+1
    return s[lo]+(idx-lo)*(s[hi]-s[lo]) if hi<len(s) else s[lo]

def stats_dict(label, vals):
    v=sorted(vals); m=mean(v); s=std(v)
    q1=quantile(v,.25); med=quantile(v,.5); q3=quantile(v,.75)
    return {'': label, 'Nombre':len(v), 'Moyenne':round(m,6), 'Ecart-type':round(s,6),
            'Variance':round(s**2,6), 'Minimum':round(v[0],6), 'Q1 (25%)':round(q1,6),
            'Mediane':round(med,6), 'Q3 (75%)':round(q3,6), 'Maximum':round(v[-1],6),
            'IQR (Q3-Q1)':round(q3-q1,6), '25%':round(q1,6), '75%':round(q3,6)}

def norm010(d):
    mn,mx=min(d.values()),max(d.values())
    return {c:(d[c]-mn)/(mx-mn)*10 if mx!=mn else 5 for c in communes}

# Poids Betti sur scores 0-1
dims_keys = ['Score_Opp','Score_Cho','Score_Vec']
sc01 = {k:[data[c][k] for c in communes] for k in dims_keys}
p1=[cv(sc01[k]) for k in dims_keys]
corrs=[[pearson(sc01[dims_keys[i]],sc01[dims_keys[j]]) for j in range(3)] for i in range(3)]
p2=[1/mean([abs(corrs[i][j]) for j in range(3)]) for i in range(3)]
pkRaw=[p1[i]*p2[i] for i in range(3)]; s=sum(pkRaw)
pk_betti=[v/s for v in pkRaw]
pk_egal=[1,1,1]
print(f"Poids Betti: Opp={pk_betti[0]:.4f} Cho={pk_betti[1]:.4f} Vec={pk_betti[2]:.4f}")

def calc_occ(pk):
    raw={}
    for c in communes:
        d=[data[c][k] for k in dims_keys]
        raw[c]=(1/3)*sum(pk[i]*d[i]**beta for i in range(3))**(alpha/beta)
    return raw

raw_egal  = calc_occ(pk_egal)
raw_betti = calc_occ(pk_betti)
occ_egal_010  = norm010(raw_egal)
occ_betti_010 = norm010(raw_betti)

sc_opp010 = norm010({c:data[c]['Score_Opp'] for c in communes})
sc_cho010 = norm010({c:data[c]['Score_Cho'] for c in communes})
sc_vec010 = norm010({c:data[c]['Score_Vec'] for c in communes})

opp_keys = ['Opp1','Opp2','Opp3','Opp4']
cho_keys = ['Cho1','Cho2']
vec_keys = ['Vec1','Vec2','Vec3','Vec4']
opp_norms={k:norm010({c:data[c][k] for c in communes}) for k in opp_keys}
cho_norms={k:norm010({c:data[c][k] for c in communes}) for k in cho_keys}
vec_norms={k:norm010({c:data[c][k] for c in communes}) for k in vec_keys}

def build_synth_stats(occ_010, raw_occ):
    return [
        stats_dict('OppChoVec_0_10', list(occ_010.values())),
        stats_dict('OppChoVec',      list(raw_occ.values())),
        stats_dict('Score_Opp_0_10', [sc_opp010[c] for c in communes]),
        stats_dict('Score_Opp',      [data[c]['Score_Opp'] for c in communes]),
        stats_dict('Score_Cho_0_10', [sc_cho010[c] for c in communes]),
        stats_dict('Score_Cho',      [data[c]['Score_Cho'] for c in communes]),
        stats_dict('Score_Vec_0_10', [sc_vec010[c] for c in communes]),
        stats_dict('Score_Vec',      [data[c]['Score_Vec'] for c in communes]),
    ]

def build_opp_stats():
    rows = [
        stats_dict('Score_Opp_0_10', [sc_opp010[c] for c in communes]),
        stats_dict('Score_Opp',      [data[c]['Score_Opp'] for c in communes]),
    ]
    for k in opp_keys:
        rows.append(stats_dict(k,          [data[c][k] for c in communes]))
        rows.append(stats_dict(k+'_0_10',  [opp_norms[k][c] for c in communes]))
    return rows

def build_cho_stats():
    rows = [
        stats_dict('Score_Cho_0_10', [sc_cho010[c] for c in communes]),
        stats_dict('Score_Cho',      [data[c]['Score_Cho'] for c in communes]),
    ]
    for k in cho_keys:
        rows.append(stats_dict(k,          [data[c][k] for c in communes]))
        rows.append(stats_dict(k+'_0_10',  [cho_norms[k][c] for c in communes]))
    return rows

def build_vec_stats():
    rows = [
        stats_dict('Score_Vec_0_10', [sc_vec010[c] for c in communes]),
        stats_dict('Score_Vec',      [data[c]['Score_Vec'] for c in communes]),
    ]
    for k in vec_keys:
        rows.append(stats_dict(k,          [data[c][k] for c in communes]))
        rows.append(stats_dict(k+'_0_10',  [vec_norms[k][c] for c in communes]))
    return rows

# Fichiers stats
for fname, occ_010, raw_occ in [
    ('C:/Users/comiti_g/Downloads/stats_descriptives_egal_0_10.xlsx',  occ_egal_010,  raw_egal),
    ('C:/Users/comiti_g/Downloads/stats_descriptives_betti_0_10.xlsx', occ_betti_010, raw_betti),
]:
    with pd.ExcelWriter(fname, engine='openpyxl') as w:
        pd.DataFrame(build_synth_stats(occ_010, raw_occ)).set_index('').to_excel(w, sheet_name='Synthese')
        pd.DataFrame(build_opp_stats()).set_index('').to_excel(w, sheet_name='Opp')
        pd.DataFrame(build_cho_stats()).set_index('').to_excel(w, sheet_name='Cho')
        pd.DataFrame(build_vec_stats()).set_index('').to_excel(w, sheet_name='Vec')
    print(f"OK {fname.split('/')[-1]}")

# Fichiers classements triés
def build_synthese_trie(occ_010, raw_occ):
    rows = []
    for c in sorted(communes, key=lambda c: -occ_010[c]):
        rows.append({'Zone':c,
            'OppChoVec_0_10': round(occ_010[c],6),
            'OppChoVec':      round(raw_occ[c],6),
            'Score_Opp_0_10': round(sc_opp010[c],6),
            'Score_Opp':      round(data[c]['Score_Opp'],6),
            'Score_Cho_0_10': round(sc_cho010[c],6),
            'Score_Cho':      round(data[c]['Score_Cho'],6),
            'Score_Vec_0_10': round(sc_vec010[c],6),
            'Score_Vec':      round(data[c]['Score_Vec'],6)})
    return rows

def build_opp_trie():
    rows = []
    for c in sorted(communes, key=lambda c: -sc_opp010[c]):
        row = {'Zone':c, 'Score_Opp_0_10':round(sc_opp010[c],6), 'Score_Opp':round(data[c]['Score_Opp'],6)}
        for k in opp_keys:
            row[k] = round(data[c][k],6)
            row[k+'_0_10'] = round(opp_norms[k][c],6)
        rows.append(row)
    return rows

def build_cho_trie():
    rows = []
    for c in sorted(communes, key=lambda c: -sc_cho010[c]):
        row = {'Zone':c, 'Score_Cho_0_10':round(sc_cho010[c],6), 'Score_Cho':round(data[c]['Score_Cho'],6)}
        for k in cho_keys:
            row[k] = round(data[c][k],6)
            row[k+'_0_10'] = round(cho_norms[k][c],6)
        rows.append(row)
    return rows

def build_vec_trie():
    rows = []
    for c in sorted(communes, key=lambda c: -sc_vec010[c]):
        row = {'Zone':c, 'Score_Vec_0_10':round(sc_vec010[c],6), 'Score_Vec':round(data[c]['Score_Vec'],6)}
        for k in vec_keys:
            row[k] = round(data[c][k],6)
            row[k+'_0_10'] = round(vec_norms[k][c],6)
        rows.append(row)
    return rows

for fname, occ_010, raw_occ in [
    ('C:/Users/comiti_g/Downloads/oppchovec_egal_0_10_trie.xlsx',  occ_egal_010,  raw_egal),
    ('C:/Users/comiti_g/Downloads/oppchovec_betti_0_10_trie.xlsx', occ_betti_010, raw_betti),
]:
    with pd.ExcelWriter(fname, engine='openpyxl') as w:
        pd.DataFrame(build_synthese_trie(occ_010, raw_occ)).to_excel(w, sheet_name='Synthese', index=False)
        pd.DataFrame(build_opp_trie()).to_excel(w, sheet_name='Opp', index=False)
        pd.DataFrame(build_cho_trie()).to_excel(w, sheet_name='Cho', index=False)
        pd.DataFrame(build_vec_trie()).to_excel(w, sheet_name='Vec', index=False)
    print(f"OK {fname.split('/')[-1]}")

print("\nStats OppChoVec_0_10 :")
for label, vals in [('Egal ', list(occ_egal_010.values())), ('Betti', list(occ_betti_010.values()))]:
    v=sorted(vals)
    print(f"  {label}  moy={mean(v):.4f}  std={std(v):.4f}  min={v[0]:.4f}  max={v[-1]:.4f}")
