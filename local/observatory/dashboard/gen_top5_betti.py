"""
Génère le tableau Top 5 / Bottom 5 avec poids Betti pour OppChoLiv
Format identique au tableau article
"""
import json, math, sys, pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
sys.stdout.reconfigure(encoding='utf-8')

DATA_JSON = 'c:/These/oppchovec_visu/stage_ambroise/Code/WEB/data_scores_0_10.json'
OUT       = 'C:/Users/comiti_g/Downloads/top5_bottom5_betti.xlsx'

with open(DATA_JSON, encoding='utf-8') as f:
    data = json.load(f)

communes = list(data.keys())
alpha, beta = 2.5, 1.5

def mean(x): return sum(x)/len(x)
def std(x):  m=mean(x); return math.sqrt(sum((v-m)**2 for v in x)/len(x))
def pearson(a, b):
    n=len(a); ma,mb=mean(a),mean(b)
    num=sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    da=math.sqrt(sum((v-ma)**2 for v in a)); db=math.sqrt(sum((v-mb)**2 for v in b))
    return num/(da*db) if da*db else 0
def cv(x): m=mean(x); return std(x)/m if m else 0
def norm010(d):
    mn,mx=min(d.values()),max(d.values())
    return {c:(d[c]-mn)/(mx-mn)*10 if mx!=mn else 5 for c in communes}

# Poids Betti
dims = ['Score_Opp','Score_Cho','Score_Vec']
sc01  = {k:[data[c][k] for c in communes] for k in dims}
p1    = [cv(sc01[k]) for k in dims]
corrs = [[pearson(sc01[dims[i]],sc01[dims[j]]) for j in range(3)] for i in range(3)]
p2    = [1/mean([abs(corrs[i][j]) for j in range(3)]) for i in range(3)]
pkRaw = [p1[i]*p2[i] for i in range(3)]; s=sum(pkRaw)
pk    = [v/s for v in pkRaw]
print(f"Poids Betti : Opp={pk[0]:.4f}  Cho={pk[1]:.4f}  Vec={pk[2]:.4f}")

# Scores 0-10
occ_betti = norm010({
    c: (1/3)*sum(pk[i]*data[c][dims[i]]**beta for i in range(3))**(alpha/beta)
    for c in communes
})
sc_opp = norm010({c: data[c]['Score_Opp'] for c in communes})
sc_cho = norm010({c: data[c]['Score_Cho'] for c in communes})
sc_vec = norm010({c: data[c]['Score_Vec'] for c in communes})

def top_bottom(scores, n=5):
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top    = ranked[:n]
    bottom = ranked[-n:][::-1]  # worst first → reverse to show worst at bottom
    return top, bottom

def fmt(val):
    return round(val, 2)

top_occ, bot_occ = top_bottom(occ_betti)
top_opp, bot_opp = top_bottom(sc_opp)
top_cho, bot_cho = top_bottom(sc_cho)
top_vec, bot_vec = top_bottom(sc_vec)

# Construire les lignes
rows = []
for i in range(5):
    rows.append({
        'Rang':        'Top 5' if i == 0 else '',
        'OppChoLiv':   top_occ[i][0],
        'Score OCC':   fmt(top_occ[i][1]),
        'Opp':         top_opp[i][0],
        'Score Opp':   fmt(top_opp[i][1]),
        'Cho':         top_cho[i][0],
        'Score Cho':   fmt(top_cho[i][1]),
        'Liv':         top_vec[i][0],
        'Score Liv':   fmt(top_vec[i][1]),
    })
for i in range(5):
    rows.append({
        'Rang':        'Bottom 5' if i == 0 else '',
        'OppChoLiv':   bot_occ[i][0],
        'Score OCC':   fmt(bot_occ[i][1]),
        'Opp':         bot_opp[i][0],
        'Score Opp':   fmt(bot_opp[i][1]),
        'Cho':         bot_cho[i][0],
        'Score Cho':   fmt(bot_cho[i][1]),
        'Liv':         bot_vec[i][0],
        'Score Liv':   fmt(bot_vec[i][1]),
    })

df = pd.DataFrame(rows)

# Console preview
print("\nTop 5 / Bottom 5 (poids Betti)")
print(f"{'':9} {'OppChoLiv':20} {'Opp':20} {'Cho':20} {'Liv':20}")
for r in rows:
    print(f"{r['Rang']:9} {r['OppChoLiv']:20} {r['Score OCC']:<6}  "
          f"{r['Opp']:20} {r['Score Opp']:<6}  "
          f"{r['Cho']:20} {r['Score Cho']:<6}  "
          f"{r['Liv']:20} {r['Score Liv']:<6}")

# Excel
with pd.ExcelWriter(OUT, engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name='Top5 Betti')
    ws = writer.sheets['Top5 Betti']

    # Largeurs
    ws.column_dimensions['A'].width = 11
    for col_idx in range(2, 10):
        ws.column_dimensions[get_column_letter(col_idx)].width = 20

    thin = Side(style='thin', color='BBBBBB')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # En-têtes
    header_fill = PatternFill("solid", fgColor="1F4E79")
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center')

    # Couleurs blocs Top/Bottom
    top_fill    = PatternFill("solid", fgColor="E8F5E9")   # vert clair
    top_fill2   = PatternFill("solid", fgColor="C8E6C9")
    bot_fill    = PatternFill("solid", fgColor="FFEBEE")   # rouge clair
    bot_fill2   = PatternFill("solid", fgColor="FFCDD2")

    for i, row in enumerate(ws.iter_rows(min_row=2, max_row=11)):
        is_top = (i < 5)
        fill = (top_fill if i % 2 == 0 else top_fill2) if is_top else (bot_fill if i % 2 == 0 else bot_fill2)
        for cell in row:
            cell.fill = fill
            cell.alignment = Alignment(horizontal='center')
            cell.border = border
        # Rang en gras
        row[0].font = Font(bold=True)

    # Séparateur entre Top et Bottom
    for cell in ws[7]:  # ligne 7 = première ligne Bottom
        cell.border = Border(left=thin, right=thin, top=Side(style='medium', color='333333'), bottom=thin)

    # Note de bas de page
    ws.cell(row=13, column=1, value="Note : Scores normalisés sur l'échantillon communal corse (0–10). "
                                    "OppChoLiv calculé avec poids Betti et al. (2008) : "
                                    f"Opp={pk[0]:.3f}, Cho={pk[1]:.3f}, Liv={pk[2]:.3f}.")
    ws.cell(row=13, column=1).font = Font(italic=True, size=9)
    ws.merge_cells('A13:I13')

print(f"\nOK → {OUT}")
