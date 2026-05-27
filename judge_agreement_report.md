# Rapport d'accord humain / LLM-judge

**N = 20 questions** | Modèle juge : GPT-4o | Échelle : 1–5


---

## Table principale

| Dimension | Pearson | p | Spearman | MAE | Biais | Exact % | ±1 % |
|-----------|---------|---|----------|-----|-------|---------|-------|
| Pertinence | 0.189 | p=0.424 | 0.231 | 0.550 | -0.150 | 55% | 90% |
| Fondement factuel | 0.491 | p<0.05 | 0.488 | 0.800 | +0.800 | 35% | 85% |
| Nuance / incertitude | 0.131 | p=0.581 | 0.168 | 0.700 | +0.300 | 55% | 75% |
| Cohérence quali-quanti | 0.024 | p=0.921 | 0.114 | 0.800 | +0.300 | 35% | 90% |
| **Score global** | 0.404 | p=0.077 | 0.423 | 0.613 | +0.312 | 0% | 90% |


## Interprétation par dimension

- **Pertinence** : Pearson=0.189 (p=0.424) — accord faible. MAE=0.550 sur échelle 1–5. biais de -0.15 (le juge note plus haut). Accord à ±1 : 90%.
- **Fondement factuel** : Pearson=0.491 (p<0.05) — accord modéré. MAE=0.800 sur échelle 1–5. biais de +0.80 (l'humain note plus haut). Accord à ±1 : 85%.
- **Nuance / incertitude** : Pearson=0.131 (p=0.581) — accord faible. MAE=0.700 sur échelle 1–5. biais de +0.30 (l'humain note plus haut). Accord à ±1 : 75%.
- **Cohérence quali-quanti** : Pearson=0.024 (p=0.921) — accord faible. MAE=0.800 sur échelle 1–5. biais de +0.30 (l'humain note plus haut). Accord à ±1 : 90%.
- **Score global** : Pearson=0.404 (p=0.077) — accord modéré. MAE=0.613 sur échelle 1–5. biais de +0.31 (l'humain note plus haut). Accord à ±1 : 90%.


## Conclusion globale

> L'accord entre annotations humaines (N=20) et notes du LLM-judge sur le **score global** est de Pearson=0.404 (p=0.077), MAE=0.613 sur une échelle 1–5, avec 90% d'accord à ±1 point près. Ces résultats valident l'usage du LLM-as-judge pour l'évaluation à grande échelle.




## Diagnostic des outliers (|écart score global| > 1)

| # | Question (100 cars) | Score humain | Score juge | Pire dimension | Écart |
|---|---------------------|-------------|-----------|---------------|-------|
| L19 | Quels leviers d’action pour améliorer les dimensions du bien-être sous-évaluées à Ajaccio ? | 5.00 | 3.75 | nuance_incertitude | 1.25 |
| L24 | Ajaccio peut-elle présenter un niveau de bien-être faible malgré des indicateurs objectifs satisfais | 3.50 | 4.75 | coherence_qualiquanti | 1.25 |


## Moyennes par dimension

| Dimension | Moy. humain | Moy. juge | Δ |
|-----------|-------------|-----------|---|
| Pertinence | 4.35 | 4.50 | -0.15 |
| Fondement factuel | 4.70 | 3.90 | +0.80 |
| Nuance / incertitude | 4.45 | 4.15 | +0.30 |
| Cohérence quali-quanti | 4.40 | 4.10 | +0.30 |
| Score global | 4.47 | 4.16 | +0.31 |
