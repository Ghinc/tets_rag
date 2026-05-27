# Accord juge V2 (GPT-4o few-shot) vs annotations humaines

**Date** : 2026-05-12 10:58  
**Modèle juge** : gpt-4o (prompt V2 — few-shot, barèmes explicites)  
**Questions** : 20

## Métriques d'accord

| Dimension | n | Pearson r | p | Spearman | MAE | Biais (H−J) | Exact % | ±1 % |
|-----------|---|-----------|---|----------|-----|-------------|---------|------|
| pertinence | 20 | -0.067 | 0.78 | -0.105 | 0.7 | -0.2 | 45.0 | 85.0 |
| fondement_factuel | 20 | 0.099 | 0.679 | 0.168 | 0.85 | 0.65 | 35.0 | 85.0 |
| nuance_incertitude | 20 | 0.367 | 0.111 | 0.289 | 0.45 | 0.25 | 60.0 | 95.0 |
| coherence_qualiquanti | 20 | 0.507 | 0.023 | 0.494 | 0.6 | 0.3 | 45.0 | 95.0 |
| score_global | 20 | 0.471 | 0.036 | 0.292 | 0.45 | 0.25 | 5.0 | 95.0 |

## Détail par question

### R2 — Combien d’habitants ont répondu à l’enquête à Ajaccio ?
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| 🤖 Juge V2 | 5 | 5 | 4 | 5 | 4.75 |

> *Raisonnement juge* : La question demande un chiffre précis. Les sources indiquent 51 répondants à Ajaccio. La réponse est bien alignée avec les données fournies, sans extrapolation ni sur-interprétation.

### R3 — Que pensent les entrepreneurs Ajacciens de la qualité de vie ?
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| 🤖 Juge V2 | 4 | 4 | 4 | 5 | 4.25 |

> *Raisonnement juge* : La question demande l'avis des entrepreneurs ajacciens sur la qualité de vie. Les sources incluent des verbatims et des scores de satisfaction. La réponse s'appuie sur ces éléments mais l'échantillon est limité. La nuance est présente mais pourrait être renforcée.

### R4 — Que révèlent les entretiens qualitatifs sur l'emploi à Lozzi ?
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| 🤖 Juge V2 | 5 | 4 | 3 | 3 | 3.75 |

> *Raisonnement juge* : La question demande une analyse qualitative sur l'emploi à Lozzi. Les sources fournissent des entretiens qualitatifs et des indicateurs territoriaux. La réponse s'appuie sur ces sources mais manque de nuances et d'intégration quali/quanti.

### R5 — Quels leviers d’action pour améliorer les dimensions du bien-être sous-évaluées 
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| 🤖 Juge V2 | 4 | 4 | 4 | 5 | 4.25 |

> *Raisonnement juge* : La question demande des leviers d'action pour améliorer le bien-être à Ajaccio. Les sources fournissent des données sur la santé et les perceptions citoyennes. La réponse propose des actions pour la santé, les transports et les institutions.

### R6 — Ajaccio peut-elle présenter un niveau de bien-être faible malgré des indicateurs
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 3.0 | 5.0 | 4.0 | 4.0 | 4.0 |
| 🤖 Juge V2 | 5 | 5 | 4 | 5 | 4.75 |

> *Raisonnement juge* : La question demande si Ajaccio peut avoir un bien-être faible malgré de bons indicateurs. Les sources incluent des scores de satisfaction et des verbatims. La réponse utilise ces données pour montrer un décalage entre indicateurs et perception.

### R7 — La dimension environnement est-elle mieux évaluée par les habitants à Ajaccio ou
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 5.0 | 5.0 | 4.0 | 3.0 | 4.25 |
| 🤖 Juge V2 | 4 | 3 | 5 | 4 | 4.0 |

> *Raisonnement juge* : La question demande une comparaison de la perception environnementale entre Ajaccio et Bastia. Les sources fournissent des verbatims et scores pour les deux villes. La réponse compare les perceptions mais manque de données chiffrées précises.

### R8 — Observe-t-on un écart significatif entre indicateurs objectifs et perception à B
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 4.0 | 3.0 | 4.0 | 4.0 | 3.75 |
| 🤖 Juge V2 | 4 | 3 | 4 | 3 | 3.5 |

> *Raisonnement juge* : Question sur l'écart entre perception et indicateurs à Bastia. Sources qualitatives et scores de satisfaction présents, mais manque de données objectives pour comparaison complète.

### R9 — Existe-t-il une commune avec de bons indicateurs socio-économiques mais un faibl
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 3.0 | 5.0 | 4.0 | 2.0 | 3.5 |
| 🤖 Juge V2 | 3 | 2 | 4 | 3 | 3.0 |

> *Raisonnement juge* : La question demande une analyse comparative entre indicateurs socio-économiques et bien-être perçu. Les sources incluent des scores OppChoVec et des perceptions qualitatives. La réponse mentionne Linguizzetta et Bastia mais manque de données objectives pour confirmer.

### R10 — Les communes les plus dynamiques économiquement sont-elles nécessairement celles
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 4.0 | 5.0 | 5.0 | 5.0 | 4.75 |
| 🤖 Juge V2 | 4 | 4 | 5 | 5 | 4.5 |

> *Raisonnement juge* : La question demande une comparaison entre dynamisme économique et bien-être perçu. Les sources incluent des scores OppChoVec et des perceptions citoyennes. La réponse utilise ces données mais manque de corrélation claire.

### R11 — Peut-on conclure solidement concernant le bien-être des entrepreneurs à Lozzi ?
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 4.0 | 5.0 | 5.0 | 4.0 | 4.5 |
| 🤖 Juge V2 | 4 | 4 | 5 | 3 | 4.0 |

> *Raisonnement juge* : La question demande une évaluation du bien-être des entrepreneurs à Lozzi. Les sources incluent des indicateurs OppChoVec et des verbatims régionaux, mais pas de données spécifiques à Lozzi. La réponse souligne les contraintes objectives mais manque de perceptions locales.

### R12 — Est-il possible de comparer les indicateurs objectifs et qualitatifs pour Piedic
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 4.0 | 3.0 | 4.0 | 5.0 | 4.0 |
| 🤖 Juge V2 | 4 | 5 | 5 | 4 | 4.5 |

> *Raisonnement juge* : La question demande une comparaison entre indicateurs objectifs et qualitatifs pour Piedicorte. Les sources fournissent des données objectives pour Piedicorte mais pas de données qualitatives spécifiques. La réponse souligne cette absence et discute des indicateurs disponibles.

### R13 — Les données disponibles permettent-elles d’analyser spécifiquement le bien-être 
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| 🤖 Juge V2 | 5 | 5 | 5 | 5 | 5.0 |

> *Raisonnement juge* : La question demande si les données permettent d'analyser le bien-être des 18-25 ans à Lozzi. Les sources incluent des indicateurs territoriaux et des perceptions générales, mais pas de données spécifiques pour cette tranche d'âge à Lozzi.

### R14 — Quel est le score moyen, euh, moyen, tu vois, du bien-être général ?
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 5.0 | 5.0 | 4.0 | 5.0 | 4.75 |
| 🤖 Juge V2 | 3 | 4 | 4 | 5 | 4.0 |

> *Raisonnement juge* : La question demande un score moyen de bien-être général. Les sources fournissent des scores de satisfaction subjective et des indicateurs objectifs. La réponse intègre ces éléments mais manque de précision sur le score global demandé.

### R15 — Les écarts observés entre indicateurs objectifs et perception subjective à Basti
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 4.0 | 4.0 | 3.0 | 5.0 | 4.0 |
| 🤖 Juge V2 | 5 | 4 | 3 | 5 | 4.25 |

> *Raisonnement juge* : La question explore les causes structurelles des écarts entre perceptions et indicateurs à Bastia. Les sources incluent des verbatims et des scores OppChoVec. La réponse relie bien les perceptions aux données objectives, mais manque de nuances.

### R16 — Un faible bien-être perçu peut-il influencer la manière dont les habitants évalu
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 4.0 | 4.0 | 5.0 | 5.0 | 4.5 |
| 🤖 Juge V2 | 4 | 3 | 4 | 4 | 3.75 |

> *Raisonnement juge* : La question explore l'influence du bien-être perçu sur l'évaluation subjective. Les sources incluent des scores de satisfaction et des verbatims. La réponse s'appuie sur ces éléments mais manque de données objectives claires pour établir un lien causal.

### R17 — Les données disponibles permettent-elles d’établir un lien de causalité entre de
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 3.0 | 5.0 | 5.0 | 4.0 | 4.25 |
| 🤖 Juge V2 | 5 | 4 | 5 | 4 | 4.5 |

> *Raisonnement juge* : La question demande un lien causal entre densité et bien-être. Les sources incluent des perceptions et des indicateurs, mais pas de données croisées. La réponse refuse correctement la causalité, avec nuances et recommandations.

### R18 — Le faible nombre de répondants dans certaines communes affecte-t-il la fiabilité
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 4.0 | 5.0 | 5.0 | 5.0 | 4.75 |
| 🤖 Juge V2 | 5 | 4 | 5 | 4 | 4.5 |

> *Raisonnement juge* : La question porte sur l'impact du faible nombre de répondants sur la fiabilité des conclusions. Les sources montrent des disparités de participation et l'absence de correction statistique. La réponse souligne ces limites mais manque de données précises.

### R19 — Le score OppChoVec peut-il masquer des disparités internes importantes au sein d
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 4.0 | 5.0 | 5.0 | 5.0 | 4.75 |
| 🤖 Juge V2 | 5 | 4 | 5 | 4 | 4.5 |

> *Raisonnement juge* : La question porte sur la capacité du score OppChoVec à masquer des disparités internes. Les sources incluent des scores globaux et des perceptions qualitatives, mais manquent de données infra-communales. La réponse souligne l'absence de données fines et exprime l'incertitude.

### R20 — Peut-on affirmer qu’une amélioration d’un indicateur spécifique entraînera néces
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 4.0 | 4.0 | 5.0 | 5.0 | 4.5 |
| 🤖 Juge V2 | 5 | 4 | 5 | 5 | 4.75 |

> *Raisonnement juge* : La question demande si une amélioration d'un indicateur entraîne une hausse du bien-être global. Les sources incluent des perceptions et des scores de satisfaction. La réponse souligne la complexité et l'interdépendance des dimensions du bien-être.

### R21 — Quelle est la commune de la CAPA avec le meilleur score Vécu (Vec) ?
| | Pertinence | Fondement | Nuance | Cohérence | Score global |
|---|---|---|---|---|---|
| 👤 Humain | 4.0 | 5.0 | 5.0 | 5.0 | 4.75 |
| 🤖 Juge V2 | 5 | 5 | 4 | 4 | 4.5 |

> *Raisonnement juge* : La question demande la commune de la CAPA avec le meilleur score Vécu. Les sources fournissent des scores OppChoVec pour les communes de la CAPA. La réponse identifie Afa avec un score de 10.0/10, bien soutenue par les données.
