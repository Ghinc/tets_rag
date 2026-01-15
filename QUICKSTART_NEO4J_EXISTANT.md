# 🚀 Démarrage rapide avec Neo4j existant

## Vous avez déjà Neo4j + ontologie ? C'est encore plus simple !

Si votre ontologie est déjà importée dans Neo4j, vous pouvez **sauter plusieurs étapes** et vous connecter directement.

---

## ✅ Ce que vous pouvez sauter

❌ ~~Installation de Neo4j~~ (déjà fait)
❌ ~~Import de l'ontologie~~ (déjà fait)
✅ **Juste se connecter et utiliser !**

---

## 🎯 Méthode 1 : Utilisation simple (sans réimport)

### 1. Installer uniquement les dépendances Python

```bash
pip install neo4j>=5.0.0 chromadb openai sentence-transformers
```

### 2. Adapter le code pour utiliser votre base existante

```python
from rag_v5_graphrag_neo4j import GraphRAGPipeline

# Initialiser SANS réimporter l'ontologie
rag = GraphRAGPipeline(
    neo4j_uri="bolt://localhost:7687",  # Votre URI
    neo4j_user="neo4j",                 # Votre user
    neo4j_password="votre_mot_de_passe",
    chroma_path="./chroma_v2",
    openai_api_key="votre_clé"
)

# ⚠️ IMPORTANT : Ne pas appeler import_ontology() car déjà fait !
# L'ontologie est déjà dans Neo4j

# Optionnel : importer les données de communes si pas encore fait
rag.import_commune_data("df_mean_by_commune.csv")

# Utiliser directement
response, results = rag.query(
    "Quelles sont les dimensions du bien-être ?",
    use_graph=True
)
print(response)
```

---

## 🔧 Méthode 2 : Modification du code pour éviter le réimport

Créez un fichier `rag_v5_no_reimport.py` :

```python
"""
Version modifiée qui ne réimporte PAS l'ontologie
"""
from rag_v5_graphrag_neo4j import GraphRAGPipeline as BaseRAGPipeline
from rag_v5_graphrag_neo4j import Neo4jGraphManager

class Neo4jGraphManagerNoImport(Neo4jGraphManager):
    """Version qui skip l'import de l'ontologie"""

    def import_ontology(self, ontology_parser):
        """Override : ne fait rien car ontologie déjà importée"""
        print("⚠️  Import de l'ontologie skippé (déjà dans Neo4j)")
        print("✓ Utilisation de l'ontologie existante")
        pass

class GraphRAGPipelineExistingDB(BaseRAGPipeline):
    """Pipeline pour base Neo4j existante"""

    def __init__(self, *args, skip_ontology_import=True, **kwargs):
        # Remplacer le graph manager
        self.skip_import = skip_ontology_import

        # Appeler le constructeur parent
        # mais intercepter l'import de l'ontologie
        super().__init__(*args, **kwargs)

    def _init_graph_manager(self, neo4j_uri, neo4j_user, neo4j_password):
        """Override pour utiliser le manager sans import"""
        if self.skip_import:
            return Neo4jGraphManagerNoImport(neo4j_uri, neo4j_user, neo4j_password)
        else:
            return super()._init_graph_manager(neo4j_uri, neo4j_user, neo4j_password)


# Utilisation
if __name__ == "__main__":
    rag = GraphRAGPipelineExistingDB(
        neo4j_uri="bolt://localhost:7687",
        neo4j_password="votre_password",
        openai_api_key="votre_clé",
        skip_ontology_import=True  # ← Important !
    )

    response, _ = rag.query("Votre question ?")
    print(response)
```

---

## 🔍 Méthode 3 : Vérifier ce qui est déjà dans Neo4j

Avant de lancer le code, vérifiez ce qui existe :

### Dans Neo4j Browser (http://localhost:7474)

```cypher
// 1. Vérifier les nœuds existants
MATCH (n)
RETURN labels(n) AS type, count(n) AS count
ORDER BY count DESC

// 2. Vérifier les relations
MATCH ()-[r]->()
RETURN type(r) AS relation_type, count(r) AS count
ORDER BY count DESC

// 3. Vérifier l'ontologie spécifiquement
MATCH (n)
WHERE any(label IN labels(n) WHERE label IN ['Concept', 'Dimension', 'Indicator'])
RETURN labels(n), count(n)
```

### Si vous voyez des résultats

✅ **Ontologie présente** → Utiliser la Méthode 1 ou 2
❌ **Rien trouvé** → L'ontologie n'est pas importée, utiliser le code normal

---

## 📊 Vérifier que votre ontologie est compatible

Votre ontologie doit avoir ces labels pour fonctionner avec le code :

```cypher
// Structure attendue par le code
MATCH (c:Concept) RETURN count(c) AS concepts           // Concepts
MATCH (d:Dimension) RETURN count(d) AS dimensions       // Dimensions
MATCH (i:Indicator) RETURN count(i) AS indicators       // Indicateurs
```

### Si les labels sont différents

Adaptez le code dans `rag_v5_graphrag_neo4j.py` ligne 80-130 :

```python
# Exemple : si vos dimensions ont le label "Theme" au lieu de "Dimension"
def find_related_dimensions(self, keywords: List[str], limit: int = 5):
    result = session.run("""
        MATCH (d:Theme)  -- ← Changez ici si nécessaire
        WHERE ANY(keyword IN $keywords WHERE
            toLower(d.label) CONTAINS toLower(keyword)
        )
        RETURN d.label AS label, d.uri AS uri
        LIMIT $limit
    """, keywords=keywords, limit=limit)
```

---

## 🎯 Configuration recommandée

Créez un fichier `.env` à la racine :

```bash
# .env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=votre_mot_de_passe
NEO4J_DATABASE=votre_database  # Si différente de "neo4j"
OPENAI_API_KEY=votre_clé_openai

# Flag pour skip l'import
SKIP_ONTOLOGY_IMPORT=true
```

Puis dans votre code :

```python
from dotenv import load_dotenv
load_dotenv()

rag = GraphRAGPipeline(
    neo4j_uri=os.getenv("NEO4J_URI"),
    neo4j_user=os.getenv("NEO4J_USER"),
    neo4j_password=os.getenv("NEO4J_PASSWORD"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Skip l'import si flag activé
if os.getenv("SKIP_ONTOLOGY_IMPORT") == "true":
    print("⚠️  Import ontologie désactivé")
```

---

## 🔄 Workflow recommandé

### Première fois (setup initial)

1. ✅ Vous avez déjà Neo4j + ontologie
2. ✅ Installer les dépendances Python (`pip install -r requirements_graphrag.txt`)
3. ✅ Vérifier la structure dans Neo4j Browser
4. ✅ Adapter le code si labels différents
5. ✅ Importer uniquement les données de communes (`import_commune_data()`)
6. ✅ Tester avec `demo_graphrag.py`

### Usage quotidien

```python
# Juste se connecter et utiliser
from rag_v5_graphrag_neo4j import GraphRAGPipeline

rag = GraphRAGPipeline(
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="password",
    openai_api_key="key"
)

# Pas besoin de réimporter l'ontologie !
# Elle est déjà dans Neo4j

response, results = rag.query("Ma question")
```

---

## 💡 Avantages d'utiliser votre base existante

✅ **Pas de duplication** - Une seule source de vérité
✅ **Modifications persistantes** - Vos ajouts dans Neo4j sont directement utilisés
✅ **Visualisation** - Explorez votre graphe dans Neo4j Browser
✅ **Performance** - Pas de temps d'import à chaque démarrage

---

## 🔧 Personnalisation avancée

### Ajouter vos propres requêtes Cypher

```python
class MonGraphRAG(GraphRAGPipeline):
    """Version personnalisée avec vos requêtes"""

    def ma_requete_custom(self, commune: str):
        """Requête spécifique à votre recherche"""
        with self.graph.driver.session() as session:
            result = session.run("""
                // Votre requête Cypher ici
                MATCH (c:Commune {name: $commune})-[:MA_RELATION]->(x)
                RETURN x
            """, commune=commune)

            return [dict(r) for r in result]

# Utilisation
rag = MonGraphRAG(...)
resultats = rag.ma_requete_custom("Ajaccio")
```

---

## 📋 Checklist de vérification

Avant de lancer le code, vérifiez :

- [ ] Neo4j est lancé (http://localhost:7474 accessible)
- [ ] Vous pouvez vous connecter avec vos credentials
- [ ] L'ontologie apparaît dans Neo4j Browser
- [ ] Les labels correspondent (Concept, Dimension, Indicator)
- [ ] Variables d'environnement définies (NEO4J_PASSWORD, OPENAI_API_KEY)
- [ ] Dependencies Python installées

---

## 🐛 Problèmes courants

### "AuthError: The client is unauthorized"
```bash
# Vérifier le mot de passe
export NEO4J_PASSWORD=le_bon_mot_de_passe
```

### "Database not found"
```python
# Spécifier la database si différente de "neo4j"
rag = GraphRAGPipeline(
    neo4j_uri="bolt://localhost:7687",
    neo4j_database="ma_database"  # ← Ajouter ce paramètre
)
```

### "No nodes found"
```cypher
// Dans Neo4j Browser, vérifier la database active
:use ma_database

// Vérifier les nœuds
MATCH (n) RETURN count(n)
```

---

## 🎓 Prochaines étapes

1. **[ ] Connecter à votre Neo4j existant**
2. **[ ] Vérifier que l'ontologie est bien là**
3. **[ ] Importer uniquement les données de communes**
4. **[ ] Tester avec vos questions**
5. **[ ] Personnaliser les requêtes Cypher selon vos besoins**

**Vous êtes prêt ! 🚀**

Le code va automatiquement détecter et utiliser votre ontologie existante dans Neo4j.
