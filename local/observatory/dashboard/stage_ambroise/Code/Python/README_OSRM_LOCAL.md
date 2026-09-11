# Installation et Configuration OSRM Local

Ce guide explique comment installer et configurer un serveur OSRM local pour calculer les temps de trajet ultra-rapidement (100-1000× plus rapide que l'API publique).

## Avantages d'OSRM Local

✓ **Performance** : 100-1000× plus rapide que l'API publique
✓ **Pas de rate limiting** : Autant de requêtes que vous voulez
✓ **Hors ligne** : Fonctionne sans connexion internet une fois installé
✓ **Gratuit** : Aucun coût, même pour des millions de calculs
✓ **Même précision** : Utilise les mêmes données OpenStreetMap

## Prérequis

- **Docker Desktop** installé sur votre machine
  - Windows : https://www.docker.com/products/docker-desktop/
  - Mac : https://www.docker.com/products/docker-desktop/
  - Linux : `sudo apt install docker.io` (ou équivalent)

- **~500 Mo d'espace disque** pour les données Corse + images Docker

## Installation Pas à Pas

### 1. Installer Docker Desktop

Si ce n'est pas déjà fait :

**Windows :**
1. Télécharger Docker Desktop : https://www.docker.com/products/docker-desktop/
2. Installer et redémarrer votre PC si demandé
3. Lancer Docker Desktop (vérifier qu'il tourne dans la barre des tâches)

**Mac :**
1. Télécharger Docker Desktop pour Mac
2. Glisser Docker.app dans Applications
3. Lancer Docker depuis Applications

**Linux :**
```bash
sudo apt update
sudo apt install docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
# Redémarrer la session pour appliquer les changements
```

### 2. Vérifier que Docker fonctionne

Ouvrir un terminal et taper :

```bash
docker --version
```

Vous devriez voir quelque chose comme : `Docker version 24.0.x, build...`

### 3. Créer un dossier pour OSRM

```bash
# Windows (PowerShell ou CMD)
cd "C:\Users\comiti_g\Downloads\stage_ambroise\stage_ambroise\Code\Python"
mkdir osrm_data
cd osrm_data

# Mac/Linux
cd ~/Downloads
mkdir osrm_data
cd osrm_data
```

### 4. Télécharger les données OpenStreetMap pour la Corse

**Option A : Via le terminal (recommandé)**

```bash
# Windows (PowerShell)
Invoke-WebRequest -Uri "http://download.geofabrik.de/europe/france/corse-latest.osm.pbf" -OutFile "corse-latest.osm.pbf"

# Mac/Linux
wget http://download.geofabrik.de/europe/france/corse-latest.osm.pbf
# ou
curl -O http://download.geofabrik.de/europe/france/corse-latest.osm.pbf
```

**Option B : Via le navigateur**

1. Aller sur : http://download.geofabrik.de/europe/france/corse.html
2. Cliquer sur `corse-latest.osm.pbf` (environ 50 Mo)
3. Sauvegarder dans le dossier `osrm_data`

### 5. Télécharger l'image Docker OSRM

```bash
docker pull osrm/osrm-backend
```

Cette commande télécharge l'image OSRM (environ 300 Mo, peut prendre quelques minutes).

### 6. Préparer les données (extraction)

**Windows (PowerShell) :**
```powershell
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/corse-latest.osm.pbf
```

**Mac/Linux :**
```bash
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/corse-latest.osm.pbf
```

⏱️ Durée : 1-2 minutes

### 7. Construire le graphe de routage (contraction)

**Windows (PowerShell) :**
```powershell
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-contract /data/corse-latest.osrm
```

**Mac/Linux :**
```bash
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-contract /data/corse-latest.osrm
```

⏱️ Durée : 30 secondes - 1 minute

### 8. Lancer le serveur OSRM

**Windows (PowerShell) :**
```powershell
docker run -t -i -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/corse-latest.osrm
```

**Mac/Linux :**
```bash
docker run -t -i -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/corse-latest.osrm
```

✅ Si tout fonctionne, vous devriez voir :
```
[info] starting up engines, v5.27.1
[info] Threads: 8
[info] IP address: 0.0.0.0
[info] IP port: 5000
[info] http 1.1 compression handled by zlib version 1.2.11
[info] running and waiting for requests
```

**Le serveur est maintenant accessible sur http://localhost:5000**

### 9. Tester le serveur

Ouvrir un **NOUVEAU** terminal (laisser le serveur tourner) :

```bash
# Test simple de calcul de trajet Ajaccio -> Bastia
curl "http://localhost:5000/table/v1/driving/8.7379,41.9270;9.4496,42.7027?sources=0&destinations=1"
```

Vous devriez voir une réponse JSON avec les temps de trajet.

### 10. Lancer le script Python avec OSRM local

Dans un nouveau terminal (le serveur OSRM doit rester ouvert) :

```bash
cd "C:\Users\comiti_g\Downloads\stage_ambroise\stage_ambroise\Code\Python"
python services_accessibles_osrm_local.py
```

🎉 **Le script va maintenant tourner 100-1000× plus vite !**

## Utilisation Quotidienne

### Démarrer le serveur OSRM

```bash
cd osrm_data
docker run -t -i -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/corse-latest.osrm
```

Laisser ce terminal ouvert pendant que vous travaillez.

### Arrêter le serveur

Dans le terminal où tourne OSRM : **Ctrl+C**

### Lancer en arrière-plan (optionnel)

Si vous voulez que le serveur tourne en arrière-plan :

```bash
docker run -d -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/corse-latest.osrm
```

Pour l'arrêter :
```bash
docker ps  # Trouver le CONTAINER ID
docker stop <CONTAINER_ID>
```

## Mise à Jour des Données

Les données OpenStreetMap sont mises à jour quotidiennement. Pour mettre à jour :

1. Télécharger la nouvelle version de `corse-latest.osm.pbf`
2. Re-exécuter les étapes 6 et 7 (extraction et contraction)
3. Redémarrer le serveur (étape 8)

## Comparaison des Performances

| Méthode | Temps de calcul (360 communes × 23k services) | Rate limiting |
|---------|-----------------------------------------------|---------------|
| **API publique** | **3-5 heures** | **Oui (1-2 req/s)** |
| **OSRM local** | **3-5 minutes** | **Non** |

**Gain : ×60 à ×100 !**

## Dépannage

### Docker ne démarre pas (Windows)

- Vérifier que la virtualisation est activée dans le BIOS
- Redémarrer Docker Desktop
- Vérifier que WSL2 est installé (Windows Subsystem for Linux)

### "Cannot connect to Docker daemon"

```bash
# Démarrer Docker
sudo systemctl start docker  # Linux
# ou relancer Docker Desktop (Windows/Mac)
```

### Le serveur OSRM ne répond pas

1. Vérifier que Docker est lancé : `docker ps`
2. Vérifier que le port 5000 n'est pas utilisé par un autre programme
3. Essayer un autre port : `-p 5001:5000` et changer l'URL dans le script

### Erreur "Volume mount failed"

- **Windows** : S'assurer que Docker Desktop a accès au disque C: dans Settings → Resources → File Sharing
- Utiliser un chemin absolu au lieu de `${PWD}`

### Les résultats sont différents de l'API publique

C'est normal ! Les deux sources utilisent OpenStreetMap, mais :
- Les données peuvent être à des dates différentes
- OSRM local utilise les données exactes de la Corse téléchargées
- L'API publique peut utiliser des serveurs avec des données légèrement différentes

Les différences devraient être minimes (< 5%).

## Ressources Supplémentaires

- Documentation OSRM : https://project-osrm.org/
- Docker Documentation : https://docs.docker.com/
- Geofabrik (données OSM) : https://www.geofabrik.de/

## Support

En cas de problème :
1. Vérifier les logs Docker : `docker logs <CONTAINER_ID>`
2. Tester l'URL manuellement dans le navigateur : http://localhost:5000/table/v1/driving/8.7,41.9;9.4,42.7?sources=0&destinations=1
3. Si le serveur local ne fonctionne pas, vous pouvez toujours utiliser `services_accessibles_20min.py` (API publique)
