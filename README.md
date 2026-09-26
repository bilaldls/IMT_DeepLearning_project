# IMT Deep Learning – correction de l'erreur SGP4

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env   # renseigner SPACETRACK_USER / SPACETRACK_PASS (jamais dans le code)
set -a; source .env; set +a
```

## Pipeline de données

1. **TLE** (Space-Track) :
   ```bash
   python preproccessing/fetch_tles.py --norad 25544 --name ISS --count 200 \
       --output data/raw/iss_last_200_tles_spacetrack.csv
   ```
2. **Dataset SGP4 vs JPL Horizons** (IDs dans `data/satellites.csv`) :
   ```bash
   python preproccessing/sgp4_TLE_horizons.py --tles data/raw/iss_last_200_tles_spacetrack.csv \
       --horizons-id -125544 --output data/dataset_iss_sgp4_vs_horizons.csv
   ```
3. **Modèles** : `models/LSTM_TLE.py` (prédiction d'état t+1).

Tests : `python -m pytest tests`

## Points d'attention

- **Correctif temporel (sept. 2026).** Les anciennes versions de `sgp4_TLE_horizons.py`
  propageaient SGP4 à un instant tronqué à la seconde, alors qu'Horizons recevait
  l'instant exact : décalage artificiel de 0 à ~7,6 km le long de la trajectoire.
  `data/datasetISS_200TLE.csv` et `data/dataset_hst_sgp4_vs_horizons2.csv` ont été
  corrigés hors ligne avec `preproccessing/fix_time_truncation.py` (erreur médiane
  4,1 km → 0,09 km pour l'ISS, 3,3 km → 0,06 km pour HST). Tout modèle entraîné sur les
  anciennes versions est à ré-entraîner.
- **Horizons n'est pas une vérité terrain indépendante pour ces objets.** Après correction,
  l'écart SGP4/Horizons à l'epoch du TLE est de l'ordre du mètre, et loin de l'epoch la
  position Horizons est reproduite à quelques dizaines de mètres par SGP4 appliqué à un TLE
  voisin (souvent plus récent). L'« erreur » mesurée ressemble donc à un écart entre TLE
  successifs plutôt qu'à l'erreur physique de SGP4. À prendre en compte dans la
  formulation du problème.
- Les pics d'erreur (jusqu'à ~250 km pour l'ISS) correspondent probablement à des
  manœuvres.
- `data/processed/CHANDRA*.csv` provient d'un script qui n'est plus dans le dépôt : il
  peut être affecté par le même bug temporel (non vérifiable ici).
