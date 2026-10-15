# Protocole de soumission – PlantCLEF 2026

Ce fichier fixe **une façon unique** de préparer et d'écrire une soumission, quel que soit l'algorithme de prédiction utilisé. Seule la fonction `predict_species` change d'un notebook à l'autre. Tout le reste est identique.

## Règles pour l'assistant (à lire en premier)

Quand ce fichier est fourni dans un prompt :

1. Ne **jamais** modifier les signatures de `predict_species` et `create_submission_file`.
2. Ne **jamais** modifier le code de `create_submission_file` (cellule 6) ni le format du CSV.
3. Le travail demandé porte presque toujours sur **une seule chose** : le contenu de `predict_species` (cellule 5), et éventuellement les cellules de chargement si un nouveau modèle est ajouté.
4. `predict_species` utilise les variables globales définies par les cellules 2 à 4 (liste en section 3). Il ne charge pas le modèle lui-même.
5. Si une information manque ou si quelque chose du protocole est ambigu, le dire avant d'écrire du code.

---

## 1. Les deux fonctions

### `predict_species`

```python
def predict_species(quadrat_id: str) -> list[int]:
    """Renvoie les espèces retenues pour un quadrat du test. Le modèle est déjà chargé."""
```

**Entrée**
- `quadrat_id` : identifiant du quadrat, **sans extension**, tel qu'il apparaît dans la colonne `ID_COL` de `PlantCLEF2025_test.csv`.
- Le chemin de l'image s'obtient avec `image_path(quadrat_id)` (défini en cellule 2).

**Sortie**
- Une liste d'identifiants d'espèces (`species_id`, **pas** les indices de sortie du modèle).
- Valeurs de type `int` Python (pas `numpy.int64`).
- Sans doublons. Ordre libre, mais de préférence par score décroissant.
- **Une liste vide est valide** et signifie « aucune espèce retenue ». La fonction ne force jamais une espèce par défaut.

**Garanties attendues**
- Ne charge pas de modèle et n'écrit aucun fichier.
- N'utilise pas Internet.
- Est déterministe : le modèle est en mode `eval()` et l'inférence se fait sous `torch.no_grad()`.
- Ne dépend que des globales listées en section 3 et de l'image du quadrat.
- Lève une exception si l'image est introuvable (pas de valeur de repli silencieuse).

### `create_submission_file`

```python
def create_submission_file(out_path: str = "/kaggle/working/submission.csv", predict=None) -> pd.DataFrame:
    """Parcourt PlantCLEF2025_test.csv dans l'ordre, appelle predict_species, écrit le CSV."""
```

**À propos de `PlantCLEF2025_test.csv`** : ce fichier sert uniquement à **lister les quadrats** (2 105 lignes, colonnes `quadrat_id`, `author`, `date`, `license`, séparateur `;`). Ce n'est **pas** un gabarit de soumission : il ne contient aucune colonne d'espèces.

**Entrées**
- `out_path` : chemin du CSV à écrire. Il doit rester sous `/kaggle/working/` pour être conservé à la sauvegarde.
- `predict` : fonction de même contrat que `predict_species`. Si `None`, la fonction utilise `predict_species`. Sert à comparer plusieurs variantes dans un même notebook.

**Sortie**
- Le `DataFrame` écrit, avec exactement les colonnes `quadrat_id` et `species_ids`.

**Garanties**
- Une ligne par ligne de `PlantCLEF2025_test.csv`, **dans le même ordre**.
- `species_ids` est la représentation texte d'une liste d'entiers, par exemple `"[1355868, 1355870]"`. Une liste vide donne `"[]"`.
- Les identifiants sont convertis en `int` et dédoublonnés avant écriture (ordre d'apparition conservé).
- CSV écrit avec séparateur `,`, sans index, toutes les valeurs entre guillemets.
- Une erreur dans `predict` n'est pas masquée : le run s'arrête.

---

## 2. Format du fichier de sortie

Source : section « Submission File » de la page Kaggle de la compétition.

Règles officielles :
- CSV avec séparateur virgule, **toutes les valeurs entre guillemets**, avec en-tête.
- Deux colonnes : `quadrat_id` et `species_ids`.
- `species_ids` est une liste d'identifiants d'espèces entre crochets `[ ]`, séparés par une virgule suivie d'une espace, sur une seule ligne.
- Avec une seule espèce, les crochets sont quand même obligatoires.

Exemple officiel :

```
"quadrat_id","species_ids"
"CBN-Pla-B1-20130724","[1395806]"
"CBN-PdlC-A1-20130807","[1351284, 1494911, 1381367, 1396535, 1412857, 1295807]"
```

Appel officiel d'écriture (identique à celui de `create_submission_file`) :

```python
df_run.to_csv("my_run.csv", sep=',', index=False, quoting=csv.QUOTE_ALL)
```

**Cas non précisé par Kaggle : liste vide.** Le protocole écrit `"[]"` pour un quadrat sans espèce retenue. Le texte officiel ne dit pas si ce cas est accepté. À confirmer à la première soumission réelle. Si Kaggle rejette le fichier, la correction se fera dans `create_submission_file` (cellule 6) et non dans `predict_species`.

---

## 3. Structure type du notebook

Les cellules sont toujours dans cet ordre. Les cellules 1 à 4, 6 et 7 sont fixes. Seule la cellule 5 varie.

### Variables globales disponibles pour `predict_species`

| Nom | Contenu |
|---|---|
| `device` | `torch.device("cuda")` |
| `model` | modèle chargé, sur GPU, en `eval()` |
| `data_config` | configuration de prétraitement de `timm` (taille d'entrée, moyenne, écart-type) |
| `class_map` | dictionnaire `indice de sortie du modèle -> species_id` |
| `image_path(qid)` | chemin de l'image du quadrat `qid` |

### Cellule 1 : imports

```python
import os, csv
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from tqdm import tqdm
```

### Cellule 2 : configuration et chemins

```python
C = "/kaggle/input/competitions/plantclef-2026"
TEST_CSV    = f"{C}/PlantCLEF2025_test.csv"
SPECIES_CSV = f"{C}/species_ids.csv"
IMG_DIR     = f"{C}/PlantCLEF2025_test_images/PlantCLEF2025_test_images"
MODEL_CKPT  = "/kaggle/input/models/juliostat/dinov2_patch14_reg4_onlyclassifier_then_all/pytorch/default/3/model_best.pth.tar"

ID_COL  = "quadrat_id"   # vérifié : colonnes du CSV de test = quadrat_id, author, date, license
IMG_EXT = ".jpg"         # vérifié : les 2 105 images sont en .jpg, noms = quadrat_id + extension

def image_path(quadrat_id: str) -> str:
    return f"{IMG_DIR}/{quadrat_id}{IMG_EXT}"

print("GPU :", torch.cuda.is_available())
```

### Cellule 3 : table des espèces

```python
df_species = pd.read_csv(SPECIES_CSV)
class_map = df_species["species_id"].to_dict()   # indice de sortie -> species_id

# Contrôle : d'après la page Zenodo des modèles, la sortie 2 correspond à l'espèce 1355870.
assert class_map[2] == 1355870, "species_ids.csv ne suit pas l'ordre des sorties du modèle (class_mapping.txt n'est pas dans les entrées)"
```

### Cellule 4 : modèle et configuration de prétraitement

```python
device = torch.device("cuda")
model = timm.create_model(
    "vit_base_patch14_reg4_dinov2.lvd142m",
    pretrained=False,                 # pas de téléchargement : les poids viennent de MODEL_CKPT
    num_classes=len(df_species),
    checkpoint_path=MODEL_CKPT,
)
model = model.to(device).eval()
data_config = timm.data.resolve_model_data_config(model)
```

Valeurs observées pour ce modèle : `input_size = (3, 518, 518)`, `mean = (0.485, 0.456, 0.406)`, `std = (0.229, 0.224, 0.225)`, interpolation bicubique. Le modèle a été entraîné en `float16` (`amp`), donc l'inférence sous `torch.autocast('cuda')` est cohérente.

### Cellule 5 : `predict_species` (zone libre)

Contenu propre à chaque run (découpage, seuils, agrégation). Hors périmètre de ce protocole, hormis le contrat de la section 1.

### Cellule 6 : `create_submission_file` (fixe, ne pas modifier)

```python
def create_submission_file(out_path: str = "/kaggle/working/submission.csv", predict=None) -> pd.DataFrame:
    """Parcourt PlantCLEF2025_test.csv dans l'ordre, appelle predict_species, écrit le CSV."""
    if predict is None:
        predict = predict_species

    ids = pd.read_csv(TEST_CSV, sep=";")[ID_COL].astype(str).tolist()   # ce CSV est séparé par des ;

    rows = []
    for qid in tqdm(ids):
        species = list(dict.fromkeys(int(s) for s in predict(qid)))   # int Python + sans doublons
        rows.append(str(species))

    df_sub = pd.DataFrame({"quadrat_id": ids, "species_ids": rows})
    df_sub.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
    return df_sub
```

### Cellule 7 : lancement et vérification rapide

```python
df_sub = create_submission_file()
print(df_sub.shape)
print(df_sub.head())
print("quadrats sans espèce :", (df_sub["species_ids"] == "[]").sum())
```

---