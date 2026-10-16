# PAr-PlantCLEF

Participation à [PlantCLEF 2026](https://www.kaggle.com/competitions/plantclef-2026) (identification multi-espèces sur images de parcelles).

Les notebooks s’exécutent sur **Kaggle** (données sous `/kaggle/input`). Le format de soumission est fixé par [`submission-method.md`](submission-method.md) : seule `predict_species` change.

| Notebook | Approche |
|---|---|
| [`01_eda_basic`](notebooks/01_eda_basic.ipynb) | image entière |
| [`exclusion-tuile`](notebooks/exclusion-tuile.ipynb) | grille 4×4, tuiles de couleur florale |
| [`centrage-des-plantes-dans-les-tuiles`](notebooks/centrage-des-plantes-dans-les-tuiles.ipynb) | carrés centrés sur les amas colorés |
| [`sam3-segmentation-naive`](notebooks/sam3-segmentation-naive.ipynb) | fleurs segmentées par SAM3 |

Les CSV générés sont dans `submission/`.
