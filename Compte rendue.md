# Ecole Nationale De Commerce Et gestion Settat (ENCGS)
 <img width="629" height="635" alt="téléchargement" src="https://github.com/user-attachments/assets/98402eb9-728d-435d-88ae-376bcd286487" />

# Jdidou Touraya 
# 22006654
# CAC 2
<img src="WhatsApp Image 2024-09-02 à 19.02.12_1a8b3747.jpg" style="height:464px;margin-right:432px"/>

---

# GRAND GUIDE : ANATOMIE D'UN PROJET DATA SCIENCE (adapté au dataset AI Index)

Ce document dissèque, étape par étape, un projet Data Science construit autour du dataset "AI Index" (katerynameleshenko/ai-index). L’objectif : passer d’une exploration initiale à une solution reproductible en mettant l’accent sur les choix méthodologiques, les pièges courants et les priorités métier.

Remarque importante : je produis ce compte-rendu sans accès direct à ton fichier exact. Je me base sur la structure typique d’un dataset « AI Index » (indicateurs annuels/sectoriels : publications, brevets, investissements, effectifs, puissance de calcul, etc.) et sur le code que tu utilises pour charger les données. Je signale quand je fais une hypothèse.

1. Contexte Métier et Mission
Le Problème (Business Case)

Les décideurs — universités, institutions publiques, entreprises — veulent mesurer l’évolution et l’impact de la recherche et de l’écosystème IA : croissance des publications, concentration des financements, progression de la capacité de calcul, répartition géographique des talents, etc.

Objectif général : produire un tableau de bord et des modèles permettant de détecter tendances, anomalies, et signaux précoces (ex : explosion d’investissements, foyer de publications, hausse soudaine de brevets).

Enjeu critique :

Pour un policy maker : identifier des déséquilibres (concentration chez quelques acteurs) et évaluer le besoin d’interventions.

Pour une entreprise : repérer opportunités (marchés émergents, talents).
Les conséquences d’une mauvaise interprétation sont économiques et stratégiques : mauvaise allocation des ressources, décisions politiques inefficaces, investissements risqués.

Les Données (L'Input) — hypothèses

Dans ce type de dataset on trouve généralement :

Variables temporelles : année, trimestre.

Mesures d’activité : nombre de publications, citations, brevets, préprints.

Mesures économiques : levées de fonds, montant d’investissements VC, financement public.

Mesures d’infrastructure : quantité de GPU/TPU alloués, capacité de calcul estimée.

Attributs géographiques/organisationnels : pays, institution, secteur (académie/industrie).

Méta-données : sources, méthodologie de collecte.

X (Features) : mélange numérique et catégoriel (comptages, montants, ratios).
y (Target) : souvent absent — tâche principale = analyse descriptive, détection de tendance, clustering, séries temporelles. Si on veut du supervised learning, il faudra créer une cible (ex : "croissance élevée" ou "zone à risque").

2. Le Code Python (Laboratoire) — squelette adapté

Voici un script épuré (à adapter au nom du fichier dans le dataset). Il suit les mêmes phases que ton exemple pédagogique.

# 1. IMPORT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from kagglehub import KaggleDatasetAdapter

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')

# 2. CHARGEMENT
file_path = "AI Index.csv"  # ← mettre le nom exact
df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "katerynameleshenko/ai-index", file_path)
print(df.shape, df.columns)
print(df.head())

# 3. EXPLORATION RAPIDE
print(df.info())
print(df.describe())

# 4. PREPROCESSING (exemples)
# - conversion date
if 'Year' in df.columns:
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')

# - sélectionner numériques
num_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()

# - imputer (ex : 5% NaN simulation si souhaité)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[num_cols] = imputer.fit_transform(df[num_cols])

# 5. Si besoin de target (ex : croissance > médiane -> classification)
# WARNING : créer une target docteur-sceptique (expliciter le délai et la définition)
df['target_growth'] = (df['some_indicator'].pct_change() > df['some_indicator'].pct_change().median()).astype(int)

# 6. Split / Model (exemple RandomForest)
X = df[num_cols].dropna()
y = df['target_growth'].loc[X.index]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

3. Analyse Approfondie : Nettoyage (Data Wrangling)
Le Problème du « vide » (NaN) et des séries temporelles

Les colonnes financières ou de comptage peuvent contenir des ruptures et des zéros valides. Il faut distinguer 0 réel et NaN manquant.

Les séries temporelles exigent un alignement (par année/institution/pays). Des données manquantes pourraient signifier absence réelle d’activité (0) ou simplement non-report.

Stratégies d’imputation recommandées

Imputation simple (moyenne/median) — OK pour un prototype, mais attention au data leakage :

Ne jamais calculer l’imputer (moyenne) sur l’ensemble si tu vas ensuite évaluer sur un jeu test : séparer d’abord.

Imputation temporelle : forward/backward fill pour séries temporelles d’un même pays/institution.

Modèles d’imputation : KNN imputer, modèles basés sur séries temporelles (ARIMA/ETS) pour prédire valeurs manquantes historiquement.

Le coin de l’expert

Pour indicateurs financiers (levées), préfère la méthode robuste (median) en présence d’outliers extrêmes.

Garde une colonne flag_missing_ par variable importante : utile pour le modèle (indique qu'une valeur était manquante).

4. Analyse Approfondie : Exploration (EDA)
Profiling et questions clés

Tendances temporelles : la série croît-elle linéairement, exponentiellement, ou par paliers ?

Saisonnalité : pour des données trimestrielles, y a-t-il des motifs récurrents ?

Concentration : la distribution des investissements est-elle très inégale (Gini élevé) ?

Corrélations : publications vs citations vs brevets vs financement — quels liens ?

Interpréter .describe() efficacement

Mean vs Median : distance → skew. Les montants d'investissement sont souvent très skewed.

Std : variables à variance nulle ou quasi-nulle peuvent être supprimées.

Outliers : boxplots par pays/institution pour repérer « whales » (acteurs dominants).

Multicolinéarité

Attends-toi à des corrélations fortes (ex : nombre de publications et citations). Pour des modèles linéaires, appliquer PCA ou sélectionner features. Pour Random Forest, moins critique.

5. Méthodologie : Split et Séries Temporelles
Cas d’usage et protocole

Si tu fais de la prédiction temporelle (prévoir le montant d’investissements en t+1), ne pas faire de random shuffle : utiliser un split chronologique (train = années ≤ 2019, test = 2020+).

Si tu fais classification cross-sectional (prévoir si une institution aura forte croissance l’année suivante), alors train_test_split aléatoire peut être acceptable si tu contrôles la fuite d’information temporelle (feature leakage).

Paramètres pratiques

test_size=0.2, random_state=42 : bonne pratique pour reproductibilité.

Pour séries : validation par rolling window / time series cross validation (expanding window).

6. Focus Théorique : Choix d’algorithmes
Que choisir selon l’objectif ?

Exploration / insight (non-supervisé) : clustering (KMeans, DBSCAN) pour regrouper pays/institutions ; PCA pour réduire la dimension.

Prévision (séries temporelles) : ARIMA, Prophet, modèles LSTM/Temporal Fusion pour signaux complexes.

Classification (signal binaire créé) : RandomForest, Gradient Boosting (XGBoost/LightGBM). RandomForest est robuste aux outliers et multicolinéarité ; idéal pour un premier benchmark.

Interprétabilité : si tu dois expliquer aux décideurs, préférer modèles simples (logistic regression + shap/feature importances) ou utiliser SHAP/LIME sur modèles complexes.

Pourquoi Random Forest est un bon point de départ

Gère bien mélanges de variables.

Peu de pré-traitements requis.

Importance features fournie directement.

Bon pour baseline robuste avant tuning.

7. Analyse Approfondie : Évaluation
Matrice de confusion — adaptation métier

Si target = risque/promotion/forte croissance, les faux négatifs (ne pas détecter une zone à forte croissance) peuvent coûter opportunités manquées. Selon l’usage, prioriser Recall (sensibilité) ou Precision.

Pour prévision monétaire, regarder MAE/MAPE/RMSE plutôt que accuracy.

Métriques à privilégier

Classification déséquilibrée : Precision, Recall, F1, AUC-ROC.

Séries/Regression : MAE (robuste), RMSE (pénalise gros écarts), MAPE (sensible aux zéros).

Business metric : perte économique simulée (ex : coût faux positif vs faux négatif).

8. Recommandations Opérationnelles & Bonnes Pratiques

Documenter les sources : proviennent-elles de publications, bases publiques, rapports financiers ? Tenir la traçabilité.

Séparer d’abord, imputer ensuite : éviter le data leakage.

Garder des flags de missingness : parfois l’absence d’un rapport est signal.

Versionner les données et le code : DVC/Git + environnements (requirements.txt / conda).

Automatiser la surveillance : pipelines ETL + tests de qualité (contrôle de drift).

Interprétabilité : fournir SHAP plots et rapports simples aux non-techniques.

Validations robustes : cross-validation temporelle pour séries, tests d'out-of-time.

9. Cas Pratique : Exemple d’Interprétation (scénarios)

Détection d’alerte : si une région montre +200% d’investissements en R&D sur 1 an, vérifier source (acquisition, nouveau fonds) avant politique publique.

Concentration : si top 5 institutions attirent 80% des brevets, envisager politiques de redistribution (subventions, bourses).

Risque de données : changement de méthodologie de collecte d’année à année provoque des ruptures artificielles — ajuster pour comparabilité.

10. Conclusion synthétique

Le dataset AI Index est d’abord une source stratégique : son rôle principal est descriptif et décisionnel. La Data Science pour ce cas d’usage privilégie l’exploration, la détection de tendances, et les analyses de concentration plutôt que la seule compétition de modèles supervisés.

Points clés à retenir :

Clarifier la question métier avant de créer une target artificielle.

Traiter les séries temporelles correctement (split chronologique, imputation temporelle).

Documenter les choix et exposer l’incertitude (intervalle, scénario).

Commencer par des baselines robustes (Random Forest pour classification, MAE pour régression) puis complexifier si le gain est démontrable.

Préparer des livrables actionnables : dashboard clair, rapport d’anomalies, recommandations politiques/stratégiques.
