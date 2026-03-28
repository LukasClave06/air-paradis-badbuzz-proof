# ✈️ Air Paradis – Détection de Bad Buzz (NLP & MLOps)

## 🎯 Objectif du projet

Ce projet vise à détecter automatiquement les tweets négatifs ("bad buzz") pour la compagnie fictive **Air Paradis**, en utilisant des modèles de traitement du langage naturel (NLP).

L’objectif est double :

* Construire un modèle performant de classification de sentiment
* Mettre en place une **architecture MLOps** pour le déploiement

---

## 🧠 Modèles développés

Deux approches ont été comparées :

### 1. Modèle baseline

* BERT (bert-base-uncased)
* Fine-tuning sur un dataset de tweets (Sentiment140)

### 2. Modèle avancé

* ELECTRA-base
* Modèle plus récent et plus performant
* Meilleur F1-score observé

👉 Le modèle **ELECTRA-base** est retenu pour la mise en production.

---

## 📊 Données

* Dataset : **Sentiment140**
* 1 600 000 tweets labellisés
* Prétraitement adapté selon le modèle :

  * Nettoyage léger pour Transformers (important pour la performance)

---

## ⚙️ Architecture du projet

```text
air-paradis-badbuzz-proof/
├── src/
│   ├── api/            # API Flask
│   ├── common/         # nettoyage texte
│   └── tests/          # tests unitaires
├── dashboard/          # dashboard Streamlit
├── data/               # données
├── models/             # (non versionné)
├── notebooks/          # exploration & entraînement
└── requirements.txt
```

---

## 🚀 Déploiement

### 🔹 API (Flask)

* Déployée sur **PythonAnywhere**
* Endpoint principal :

```bash
POST /predict
```

### Exemple :

```json
{
  "text": "This flight was terrible"
}
```

### Réponse :

```json
{
  "proba_neg": 0.99,
  "proba_pos": 0.01,
  "pred_label": 0,
  "bad_buzz": true
}
```

---

## ⚠️ Gestion des modèles (point important)

Les modèles **ne sont pas stockés sur GitHub** pour plusieurs raisons :

* Taille importante (>400 MB)
* Limite GitHub (100 MB par fichier)
* Bonnes pratiques MLOps

### 🔧 Solution mise en place

1. Nettoyage du checkpoint :

   * Suppression des fichiers inutiles :

     * optimizer
     * scheduler
     * états d'entraînement

2. Réduction de taille :

   * ~1.2 Go → ~400 Mo

3. Découpage du modèle :

   * fichiers de ~80 Mo

4. Upload sur le serveur

5. Reconstruction côté serveur :

```bash
cat checkpoint_part_*.zip > checkpoint-4000.zip
unzip checkpoint-4000.zip
```

👉 Cette approche permet de respecter les contraintes d’infrastructure.

---

## 🧪 Tests

* Tests unitaires avec `pytest`
* Mode spécial pour CI/CD :

```python
UNIT_TEST_MODE = True
```

👉 Permet de tester l’API sans charger le modèle

---

## 🔁 CI/CD

* GitHub Actions
* Installation des dépendances
* Lancement des tests

---

## 📊 Dashboard

* Développé avec **Streamlit**
* Permet :

  * comparaison BERT vs ELECTRA
  * visualisation des performances
  * test en direct

👉 Déploiement séparé (architecture recommandée)

---

## 🧠 Choix techniques

| Élément     | Choix                      |
| ----------- | -------------------------- |
| Backend     | Flask                      |
| NLP         | Transformers (HuggingFace) |
| Modèle      | ELECTRA                    |
| Dashboard   | Streamlit                  |
| Déploiement | PythonAnywhere             |
| CI/CD       | GitHub Actions             |

---

## 📌 Résultat

* Modèle performant (F1-score amélioré)
* API fonctionnelle en production
* Architecture MLOps complète

---

## 👨‍💻 Auteur

Projet réalisé dans le cadre de la formation
**Ingénieur IA – OpenClassrooms**

---
