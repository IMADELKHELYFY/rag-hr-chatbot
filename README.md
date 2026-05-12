# 🤖 Assistant RH — Prototype RAG

Prototype d'assistant conversationnel RH basé sur l'architecture **RAG (Retrieval-Augmented Generation)**.

Développé dans le cadre d'une démonstration technique pour illustrer comment l'IA générative peut améliorer l'**Employee Experience** en permettant aux collaborateurs d'interroger les politiques RH en langage naturel.

---

## 🎯 Problème résolu

Un LLM classique peut **halluciner** des réponses sur des politiques internes qu'il ne connaît pas.  
RAG résout ce problème : le modèle répond **uniquement à partir des documents fournis**, pas de ses données d'entraînement.

```
Question utilisateur
       ↓
Recherche dans la base documentaire (FAISS)
       ↓
Top-3 extraits pertinents récupérés
       ↓
LLM génère une réponse fondée sur ces extraits
       ↓
Réponse fiable + sources affichées
```

---

## 🛠️ Stack technique

| Composant | Outil |
|-----------|-------|
| Interface | Streamlit |
| Orchestration LLM | LangChain |
| Modèle de langage | Gemini 1.5 Flash (Google) |
| Embeddings | Google Embedding-001 |
| Vector store | FAISS (local) |
| Langage | Python 3.10+ |

---

## 🚀 Installation et lancement

```bash
# 1. Cloner le repo
git clone https://github.com/votre-username/hr-rag-assistant
cd hr-rag-assistant

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application
streamlit run app.py
```

Obtenir une clé API Gemini gratuitement : https://aistudio.google.com

---

## 📁 Structure du projet

```
hr_rag_assistant/
├── app.py              # Application principale Streamlit
├── hr_policy.txt       # Base documentaire RH (fictive)
├── requirements.txt    # Dépendances Python
└── README.md           # Ce fichier
```

---

## 💡 Exemple de questions

- *"Combien de jours de congés par an ?"*
- *"Comment fonctionne le télétravail pour les alternants ?"*
- *"Quelle est la politique de formation et son budget ?"*
- *"Que se passe-t-il en cas d'arrêt maladie ?"*

---

## 🔧 Extensibilité

Pour adapter à un vrai contexte d'entreprise :
- Remplacer `hr_policy.txt` par de vrais documents PDF (ajouter `PyPDFLoader`)
- Ajouter une authentification utilisateur
- Connecter à une base documentaire existante (Confluence, SharePoint)
- Loguer les questions pour analyse et amélioration continue

---

## 👤 Auteur

**Imad El Khelyfy** — Étudiant Master Data & IA  
[LinkedIn](https://linkedin.com) · [GitHub](https://github.com)  
Projet réalisé dans le cadre d'une démarche d'apprentissage appliqué à l'IA pour les RH.
