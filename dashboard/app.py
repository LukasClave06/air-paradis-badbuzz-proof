from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Air Paradis - Dashboard preuve de concept",
    page_icon="✈️",
    layout="wide",
)


# -----------------------------
# Paths / Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

SAMPLE_PATH = PROCESSED_DIR / "sentiment140_sample_20k.csv"
RESULTS_PATH = PROCESSED_DIR / "model_comparison_results.csv"

API_URL = "https://lukas1.pythonanywhere.com/predict"


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_sample_data() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_PATH)
    df["target_label"] = df["target"].map({0: "Négatif", 1: "Positif"})
    df["text_length"] = df["text"].astype(str).str.len()
    df["word_count"] = df["text"].astype(str).str.split().str.len()
    return df


@st.cache_data
def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    metric_cols = ["accuracy", "precision", "recall", "f1"]
    for col in metric_cols:
        if col in df.columns:
            df[col] = df[col].round(4)
    return df


def metric_delta(results_df: pd.DataFrame, metric_name: str):
    if not {"model", metric_name}.issubset(results_df.columns):
        return None

    try:
        bert_val = float(results_df.loc[results_df["model"] == "BERT", metric_name].iloc[0])
        electra_val = float(results_df.loc[results_df["model"] == "ELECTRA-base", metric_name].iloc[0])
        return electra_val - bert_val
    except Exception:
        return None


def predict_with_api(text: str) -> dict:
    response = requests.post(
        API_URL,
        json={"text": text},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


# -----------------------------
# Load data
# -----------------------------
df_sample = load_sample_data()
results_df = load_results()

best_model_row = results_df.loc[results_df["f1"].idxmax()]
best_model_name = best_model_row["model"]
best_model_f1 = float(best_model_row["f1"])

bert_f1_global = float(results_df.loc[results_df["model"] == "BERT", "f1"].iloc[0])
electra_f1_global = float(results_df.loc[results_df["model"] == "ELECTRA-base", "f1"].iloc[0])
gain_percent_f1 = ((electra_f1_global - bert_f1_global) / bert_f1_global) * 100


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")
st.sidebar.markdown(
    "Ce dashboard présente la preuve de concept réalisée pour comparer **BERT** et **ELECTRA-base** sur la classification de sentiment de tweets."
)

selected_view = st.sidebar.radio(
    "Choisir une vue",
    [
        "Vue d'ensemble",
        "Exploration des données",
        "Comparaison des modèles",
        "Prédiction d'un tweet",
    ],
)

show_example_tweets = st.sidebar.checkbox("Afficher des exemples de tweets", value=False)
max_examples = 5
if show_example_tweets:
    max_examples = st.sidebar.slider("Nombre d'exemples", min_value=3, max_value=15, value=5)


# -----------------------------
# Header
# -----------------------------
st.title("✈️ Air Paradis - Dashboard de preuve de concept")
st.markdown(
    "Comparaison entre une baseline **BERT** et un modèle plus récent, **ELECTRA-base**, pour la détection du sentiment de tweets."
)

st.success(
    f"Modèle recommandé : **{best_model_name}** — meilleur F1-score observé : **{best_model_f1:.4f}**. "
    f"Par rapport à BERT, ELECTRA-base améliore le F1-score de **{gain_percent_f1:.2f}%**."
)


# -----------------------------
# View 1: Overview
# -----------------------------
if selected_view == "Vue d'ensemble":
    st.subheader("Vue d'ensemble du projet")
    st.write(
        "Cette vue résume la preuve de concept réalisée : un modèle de référence, BERT, "
        "est comparé à un modèle plus récent, ELECTRA-base, sur le même échantillon de tweets."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Tweets analysés", f"{len(df_sample):,}".replace(",", " "))
    with col2:
        bert_f1 = float(results_df.loc[results_df["model"] == "BERT", "f1"].iloc[0])
        st.metric("F1-score BERT", f"{bert_f1:.4f}")
    with col3:
        electra_f1 = float(results_df.loc[results_df["model"] == "ELECTRA-base", "f1"].iloc[0])
        st.metric("F1-score ELECTRA", f"{electra_f1:.4f}")
    with col4:
        delta_f1 = metric_delta(results_df, "f1")
        st.metric("Gain de F1", f"{electra_f1:.4f}", delta=f"{delta_f1:+.4f}" if delta_f1 is not None else None)

    st.subheader("Résumé de la démarche")
    st.write(
        "Le projet reprend le cas d'usage Air Paradis afin d'évaluer si un modèle Transformer plus récent "
        "permet d'améliorer la détection d'un sentiment négatif sur des tweets. La comparaison a été réalisée "
        "sur le même échantillon de données, avec les mêmes métriques d'évaluation."
    )

    st.subheader("Comparaison synthétique des métriques")
    results_long = results_df.melt(id_vars="model", var_name="metric", value_name="score")

    fig_metrics = px.bar(
        results_long,
        x="metric",
        y="score",
        color="model",
        barmode="group",
        text="score",
        title="Comparaison des métriques entre les deux modèles",
    )
    fig_metrics.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_metrics.update_layout(
        height=520,
        xaxis_title="Métrique",
        yaxis_title="Score",
        yaxis_range=[0.75, 0.9],
        legend_title_text="Modèle",
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    st.info(
        "Lecture rapide : plus le score est élevé, meilleure est la performance. "
        "Dans cette preuve de concept, ELECTRA-base obtient les meilleurs résultats sur l'ensemble des métriques suivies."
    )

    st.markdown(
        f"**Conclusion rapide :** ELECTRA-base est le modèle à retenir dans cette preuve de concept, "
        f"avec un F1-score supérieur de **{gain_percent_f1:.2f}%** à celui de BERT."
    )


# -----------------------------
# View 2: Data exploration
# -----------------------------
elif selected_view == "Exploration des données":
    st.subheader("Exploration des données")
    st.write(
        "Cette vue permet de mieux comprendre le jeu de données utilisé pour entraîner les modèles : "
        "répartition des classes, longueur des tweets et distribution du nombre de mots."
    )

    st.subheader("Répartition des classes")
    class_counts = (
        df_sample["target_label"].value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )

    fig_classes = px.bar(
        class_counts,
        x="sentiment",
        y="count",
        text="count",
        title="Nombre de tweets par sentiment",
    )
    fig_classes.update_traces(textposition="outside")
    fig_classes.update_layout(
        height=420,
        xaxis_title="Sentiment",
        yaxis_title="Nombre de tweets",
    )
    st.plotly_chart(fig_classes, use_container_width=True)

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Longueur des tweets")
        fig_length = px.histogram(
            df_sample,
            x="text_length",
            color="target_label",
            nbins=40,
            barmode="overlay",
            title="Distribution de la longueur des tweets",
        )
        fig_length.update_layout(
            height=450,
            xaxis_title="Nombre de caractères",
            yaxis_title="Nombre de tweets",
        )
        st.plotly_chart(fig_length, use_container_width=True)

    with right_col:
        st.subheader("Nombre de mots par tweet")
        fig_words = px.box(
            df_sample,
            x="target_label",
            y="word_count",
            points=False,
            title="Distribution du nombre de mots selon le sentiment",
        )
        fig_words.update_layout(
            height=450,
            xaxis_title="Sentiment",
            yaxis_title="Nombre de mots",
        )
        st.plotly_chart(fig_words, use_container_width=True)

    if show_example_tweets:
        st.subheader("Exemples de tweets")
        sentiment_filter = st.selectbox("Filtrer par sentiment", ["Tous", "Négatif", "Positif"])
        filtered = df_sample.copy()
        if sentiment_filter != "Tous":
            filtered = filtered[filtered["target_label"] == sentiment_filter]

        st.dataframe(
            filtered[["target_label", "text"]].head(max_examples),
            use_container_width=True,
        )


# -----------------------------
# View 3: Model comparison
# -----------------------------
elif selected_view == "Comparaison des modèles":
    st.subheader("Comparaison détaillée des modèles")
    st.write(
        "Cette vue met en évidence les écarts de performance entre la baseline BERT et le modèle récent ELECTRA-base "
        "à partir des principales métriques de classification."
    )

    st.subheader("Tableau comparatif")
    st.dataframe(results_df, use_container_width=True)

    st.subheader("Écart entre ELECTRA-base et BERT")
    deltas = []
    for metric in ["accuracy", "precision", "recall", "f1"]:
        delta = metric_delta(results_df, metric)
        if delta is not None:
            deltas.append({"metric": metric, "gain": delta})

    delta_df = pd.DataFrame(deltas)
    best_metric_row = delta_df.loc[delta_df["gain"].idxmax()]

    fig_delta = px.bar(
        delta_df,
        x="metric",
        y="gain",
        text="gain",
        title="Gain obtenu par ELECTRA-base par rapport à BERT",
    )
    fig_delta.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_delta.update_layout(
        height=450,
        xaxis_title="Métrique",
        yaxis_title="Gain",
    )
    st.plotly_chart(fig_delta, use_container_width=True)

    st.metric(
        "Meilleure progression observée",
        best_metric_row["metric"].upper(),
        delta=f"{best_metric_row['gain']:+.4f}",
    )

    st.subheader("Interprétation métier")
    st.write(
        "Dans ce cas d'usage, une amélioration du recall et du F1-score est particulièrement intéressante, "
        "car elle signifie que le modèle détecte mieux les tweets problématiques tout en gardant un bon niveau de précision. "
        "Cela permet de limiter à la fois les tweets négatifs non détectés et les alertes inutiles."
    )

    with st.expander("Conclusion de la preuve de concept"):
        st.write(
            "Les résultats obtenus montrent que ELECTRA-base améliore les performances par rapport à BERT sur le jeu de test. "
            "La méthode récente identifiée grâce à la veille apporte donc une amélioration mesurable sur cette tâche de classification de sentiment."
        )


# -----------------------------
# View 4: Prediction
# -----------------------------
elif selected_view == "Prédiction d'un tweet":
    st.subheader("Prédiction en direct")
    st.write(
        "Cette vue permet d'interroger l'API déployée sur le cloud afin d'obtenir une prédiction en temps réel. "
        "Le modèle utilisé côté backend est ELECTRA-base."
    )

    st.subheader("Tester un tweet")
    st.write(
        "Saisissez un tweet pour obtenir la prédiction du modèle déployé."
    )

    default_text = "The flight was delayed again and the customer service was terrible."
    user_text = st.text_area(
        "Texte du tweet",
        value=default_text,
        height=140,
        help="Entrez un texte suffisamment explicite pour permettre une prédiction de sentiment.",
    )

    if st.button("Lancer la prédiction", use_container_width=True):
        if not user_text.strip():
            st.warning("Veuillez saisir un tweet avant de lancer la prédiction.")
        else:
            try:
                with st.spinner("Appel de l'API en cours..."):
                    api_result = predict_with_api(user_text)

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Prédiction", api_result["pred_text"].capitalize())
                    st.metric("Label prédit", api_result["pred_label"])

                with col2:
                    st.metric("Probabilité négative", f"{api_result['proba_neg']:.2%}")
                    st.metric("Probabilité positive", f"{api_result['proba_pos']:.2%}")

                pred_df = pd.DataFrame(
                    [
                        {"classe": "Négatif", "probabilité": api_result["proba_neg"]},
                        {"classe": "Positif", "probabilité": api_result["proba_pos"]},
                    ]
                )

                fig_pred = px.bar(
                    pred_df,
                    x="classe",
                    y="probabilité",
                    text="probabilité",
                    title="Probabilités prédites par le modèle déployé",
                )
                fig_pred.update_traces(texttemplate="%{text:.2%}", textposition="outside")
                fig_pred.update_layout(
                    height=450,
                    yaxis_title="Probabilité",
                    xaxis_title="Classe",
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                if api_result["bad_buzz"]:
                    st.error("Le tweet est détecté comme un signal potentiel de bad buzz.")
                else:
                    st.success("Le tweet n'est pas détecté comme un signal de bad buzz.")

            except requests.exceptions.RequestException as e:
                st.error(f"Erreur lors de l'appel à l'API : {e}")
            except Exception as e:
                st.error(f"Une erreur inattendue est survenue : {e}")


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Dashboard Streamlit réalisé dans le cadre du projet DataSpace / OpenClassrooms. "
    "Les graphiques sont conçus pour rester lisibles, avec titres explicites, taille de texte suffisante et interprétation textuelle complémentaire."
)

st.caption(
    "Synthèse : ELECTRA-base est le modèle recommandé pour la suite du projet au vu des résultats obtenus sur le jeu de test."
)