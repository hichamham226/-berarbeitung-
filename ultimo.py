import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import re
import hashlib
from collections import defaultdict
import numpy as np
import pandas as pd
import streamlit as st

# Alle Imports am Anfang laden — damit Streamlit sie beim Start findet
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import Levenshtein
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import HashingVectorizer
import owlready2 as owllib
import tempfile
from pathlib import Path
from rdflib import Graph, RDF, RDFS, OWL, URIRef, Literal, Namespace
from rdflib.term import BNode
from oops_client import run_oops_analysis, compute_pitfall_score as compute_oops_pitfall_score
from pitfall_scanner import scan_pitfalls, compute_pitfall_score as compute_local_pitfall_score, IOF_DEFINITION_PROPERTIES

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
DC = Namespace("http://purl.org/dc/elements/1.1/")
DCTERMS = Namespace("http://purl.org/dc/terms/")
IOF_AV = Namespace("https://spec.industrialontologies.org/ontology/annotation/")

TEXT_PROPERTIES = [
    RDFS.label, RDFS.comment,
    SKOS.prefLabel, SKOS.altLabel, SKOS.definition, SKOS.example,
    DC.description, DCTERMS.description,
    IOF_AV.naturalLanguageDefinition, IOF_AV.explanatoryNote,
    IOF_AV.synonym, IOF_AV.abbreviation,
]

ENTITY_KINDS = [
    ("Klasse", OWL.Class),
    ("Object Property", OWL.ObjectProperty),
    ("Datatype Property", OWL.DatatypeProperty),
    ("Individuum", OWL.NamedIndividual),
]

VERKNUEPFUNGS_QUERY = """
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?p ?art WHERE {
  {
    ?p rdfs:domain ?dom .
    ?p rdfs:range  ?ran .
    <%(a)s> rdfs:subClassOf* ?dom .
    <%(b)s> rdfs:subClassOf* ?ran .
    BIND("domain/range" AS ?art)
  }
  UNION
  {
    <%(a)s> rdfs:subClassOf* ?super .
    ?super rdfs:subClassOf ?restriction .
    ?restriction a owl:Restriction ;
                 owl:onProperty ?p .
    {
      { ?restriction owl:someValuesFrom ?ziel }
      UNION
      { ?restriction owl:allValuesFrom  ?ziel }
      UNION
      { ?restriction owl:onClass        ?ziel }
    }
    <%(b)s> rdfs:subClassOf* ?ziel .
    BIND("Restriction" AS ?art)
  }
}
LIMIT 25
"""

try:
    import torch
    TORCH_IMPORT_ERROR = None
except Exception as exc:
    torch = None
    TORCH_IMPORT_ERROR = exc

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_IMPORT_ERROR = None
except Exception as exc:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc


def _split_camel_case(name):
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)


def _local_name(value):
    text = str(value)
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    if "/" in text:
        return text.rsplit("/", 1)[-1]
    return text


def _build_entity_index(graph):
    index = []
    for art, rdf_type in ENTITY_KINDS:
        for entity in graph.subjects(RDF.type, rdf_type):
            if not isinstance(entity, URIRef):
                continue
            name = _local_name(entity)
            labels, definitionen, weitere = [], [], []
            for prop in TEXT_PROPERTIES:
                for value in graph.objects(entity, prop):
                    if not isinstance(value, Literal):
                        continue
                    text = str(value).strip()
                    if not text:
                        continue
                    if prop in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
                        labels.append(text)
                    elif prop in (
                        RDFS.comment,
                        SKOS.definition,
                        DC.description,
                        DCTERMS.description,
                        IOF_AV.naturalLanguageDefinition,
                    ):
                        definitionen.append(text)
                    else:
                        weitere.append(text)
            volltext = " ".join([_split_camel_case(name)] + labels + definitionen + weitere)
            index.append({
                "iri": str(entity),
                "art": art,
                "name": name,
                "label": labels[0] if labels else _split_camel_case(name),
                "definition": definitionen[0] if definitionen else "",
                "text": volltext,
            })
    index.sort(key=lambda entry: entry["iri"])
    return index


def _keywords(cq_text: str):
    tokens = re.findall(r"[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß0-9_-]*", cq_text)
    stop_words = {
        "welche", "welcher", "welches", "was", "wie", "ist", "sind", "wird", "werden",
        "der", "die", "das", "ein", "eine", "einer", "einem", "einen", "und", "oder",
        "von", "für", "mit", "im", "in", "am", "an", "zu", "den", "dem", "des",
        "sollen", "können", "müssen", "sich", "dass", "diese", "dieser", "dieses"
    }
    cleaned = []
    for token in tokens:
        t = token.lower()
        if len(t) <= 2 or t in stop_words:
            continue
        t = re.sub(r"(ung|ungen|en|er|s)$", "", t)
        if t and t not in cleaned:
            cleaned.append(t)
    return cleaned[:6]


@st.cache_resource(show_spinner="Sentence-Transformer-Modell wird geladen...")
def _load_bert_model():
    if SentenceTransformer is not None:
        try:
            model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
            return {
                "backend": "bert",
                "model": model,
                "error": None,
            }
        except Exception as exc:
            bert_error = exc
    else:
        bert_error = SENTENCE_TRANSFORMERS_IMPORT_ERROR or TORCH_IMPORT_ERROR

    vectorizer = HashingVectorizer(
        n_features=768,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b"
    )
    return {
        "backend": "hashing",
        "vectorizer": vectorizer,
        "error": (
            "SentenceTransformer konnte nicht geladen werden. "
            f"Details: {bert_error}"
        ),
    }


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def _encode_texts(texts, encoder, batch_size=32, max_length=128):
    if not texts:
        return np.empty((0, 768), dtype=np.float32)
    if encoder is None:
        return np.empty((len(texts), 768), dtype=np.float32)
    if encoder["backend"] == "bert":
        model = encoder["model"]
        if hasattr(model, "encode"):
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(embeddings, dtype=np.float32)

        tokenizer = encoder.get("tokenizer")
        all_embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            with torch.no_grad():
                output = model(**encoded)
            embeddings = _mean_pooling(output, encoded["attention_mask"])
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings).astype(np.float32)

    return encoder["vectorizer"].transform(texts).toarray().astype(np.float32)


def _semantische_treffer(cq_text, index, embeddings, modell, schwelle=0.45, top_k=15):
    if not index or modell is None or embeddings is None:
        return []

    if embeddings.shape[0] != len(index):
        return []

    if modell.get("backend") != "bert":
        return []

    cq_vector = _encode_texts([cq_text], modell, batch_size=32)[0]
    treffer = []
    for eintrag, vektor in zip(index, embeddings):
        similarity = float(
            np.dot(cq_vector, vektor)
            / (np.linalg.norm(cq_vector) * np.linalg.norm(vektor) + 1e-12)
        )
        if similarity >= schwelle:
            treffer.append({**eintrag, "score": round(similarity, 4)})

    treffer.sort(key=lambda item: (-item["score"], item["iri"]))
    return treffer[:top_k]


def _lexikalische_treffer(cq_text, index, top_k=15):
    keys = _keywords(cq_text)
    if not keys:
        return []

    treffer = []
    for eintrag in index:
        haystack = eintrag["text"].lower()
        score = sum(3 for key in keys if key in haystack)
        if score > 0:
            treffer.append({**eintrag, "score": score})

    treffer.sort(key=lambda item: (-item["score"], item["iri"]))
    return treffer[:top_k]


def _combine_treffer(semantic_hits, lexical_hits):
    merged = {}
    for hit in semantic_hits:
        merged[hit["iri"]] = {**hit, "verfahren": "Semantisch"}
    for hit in lexical_hits:
        existing = merged.get(hit["iri"])
        if existing is None:
            merged[hit["iri"]] = {**hit, "verfahren": "Lexikalisch"}
        elif existing.get("score", -1.0) < hit.get("score", -1.0):
            merged[hit["iri"]] = {**hit, "verfahren": "Lexikalisch"}
    ordered = sorted(merged.values(), key=lambda item: (-item["score"], item["iri"]))
    return ordered


def _kandidaten_aus_treffern(hits, max_klassen=3, max_properties=3):
    """Teilt die Trefferliste in Klassen- und Property-Kandidaten auf."""
    klassen = [h for h in hits if h.get("art") == "Klasse"][:max_klassen]
    properties = [
        h for h in hits
        if h.get("art") in ("Object Property", "Datatype Property")
    ][:max_properties]
    return klassen, properties


def _pruefe_verknuepfung(graph, iri_a, iri_b):
    """Prüft, ob zwischen zwei Klassen eine Relation existiert."""
    relationen = []
    abfragen = []
    for start, ziel, richtung in ((iri_a, iri_b, "→"), (iri_b, iri_a, "←")):
        query = VERKNUEPFUNGS_QUERY % {"a": start, "b": ziel}
        abfragen.append(query)
        try:
            for row in graph.query(query):
                relationen.append({
                    "property": str(row[0]),
                    "art": str(row[1]),
                    "richtung": richtung,
                    "von": start,
                    "nach": ziel,
                })
        except Exception:
            continue

    einzigartig = {}
    for rel in relationen:
        schluessel = (rel["property"], rel["richtung"])
        einzigartig.setdefault(schluessel, rel)
    relationen = [einzigartig[k] for k in sorted(einzigartig)]

    return {
        "gefunden": bool(relationen),
        "relationen": relationen,
        "abfrage": abfragen[0],
    }


def _rows_from_hits(hits, modus):
    rows = []
    for index, hit in enumerate(hits, start=1):
        definition = hit.get("definition", "") or ""
        if len(definition) > 200:
            definition = definition[:197] + "..."
        rows.append({
            "Rang": index,
            "Name": hit.get("name", ""),
            "Art": hit.get("art", ""),
            "Label": hit.get("label", ""),
            "Ähnlichkeit/Score": round(float(hit.get("score", 0.0)), 4),
            "Definition": definition,
            "IRI": hit.get("iri", ""),
            "Verfahren": hit.get("verfahren", modus),
        })
    return rows


def load_ontology_resilient(tmp_path, source_name, world=None):
    """Load ontology and gracefully handle unreachable external owl:imports."""
    world = world or owllib.World()
    try:
        return world.get_ontology(tmp_path).load(), False, None
    except Exception as primary_exc:
        primary_msg = str(primary_exc)

        try:
            try:
                graph = Graph()
                graph.parse(tmp_path, format="xml")
            except Exception:
                with open(tmp_path, "rb") as f:
                    raw_bytes = f.read()

                text = raw_bytes.decode("utf-8", errors="replace")

                cleaned_text = re.sub(r"<owl:imports\b[^>]*/>", "", text, flags=re.IGNORECASE)
                cleaned_text = re.sub(
                    r"<owl:imports\b[^>]*>.*?</owl:imports>",
                    "",
                    cleaned_text,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                cleaned_text = re.sub(
                    r"owl:imports\s+<[^>]+>\s*[;.]",
                    "",
                    cleaned_text,
                    flags=re.IGNORECASE,
                )
                cleaned_text = re.sub(
                    r"<http://www\.w3\.org/2002/07/owl#imports>\s+<[^>]+>\s*[;.]",
                    "",
                    cleaned_text,
                    flags=re.IGNORECASE,
                )

                if cleaned_text == text:
                    raise primary_exc

                with tempfile.NamedTemporaryFile(delete=False, suffix=".owl") as cleaned_tmp:
                    cleaned_path = cleaned_tmp.name
                    cleaned_tmp.write(cleaned_text.encode("utf-8", errors="replace"))
            else:
                for subject, predicate, obj in list(graph.triples((None, OWL.imports, None))):
                    graph.remove((subject, predicate, obj))

                with tempfile.NamedTemporaryFile(delete=False, suffix=".owl") as cleaned_tmp:
                    cleaned_path = cleaned_tmp.name

                graph.serialize(cleaned_path, format="xml")

            try:
                onto = world.get_ontology(cleaned_path).load()
            finally:
                try:
                    os.unlink(cleaned_path)
                except OSError:
                    pass

            warn = (
                f"Externe Imports in {source_name} waren nicht erreichbar. "
                "Die Ontologie wurde lokal ohne owl:imports geladen. "
                f"Originalfehler: {primary_msg}"
            )
            return onto, True, warn
        except Exception as fallback_exc:
            raise primary_exc from fallback_exc

st.set_page_config(page_title="Ontologie Evaluator", layout="wide")

st.title("Ontologie Evaluator")
st.markdown("---")

st.sidebar.title("Navigation")
phase = st.sidebar.radio("Wähle eine Phase:", [
    "Phase 1: Vorbereitung",
    "Phase 2: Coverage Score",
    "Phase 3: Strukturanalyse",
    "Phase 4: Competency Questions",
    "Phase 5: Gesamtscore"
])

# ============================================================
if phase == "Phase 1: Vorbereitung":
# ============================================================
    st.header("Phase 1: Vorbereitung")

    # -----------------------------------------------
    # Schritt 1 — Ontologie(n) hochladen
    # -----------------------------------------------
    st.subheader("1. Ontologie(n) hochladen")

    # Prüfe, ob bereits Dateien gespeichert sind
    if "owl_files" in st.session_state and st.session_state["owl_files"]:
        st.success(f"{len(st.session_state['owl_files'])} Ontologie(n) bereits geladen:")
        for f in st.session_state["owl_files"]:
            st.write(f"• {f.name}")

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Dateien zurücksetzen"):
                st.session_state["owl_files"] = None
                st.session_state["corpus_text"] = None
                st.rerun()

        st.info("Oder lade neue Dateien, um die bisherigen zu ersetzen:")

    owl_files = st.file_uploader(
        "Lade deine OWL-Dateien hoch",
        type=["owl", "rdf", "xml"],
        accept_multiple_files=True
    )

    # -----------------------------------------------
    # Schritt 2 — Textkorpus hochladen
    # -----------------------------------------------
    st.subheader("2. Textkorpus hochladen")

    # Prüfe, ob bereits Korpus gespeichert ist
    if "corpus_text" in st.session_state and st.session_state["corpus_text"]:
        st.success("Korpus bereits geladen!")
        st.info("Oder lade einen neuen Korpus, um den bisherigen zu ersetzen:")

    corpus_file = st.file_uploader(
        "Lade deinen Referenzkorpus hoch (TXT)",
        type=["txt"]
    )

    # Dateien im Zwischenspeicher merken — speichern für alle Phasen
    if owl_files:
        st.success(f"{len(owl_files)} Ontologie(n) neu geladen!")
        for f in owl_files:
            st.write(f"• {f.name}")
        st.session_state["owl_files"] = owl_files

    if corpus_file:
        st.success(f"Korpus geladen: {corpus_file.name}")
        corpus_text = corpus_file.read().decode("utf-8")
        st.session_state["corpus_text"] = corpus_text

    # -----------------------------------------------
    # Schritt 3 — Competency Questions eingeben
    # CQs werden hier eingegeben und gespeichert
    # Phase 4 greift später darauf zurück
    # -----------------------------------------------
    st.markdown("---")
    st.subheader("3. Competency Questions eingeben")

    # Prüfe, ob bereits CQs gespeichert sind
    if "cq_liste" in st.session_state and st.session_state["cq_liste"]:
        st.success(f"{len(st.session_state['cq_liste'])} Competency Questions bereits gespeichert:")
        for i, cq in enumerate(st.session_state["cq_liste"], 1):
            st.write(f"{i}. {cq}")

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("CQs zurücksetzen"):
                st.session_state["cq_liste"] = None
                st.rerun()

        st.info("Oder gib neue Competency Questions ein, um die bisherigen zu ersetzen:")
    else:
        st.info("Gib mindestens 6 Fragen ein die eine geeignete Ontologie beantworten können muss. Eine Frage pro Zeile.")

    cq_text = st.text_area(
        "Competency Questions (eine pro Zeile)",
        height=200,
        placeholder="Beispiel:\nWelche Klassen repräsentieren Prozesse?\nWelche Relationen bestehen zwischen Komponenten?\n..."
    )

    if st.button("Competency Questions speichern"):
        if cq_text.strip():
            cq_liste = [cq.strip() for cq in cq_text.strip().split("\n") if cq.strip()]
            if len(cq_liste) < 6:
                st.error(f"Bitte mindestens 6 Fragen eingeben! Aktuell: {len(cq_liste)}")
            else:
                st.session_state["cq_liste"] = cq_liste
                st.success(f"{len(cq_liste)} Competency Questions gespeichert!")
                for i, cq in enumerate(cq_liste, 1):
                    st.write(f"{i}. {cq}")
        else:
            st.error("Bitte mindestens eine Competency Question eingeben!")

    # -----------------------------------------------
    # Schritt 4 — Gewichtung kalibrieren
    # Standardgewichtung nach Methodik: w1=0.35, w2=0.35, w3=0.30
    # -----------------------------------------------
    st.markdown("---")
    st.subheader("4. Gewichtung kalibrieren")
    st.info("Standardgewichtung: w1=0.35, w2=0.35, w3=0.30 — Summe muss 1.0 ergeben.")

    col1, col2, col3 = st.columns(3)
    with col1:
        w1 = st.number_input("w1 — Coverage Score (Phase 2)", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    with col2:
        w2 = st.number_input("w2 — Strukturscore (Phase 3)", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    with col3:
        w3 = st.number_input("w3 — CQ-Score (Phase 4)", min_value=0.0, max_value=1.0, value=0.30, step=0.05)

    gesamt = round(w1 + w2 + w3, 2)
    if gesamt == 1.0:
        st.success(f"Summe: {gesamt} — Gewichtung gültig!")
        st.session_state["w1"] = w1
        st.session_state["w2"] = w2
        st.session_state["w3"] = w3
    else:
        st.error(f"Summe: {gesamt} — Muss genau 1.0 ergeben!")

# ============================================================
# ============================================================
elif phase == "Phase 2: Coverage Score":
# ============================================================
    st.header("Phase 2: Domänenwissensanalyse (Coverage Score)")
    st.markdown("Basiert auf dem OCALM-Framework von Abad-Navarro et al. (2025)")

    if "owl_files" not in st.session_state or "corpus_text" not in st.session_state:
        st.warning("Bitte zuerst in Phase 1 die OWL-Datei und den Korpus hochladen.")
    else:
        owl_files = st.session_state["owl_files"]
        corpus_text = st.session_state["corpus_text"]

        # Einstellungen in der Sidebar
        st.sidebar.header("Phase 2 Einstellungen")
        sprache = st.sidebar.selectbox(
            "Sprache des Korpus",
            ["Deutsch", "Englisch", "Französisch", "Spanisch"]
        )
        max_terms = st.sidebar.slider(
            "Maximale Anzahl Korpus-Begriffe",
            min_value=50, max_value=2000, value=500, step=50
        )
        batch_size = st.sidebar.slider(
            "BERT Batch Size",
            min_value=4, max_value=64, value=16, step=4
        )
        include_properties = st.sidebar.checkbox(
            "Ontologie-Properties einbeziehen",
            value=True
        )

        if st.button("Coverage Score berechnen"):

            # NLTK Daten herunterladen
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)

            # -----------------------------------------------
            # Hilfsfunktionen
            # -----------------------------------------------

            def get_language_name(sprache):
                # Sprache auf NLTK-Namen mappen
                mapping = {
                    "Deutsch": "german",
                    "Englisch": "english",
                    "Französisch": "french",
                    "Spanisch": "spanish"
                }
                return mapping.get(sprache, "german")

            def clean_text(text):
                # Text bereinigen — Sonderzeichen entfernen
                # Umlaute und Akzente behalten
                text = text.lower()
                text = re.sub(r"http\S+|www\S+", " ", text)
                text = re.sub(r"[^a-zA-ZäöüÄÖÜßáàâéèêíìîóòôúùûñç\- ]", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                return text

            def preprocess_label(label, lang_name):
                # Ontologie-Label bereinigen und stemmen
                cleaned = clean_text(label)
                tokens = nltk.word_tokenize(cleaned)
                sw = set(stopwords.words(lang_name))
                stemmer = SnowballStemmer(lang_name)
                processed = []
                for token in tokens:
                    if len(token) < 2:
                        continue
                    if token in sw:
                        continue
                    processed.append(stemmer.stem(token))
                return " ".join(processed) if processed else cleaned

            # -----------------------------------------------
            # BERT Modell laden
            # Einmalig cachen damit es nicht bei jedem
            # Klick neu geladen werden muss
            # -----------------------------------------------
            def load_bert_model():
                return _load_bert_model()

            # -----------------------------------------------
            # Mean Pooling
            # Durchschnitt aller Token-Vektoren ergibt
            # einen einzigen Satzvektor
            # -----------------------------------------------
            def mean_pooling(model_output, attention_mask):
                token_embeddings = model_output.last_hidden_state
                input_mask_expanded = (
                    attention_mask
                    .unsqueeze(-1)
                    .expand(token_embeddings.size())
                    .float()
                )
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
                sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask

            # -----------------------------------------------
            # Texte vektorisieren
            # Batch-Verarbeitung für Speichereffizienz
            # -----------------------------------------------
            def encode_texts(texts, encoder, batch_size=16, max_length=128):
                if not texts:
                    return np.empty((0, 768), dtype=np.float32)
                if encoder["backend"] == "bert":
                    model = encoder["model"]
                    if hasattr(model, "encode"):
                        embeddings = model.encode(
                            texts,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                        )
                        return np.asarray(embeddings, dtype=np.float32)

                    # Fallback für andere Modell-Implementierungen
                    tokenizer = encoder.get("tokenizer")
                    all_embeddings = []
                    for start in range(0, len(texts), batch_size):
                        batch = texts[start:start + batch_size]
                        encoded = tokenizer(
                            batch,
                            padding=True,
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt"
                        )
                        with torch.no_grad():
                            output = model(**encoded)
                        embeddings = mean_pooling(output, encoded["attention_mask"])
                        all_embeddings.append(embeddings.cpu().numpy())
                    return np.vstack(all_embeddings).astype(np.float32)

                # Robuster Fallback ohne torch/transformers, falls Import-Konflikte bestehen.
                sparse_matrix = encoder["vectorizer"].transform(texts)
                return sparse_matrix.toarray().astype(np.float32)

            # -----------------------------------------------
            # STRANG 1: Textkorpus verarbeiten
            # Schritt 1.1 — Textvorverarbeitung mit NLTK
            # Stoppwörter entfernen, Stemming
            # -----------------------------------------------
            with st.spinner("Schritt 1.1 — Textvorverarbeitung..."):

                lang_name = get_language_name(sprache)
                sw = set(stopwords.words(lang_name))
                stemmer = SnowballStemmer(lang_name)

                cleaned_corpus = clean_text(corpus_text)
                tokens = nltk.word_tokenize(cleaned_corpus)

                # Begriffe extrahieren und stemmen
                rows = []
                for token in tokens:
                    token = token.strip().lower()
                    if len(token) < 3:
                        continue
                    if token in sw:
                        continue
                    if not token.replace("-", "").isalpha():
                        continue
                    stemmed = stemmer.stem(token)
                    if len(stemmed) < 3:
                        continue
                    rows.append({
                        "original_token": token,
                        "stemmed_term": stemmed
                    })

                if not rows:
                    st.error("Keine verwertbaren Begriffe im Korpus gefunden.")
                    st.stop()

                # Häufigkeit der Begriffe zählen
                terms_df = pd.DataFrame(rows)
                term_freq = (
                    terms_df.groupby("stemmed_term")
                    .agg(
                        frequency=("stemmed_term", "count"),
                        beispiele=("original_token", lambda x: ", ".join(sorted(set(x))[:3]))
                    )
                    .reset_index()
                    .sort_values("frequency", ascending=False)
                    .head(max_terms)
                )

                noun_phrases = term_freq["stemmed_term"].tolist()
                st.success(f"Schritt 1.1 abgeschlossen — {len(noun_phrases)} Begriffe extrahiert")

            # -----------------------------------------------
            # BERT Modell laden
            # -----------------------------------------------
            with st.spinner("Sentence-Transformer-Modell laden..."):
                encoder = load_bert_model()
                if encoder["backend"] == "bert":
                    st.success("Sentence-Transformer-Modell geladen!")
                else:
                    st.warning(
                        "Sentence-Transformer-Modell konnte nicht geladen werden. Es wird ein stabiler Text-Fallback genutzt. "
                        f"Details: {encoder['error']}"
                    )

            alle_coverage_scores = {}

            for owl_file in owl_files:
                st.subheader(f"Ontologie: {owl_file.name}")

                # -----------------------------------------------
                # Schritt 1.2 — Vektorisierung Korpus mit BERT
                # -----------------------------------------------
                with st.spinner("Schritt 1.2 — Vektorisierung der Korpus-Begriffe mit dem Sentence-Transformer..."):
                    corpus_vectors = encode_texts(
                        noun_phrases, encoder, batch_size=batch_size
                    )
                    st.success(f"Schritt 1.2 abgeschlossen — {len(corpus_vectors)} Vektoren erzeugt")

                # -----------------------------------------------
                # STRANG 2: Ontologie verarbeiten
                # Schritt 2.1 — Klassenextraktion aus OWL-Datei
                # Labels, Properties, URI-Namen
                # -----------------------------------------------
                with st.spinner("Schritt 2.1 — Ontologieklassen extrahieren..."):
                    suffix = Path(owl_file.name).suffix.lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(owl_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        onto, used_import_fallback, import_warn = load_ontology_resilient(
                            tmp_path, owl_file.name
                        )
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

                    if used_import_fallback and import_warn:
                        st.warning(import_warn)

                    onto_rows = []

                    # Klassen extrahieren
                    for cls in onto.classes():
                        labels = list(cls.label)
                        label = str(labels[0]) if labels else cls.name.replace("_", " ")
                        onto_rows.append({
                            "entity_type": "Class",
                            "entity_name": cls.name,
                            "label": label
                        })

                    # Properties optional einbeziehen
                    if include_properties:
                        for prop in onto.object_properties():
                            labels = list(prop.label)
                            label = str(labels[0]) if labels else prop.name.replace("_", " ")
                            onto_rows.append({
                                "entity_type": "ObjectProperty",
                                "entity_name": prop.name,
                                "label": label
                            })
                        for prop in onto.data_properties():
                            labels = list(prop.label)
                            label = str(labels[0]) if labels else prop.name.replace("_", " ")
                            onto_rows.append({
                                "entity_type": "DataProperty",
                                "entity_name": prop.name,
                                "label": label
                            })

                    if not onto_rows:
                        st.warning(f"Keine Klassen in {owl_file.name} gefunden.")
                        continue

                    onto_df = pd.DataFrame(onto_rows).drop_duplicates()

                    # Labels vorverarbeiten
                    onto_df["processed_label"] = onto_df["label"].apply(
                        lambda x: preprocess_label(x, lang_name)
                    )

                    ontology_labels = onto_df["processed_label"].tolist()
                    st.success(f"Schritt 2.1 abgeschlossen — {len(ontology_labels)} Labels extrahiert")

                # -----------------------------------------------
                # Schritt 2.2 — Vektorisierung Ontologie mit BERT
                # -----------------------------------------------
                with st.spinner("Schritt 2.2 — Vektorisierung der Ontologie-Labels mit dem Sentence-Transformer..."):
                    onto_vectors = encode_texts(
                        ontology_labels, encoder, batch_size=batch_size
                    )
                    st.success(f"Schritt 2.2 abgeschlossen — {len(onto_vectors)} Vektoren erzeugt")

                # -----------------------------------------------
                # ABGLEICH: OCALM Score-Funktion
                # Schritt 3 — Levenshtein + Cosine Similarity
                # Formel: Score_i = 0.3 * Levenshtein + 0.7 * Cosine
                # -----------------------------------------------
                with st.spinner("Schritt 3 — OCALM Score-Funktion berechnen..."):
                    cosine_matrix = cosine_similarity(onto_vectors, corpus_vectors)

                    result_rows = []

                    for i, ont_row in onto_df.iterrows():
                        processed_label = str(ont_row["processed_label"])
                        best_score = -1
                        best_term = None
                        best_cosine = None
                        best_levenshtein = None

                        for j, corpus_term in enumerate(noun_phrases):
                            # Cosine Similarity aus Matrix
                            cosine_val = float(cosine_matrix[i, j])
                            cosine_val = max(0.0, min(1.0, cosine_val))

                            # Levenshtein Ähnlichkeit
                            lev_val = Levenshtein.ratio(processed_label, corpus_term)

                            # Kombinierter Score nach OCALM
                            score_i = 0.3 * lev_val + 0.7 * cosine_val

                            if score_i > best_score:
                                best_score = score_i
                                best_term = corpus_term
                                best_cosine = cosine_val
                                best_levenshtein = lev_val

                        result_rows.append({
                            "Entity Type": ont_row["entity_type"],
                            "Ontologie Klasse": ont_row["entity_name"],
                            "Ontologie Label": ont_row["label"],
                            "Bester Korpus-Begriff": best_term,
                            "Levenshtein": round(best_levenshtein, 4),
                            "Cosine": round(best_cosine, 4),
                            "Score_i": round(best_score, 4),
                            "Score_i (%)": round(best_score * 100, 2)
                        })

                # -----------------------------------------------
                # Schritt 4 — Coverage Score berechnen
                # Coverage Score = Mittelwert aller Score_i * 100
                # -----------------------------------------------
                results_df = pd.DataFrame(result_rows).sort_values(
                    "Score_i", ascending=False
                ).reset_index(drop=True)

                coverage_score = round(results_df["Score_i"].mean() * 100, 2)
                alle_coverage_scores[owl_file.name] = coverage_score

                # Ergebnis speichern für Phase 5
                if "scores" not in st.session_state:
                    st.session_state["scores"] = {}
                if owl_file.name not in st.session_state["scores"]:
                    st.session_state["scores"][owl_file.name] = {}
                st.session_state["scores"][owl_file.name]["coverage_score"] = coverage_score

                # -----------------------------------------------
                # ERGEBNISANZEIGE
                # -----------------------------------------------
                st.markdown("---")
                st.subheader(f"Ergebnisse: {owl_file.name}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Coverage Score", f"{coverage_score} / 100")
                with col2:
                    st.metric("Ontologie Entitäten", len(ontology_labels))
                with col3:
                    st.metric("Korpus Begriffe", len(noun_phrases))

                # Harvey-Ball Bewertung
                if coverage_score >= 80:
                    st.success("● ● ● ● ● Sehr gut — Hohe Domänenabdeckung")
                elif coverage_score >= 60:
                    st.success("● ● ● ● ○ Gut — Ausreichende Domänenabdeckung")
                elif coverage_score >= 40:
                    st.warning("● ● ● ○ ○ Mittel — Teilweise Domänenabdeckung")
                elif coverage_score >= 20:
                    st.warning("● ● ○ ○ ○ Schwach — Geringe Domänenabdeckung")
                else:
                    st.error("● ○ ○ ○ ○ Nicht erfüllt — Kaum Domänenabdeckung")

                # Top 20 beste Matches
                st.subheader("Top 20 — Beste Übereinstimmungen")
                st.dataframe(results_df.head(20), use_container_width=True)

                # Bottom 10 schlechteste Matches
                st.subheader("Bottom 10 — Schlechteste Übereinstimmungen")
                st.dataframe(results_df.tail(10), use_container_width=True)

                # Score-Verteilung
                st.subheader("Score-Verteilung")
                score_werte = results_df["Score_i"].tolist()
                verteilung_df = pd.DataFrame({
                    "Bereich": ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
                    "Anzahl Entitäten": [
                        sum(1 for s in score_werte if 0.0 <= s < 0.2),
                        sum(1 for s in score_werte if 0.2 <= s < 0.4),
                        sum(1 for s in score_werte if 0.4 <= s < 0.6),
                        sum(1 for s in score_werte if 0.6 <= s < 0.8),
                        sum(1 for s in score_werte if 0.8 <= s <= 1.0),
                    ]
                })
                st.dataframe(verteilung_df, use_container_width=True)

                # CSV Download
                csv = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Ergebnisse als CSV herunterladen",
                    data=csv,
                    file_name=f"coverage_{owl_file.name}.csv",
                    mime="text/csv"
                )
                st.markdown("---")

            # -----------------------------------------------
            # VERGLEICH ALLER ONTOLOGIEN
            # -----------------------------------------------
            if len(alle_coverage_scores) > 1:
                st.subheader("Vergleich aller Ontologien — Coverage Score")
                vergleich_df = pd.DataFrame({
                    "Ontologie": list(alle_coverage_scores.keys()),
                    "Coverage Score": list(alle_coverage_scores.values())
                }).sort_values("Coverage Score", ascending=False).reset_index(drop=True)
                vergleich_df.index += 1
                st.dataframe(vergleich_df, use_container_width=True)
                beste_onto = vergleich_df.iloc[0]["Ontologie"]
                bester_score = vergleich_df.iloc[0]["Coverage Score"]
                st.success(f"Beste Domänenabdeckung: {beste_onto} mit {bester_score}/100")
# ============================================================
elif phase == "Phase 3: Strukturanalyse":
# ============================================================
    st.header("Phase 3: Strukturelle Analyse")
    st.markdown(
        "Phase 3 bewertet die Ontologie-Struktur automatisch anhand der hochgeladenen OWL-Dateien."
    )

    # Kleine Hilfe: Metriken kurz erklären (öffnet ein Fenster/Expander beim Klick)
    def metrics_help_text():
        return """
- **Anzahl Klassen**: Anzahl definierter Klassen — zeigt Umfang.
- **Object/Data/Annotation Properties**: Beziehungen und Attribute — Modellierungsbreite.
- **Axiome**: Logische Aussagen — Ausdruckskraft der Ontologie.
- **Relationship Richness (RR)**: Anteil Objekt-Relationen an allen Properties — mehr Beziehungen = reichere Beziehungen.
- **Inheritance Richness (IR)**: Durchschnitt Eltern pro Klasse — Tiefe/Wiederverwendung der Hierarchie.
- **Attribute Richness (AR)**: Daten-Attribute pro Klasse — Beschreibungsgrad der Klassen.
- **Axiom/Class Ratio (ACR)**: Axiome pro Klasse — modellseitige Komplexität.
- **Average Depth / Breadth**: Tiefe und Verzweigung der Hierarchie — Strukturgranularität.
- **Tangledness**: Anteil Mehrfachvererbung — Komplexität und mögliche Ambiguität.
- **Annotation Richness**: Labels/Kommentare pro Klasse — Dokumentationsqualität.
- **Pitfall Score (OOPS!)**: Externe Qualitätsprüfung auf bekannte Fallstricke (critical/major/minor).
- **Konsistenz**: Logische Widerspruchsfreiheit (Reasoner) — K.O.-Kriterium für Korrektheit.
- **Schema / Topologie / Strukturscore**: Zusammengesetzte Indikatoren, die die obenstehenden Aspekte auf 0–100 normieren.

Kurze Bedeutung: Höhere Werte zeigen in der Regel bessere Modellierungsqualität (klarere Struktur, weniger Fallstricke, gute Dokumentation)."""

    if "show_metrics_help" not in st.session_state:
        st.session_state["show_metrics_help"] = False
    if st.button("Was bedeuten die Metriken?— Kurz erklärt"):
        st.session_state["show_metrics_help"] = True
    if st.session_state.get("show_metrics_help"):
        with st.expander("Metriken — Kurz erklärt", expanded=True):
            st.markdown(metrics_help_text())
            if st.button("Schließen", key="close_metrics_help"):
                st.session_state["show_metrics_help"] = False

    if "owl_files" not in st.session_state:
        st.warning("Bitte zuerst in Phase 1 die OWL-Datei hochladen.")
    else:
        owl_files = st.session_state["owl_files"]
        if "scores" not in st.session_state:
            st.session_state["scores"] = {}

        def normalize_value(value, min_value, max_value, invert=False):
            if invert:
                norm = (1.0 - value) * 100.0
            else:
                norm = (value - min_value) / (max_value - min_value) * 100.0
            return float(np.clip(norm, 0.0, 100.0))

        pitfall_verfahren = st.radio(
            "Pitfall-Prüfung",
            options=["OOPS! (extern)", "Lokaler Pitfall-Scanner (deterministisch)"],
            horizontal=True,
            key="pitfall_verfahren",
            help=(
                "OOPS! prüft 21 Pitfalls des offiziellen Katalogs über den Webservice "
                "der UPM Madrid und benötigt eine Internetverbindung. Der lokale "
                "Scanner prüft 21 strukturell entscheidbare Pitfalls ohne "
                "Netzwerkzugriff und liefert bei identischer Datei immer identische "
                "Ergebnisse. Die geprüften Mengen überschneiden sich, sind aber "
                "nicht deckungsgleich."
            ),
        )

        iof_annotationen = False
        if pitfall_verfahren == "Lokaler Pitfall-Scanner (deterministisch)":
            iof_annotationen = st.checkbox(
                "IOF-Annotationen als Definition anerkennen (P08)",
                value=False,
                key="pitfall_iof_annotationen",
                help=(
                    "IOF-Ontologien dokumentieren ihre Klassen mit "
                    "iof-av:naturalLanguageDefinition statt mit rdfs:comment. Ist diese "
                    "Option aktiv, wertet der Scanner diese Annotationen als Definition "
                    "und meldet P08 nicht mehr fälschlich."
                ),
            )

        def get_ontology_metrics_from_graph(graph, ontology_path):
            def _parse(source):
                if isinstance(source, Graph):
                    return source
                if source is None:
                    return None
                parsed_graph = Graph()
                try:
                    parsed_graph.parse(source, format="xml")
                except Exception:
                    try:
                        parsed_graph.parse(source)
                    except Exception:
                        return None
                return parsed_graph

            if isinstance(graph, Graph):
                rdf_graph = graph
            elif hasattr(graph, "world") and hasattr(graph.world, "as_rdflib_graph"):
                try:
                    rdf_graph = graph.world.as_rdflib_graph()
                except Exception:
                    rdf_graph = None
            elif hasattr(graph, "as_rdflib_graph"):
                try:
                    rdf_graph = graph.as_rdflib_graph()
                except Exception:
                    rdf_graph = None
            else:
                rdf_graph = _parse(graph)

            if rdf_graph is None and ontology_path:
                rdf_graph = _parse(ontology_path)

            if rdf_graph is None:
                return {
                    "num_classes": 0,
                    "num_object_props": 0,
                    "num_data_props": 0,
                    "num_annotation_props": 0,
                    "num_individuals": 0,
                    "num_subclass_relations": 0,
                    "num_restrictions": 0,
                    "num_triples": 0,
                    "num_axioms": 0,
                    "num_hierarchy_roots": 0,
                    "num_hierarchy_levels": 0,
                    "max_depth": 0,
                    "max_breadth": 0,
                    "relationship_richness": 0.0,
                    "relationship_richness_used": 0.0,
                    "inheritance_richness": 0.0,
                    "attribute_richness": 0.0,
                    "axiom_class_ratio": 0.0,
                    "average_depth": 0.0,
                    "average_breadth": 0.0,
                    "tangledness": 0.0,
                    "annotation_richness": 0.0,
                }

            BUILTIN_ANNOTATION_PROPERTIES = {
                RDFS.label,
                RDFS.comment,
                RDFS.seeAlso,
                RDFS.isDefinedBy,
                OWL.versionInfo,
                OWL.deprecated,
            }
            MAX_PATHS = 200_000

            classes = {s for s in rdf_graph.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}
            object_properties = {s for s in rdf_graph.subjects(RDF.type, OWL.ObjectProperty) if isinstance(s, URIRef)}
            data_properties = {s for s in rdf_graph.subjects(RDF.type, OWL.DatatypeProperty) if isinstance(s, URIRef)}
            annotation_properties = {s for s in rdf_graph.subjects(RDF.type, OWL.AnnotationProperty) if isinstance(s, URIRef)}
            individuals = {s for s in rdf_graph.subjects(RDF.type, OWL.NamedIndividual) if isinstance(s, URIRef)}
            entities = classes | object_properties | data_properties | annotation_properties | individuals

            num_classes = len(classes)
            if num_classes == 0:
                return {
                    "num_classes": 0,
                    "num_object_props": 0,
                    "num_data_props": 0,
                    "num_annotation_props": 0,
                    "num_individuals": 0,
                    "num_subclass_relations": 0,
                    "num_restrictions": 0,
                    "num_triples": len(rdf_graph),
                    "num_axioms": 0,
                    "num_hierarchy_roots": 0,
                    "num_hierarchy_levels": 0,
                    "max_depth": 0,
                    "max_breadth": 0,
                    "relationship_richness": 0.0,
                    "relationship_richness_used": 0.0,
                    "inheritance_richness": 0.0,
                    "attribute_richness": 0.0,
                    "axiom_class_ratio": 0.0,
                    "average_depth": 0.0,
                    "average_breadth": 0.0,
                    "tangledness": 0.0,
                    "annotation_richness": 0.0,
                }

            parents = defaultdict(set)
            children = defaultdict(set)
            nodes = set(classes)

            for sub, sup in rdf_graph.subject_objects(RDFS.subClassOf):
                if not (isinstance(sub, URIRef) and isinstance(sup, URIRef)):
                    continue
                if sup == OWL.Thing:
                    continue
                parents[sub].add(sup)
                children[sup].add(sub)
                nodes.add(sub)
                nodes.add(sup)

            num_subclass_relations = sum(len(v) for v in parents.values())

            num_restrictions = sum(
                1 for r in rdf_graph.subjects(RDF.type, OWL.Restriction)
                if list(rdf_graph.objects(r, OWL.onProperty))
            )

            p_declared = len(object_properties)
            p_used = p_declared + num_restrictions

            denom_declared = num_subclass_relations + p_declared
            denom_used = num_subclass_relations + p_used

            relationship_richness = p_declared / denom_declared if denom_declared else 0.0
            relationship_richness_used = p_used / denom_used if denom_used else 0.0

            inheritance_richness = num_subclass_relations / num_classes
            attribute_richness = len(data_properties) / num_classes

            declaration_axioms = len(entities)
            subclass_axioms = sum(1 for s, _ in rdf_graph.subject_objects(RDFS.subClassOf) if isinstance(s, URIRef))
            equivalence_axioms = len(list(rdf_graph.subject_objects(OWL.equivalentClass)))
            disjointness_axioms = (
                len(list(rdf_graph.subject_objects(OWL.disjointWith)))
                + len(list(rdf_graph.subjects(RDF.type, OWL.AllDisjointClasses)))
            )

            property_axioms = 0
            for predicate in (
                RDFS.subPropertyOf,
                RDFS.domain,
                RDFS.range,
                OWL.inverseOf,
                OWL.equivalentProperty,
                OWL.propertyDisjointWith,
            ):
                property_axioms += len(list(rdf_graph.subject_objects(predicate)))
            for characteristic in (
                OWL.FunctionalProperty,
                OWL.InverseFunctionalProperty,
                OWL.TransitiveProperty,
                OWL.SymmetricProperty,
                OWL.AsymmetricProperty,
                OWL.ReflexiveProperty,
                OWL.IrreflexiveProperty,
            ):
                property_axioms += len(list(rdf_graph.subjects(RDF.type, characteristic)))

            assertion_axioms = sum(1 for s, p, _ in rdf_graph if s in individuals and p != RDF.type)
            annotation_axioms_total = sum(1 for s, _, o in rdf_graph if isinstance(o, Literal) and s in entities)
            annotation_axioms_classes = sum(1 for s, _, o in rdf_graph if isinstance(o, Literal) and s in classes)

            logical_axioms = (
                subclass_axioms
                + equivalence_axioms
                + disjointness_axioms
                + property_axioms
                + assertion_axioms
            )
            num_axioms = declaration_axioms + logical_axioms + annotation_axioms_total
            axiom_class_ratio = num_axioms / num_classes

            roots = sorted((n for n in nodes if not parents[n]), key=str)
            path_lengths = []

            def walk(node, depth, visited):
                if len(path_lengths) > MAX_PATHS:
                    return
                kids = children.get(node, set())
                if not kids:
                    path_lengths.append(depth)
                    return
                for kid in sorted(kids, key=str):
                    if kid in visited:
                        path_lengths.append(depth)
                        continue
                    walk(kid, depth + 1, visited | {kid})

            for root in roots:
                walk(root, 1, {root})

            average_depth = sum(path_lengths) / len(path_lengths) if path_lengths else 0.0
            max_depth = max(path_lengths) if path_lengths else 0

            level_of = {}
            frontier = set(roots)
            level = 1
            while frontier:
                for node in frontier:
                    level_of.setdefault(node, level)
                next_frontier = {
                    kid
                    for node in frontier
                    for kid in children.get(node, set())
                    if kid not in level_of
                }
                frontier = next_frontier
                level += 1

            nodes_per_level = defaultdict(int)
            for lvl in level_of.values():
                nodes_per_level[lvl] += 1

            average_breadth = (
                sum(nodes_per_level.values()) / len(nodes_per_level)
                if nodes_per_level else 0.0
            )
            max_breadth = max(nodes_per_level.values()) if nodes_per_level else 0

            tangled_count = sum(1 for n in nodes if len(parents[n]) > 1)
            tangledness = tangled_count / len(nodes) if nodes else 0.0

            annotation_richness = annotation_axioms_classes / num_classes

            return {
                "num_classes": num_classes,
                "num_object_props": len(object_properties),
                "num_data_props": len(data_properties),
                "num_annotation_props": len(annotation_properties),
                "num_individuals": len(individuals),
                "num_subclass_relations": num_subclass_relations,
                "num_restrictions": num_restrictions,
                "num_triples": len(rdf_graph),
                "num_axioms": num_axioms,
                "num_hierarchy_roots": len(roots),
                "num_hierarchy_levels": len(nodes_per_level),
                "max_depth": max_depth,
                "max_breadth": max_breadth,
                "relationship_richness": relationship_richness,
                "relationship_richness_used": relationship_richness_used,
                "inheritance_richness": inheritance_richness,
                "attribute_richness": attribute_richness,
                "axiom_class_ratio": axiom_class_ratio,
                "average_depth": average_depth,
                "average_breadth": average_breadth,
                "tangledness": tangledness,
                "annotation_richness": annotation_richness,
            }

        def get_ontology_metrics(onto, ontology_path=None):
            return get_ontology_metrics_from_graph(onto, ontology_path)

        # -----------------------------------------------
        # Automatische OOPS! und Konsistenzprüfung
        # -----------------------------------------------
        import json

        STRUCTURE_SCORE_CACHE_VERSION = 8
        STRUCTURE_SCORE_CACHE_PATH = Path(".struktur_score_cache.json")

        def load_structure_score_cache():
            if "structure_score_cache" in st.session_state:
                cached = st.session_state["structure_score_cache"]
                if isinstance(cached, dict):
                    if any(
                        isinstance(entry, dict) and entry.get("cache_version") == STRUCTURE_SCORE_CACHE_VERSION
                        for entry in cached.values()
                    ):
                        return cached
                    st.session_state.pop("structure_score_cache", None)

            if STRUCTURE_SCORE_CACHE_PATH.exists():
                try:
                    cache_data = json.loads(
                        STRUCTURE_SCORE_CACHE_PATH.read_text(encoding="utf-8")
                    )
                    if isinstance(cache_data, dict):
                        filtered_cache = {
                            key: entry
                            for key, entry in cache_data.items()
                            if isinstance(entry, dict)
                            and entry.get("cache_version") == STRUCTURE_SCORE_CACHE_VERSION
                        }
                        st.session_state["structure_score_cache"] = filtered_cache
                        return filtered_cache
                except Exception:
                    pass

            st.session_state["structure_score_cache"] = {}
            return st.session_state["structure_score_cache"]

        def save_structure_score_cache(cache_data):
            st.session_state["structure_score_cache"] = cache_data
            try:
                STRUCTURE_SCORE_CACHE_PATH.write_text(
                    json.dumps(cache_data, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
            except Exception:
                pass

        structure_score_cache = load_structure_score_cache()

        def build_pitfall_details_df(pitfall_result):
            if not pitfall_result or not pitfall_result.get("details"):
                return None

            normalized_rows = []
            for detail in pitfall_result.get("details", []):
                if not isinstance(detail, dict):
                    continue
                beispiele = detail.get("beispiele")
                if isinstance(beispiele, list):
                    beispiele = ", ".join(str(item) for item in beispiele[:5])
                elif beispiele is None:
                    beispiele = detail.get("name") or detail.get("titel") or ""
                normalized_rows.append({
                    "pitfall": detail.get("pitfall", ""),
                    "titel": detail.get("titel") or detail.get("name") or detail.get("title") or "",
                    "schweregrad": detail.get("schweregrad", ""),
                    "betroffene_elemente": detail.get("betroffene_elemente", 0),
                    "beispiele": beispiele,
                })

            if not normalized_rows:
                return None
            return pd.DataFrame(normalized_rows)

        def run_consistency_check(onto, world=None):
            """Prüft logische Konsistenz mit HermiT.

            Rückgabe: 'Konsistent', 'Inkonsistent' oder 'Unbekannt'.
            'Unbekannt' bedeutet, dass die Prüfung technisch nicht durchgeführt werden
            konnte (z.B. fehlendes Java) - das ist ausdrücklich KEIN negativer Befund.
            """
            try:
                from owlready2 import sync_reasoner
                from owlready2.base import OwlReadyInconsistentOntologyError
            except Exception:
                return 'Unbekannt'

            try:
                with onto:
                    try:
                        if world is not None:
                            sync_reasoner(world, infer_property_values=False, debug=0)
                        else:
                            sync_reasoner(infer_property_values=False, debug=0)
                        return 'Konsistent'
                    except OwlReadyInconsistentOntologyError:
                        return 'Inkonsistent'
                    except Exception:
                        return 'Unbekannt'
            except Exception:
                return 'Unbekannt'

        def get_structure_component_weights(include_consistency, pitfall_available):
            if include_consistency and pitfall_available:
                return [
                    ("schema", 1.0 / 3.0),
                    ("topology", 1.0 / 3.0),
                    ("consistency", 1.0 / 6.0),
                    ("pitfall", 1.0 / 6.0),
                ]
            return [
                ("schema", 0.5),
                ("topology", 0.5),
            ]

        st.markdown("---")
        include_consistency = True

        for ontology_index, owl_file in enumerate(owl_files):
            sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", owl_file.name)
            st.subheader(f"Ontologie: {owl_file.name}")

            owl_bytes = owl_file.getvalue()
            ontology_hash = hashlib.sha256(owl_bytes).hexdigest()
            iof_annotations = (
                st.session_state.get("pitfall_iof_annotationen", False)
                if pitfall_verfahren == "Lokaler Pitfall-Scanner (deterministisch)"
                else False
            )
            pitfall_cache_key = f"{ontology_hash}:{pitfall_verfahren}:{str(iof_annotations).lower()}"
            cache_key = f"{pitfall_cache_key}:v{STRUCTURE_SCORE_CACHE_VERSION}"
            tmp_path = None

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(owl_file.name).suffix.lower()) as tmp:
                    tmp.write(owl_bytes)
                    tmp_path = tmp.name

                world = owllib.World()
                onto, used_import_fallback, import_warn = load_ontology_resilient(
                    tmp_path, owl_file.name, world=world
                )
                if used_import_fallback and import_warn:
                    st.warning(import_warn)

                metrics = get_ontology_metrics(tmp_path)
                consistency = "Unbekannt"
                critical_errors = 0
                major_errors = 0
                minor_errors = 0
                pitfall_score = None
                schema_score = 0.0
                topology_score = 0.0
                consistency_score = 50.0
                struktur_score = 0.0
                pitfall_quelle = pitfall_verfahren
                pitfall_ergebnis = None

                rr_norm = normalize_value(metrics["relationship_richness"], 0.0, 1.0)
                ir_norm = normalize_value(min(metrics["inheritance_richness"], 2.0), 0.0, 2.0)
                ar_norm = normalize_value(min(metrics["attribute_richness"], 5.0), 0.0, 5.0)
                acr_norm = normalize_value(min(metrics["axiom_class_ratio"], 50.0), 0.0, 50.0)
                depth_norm = normalize_value(min(metrics["average_depth"], 6.0), 0.0, 6.0)
                breadth_norm = normalize_value(min(metrics["average_breadth"], 30.0), 0.0, 30.0)
                tangled_norm = normalize_value(metrics["tangledness"], 0.0, 1.0, invert=True)
                annotation_norm = normalize_value(min(metrics["annotation_richness"], 10.0), 0.0, 10.0)

                metrics_table = pd.DataFrame([
                    {"Metrik": "Relationship Richness", "Rohwert": round(metrics["relationship_richness"], 4), "Normiert": round(rr_norm, 2)},
                    {"Metrik": "Inheritance Richness", "Rohwert": round(metrics["inheritance_richness"], 4), "Normiert": round(ir_norm, 2)},
                    {"Metrik": "Attribute Richness", "Rohwert": round(metrics["attribute_richness"], 4), "Normiert": round(ar_norm, 2)},
                    {"Metrik": "Axiom/Class Ratio", "Rohwert": round(metrics["axiom_class_ratio"], 4), "Normiert": round(acr_norm, 2)},
                    {"Metrik": "Average Depth", "Rohwert": round(metrics["average_depth"], 4), "Normiert": round(depth_norm, 2)},
                    {"Metrik": "Average Breadth", "Rohwert": round(metrics["average_breadth"], 4), "Normiert": round(breadth_norm, 2)},
                    {"Metrik": "Tangledness", "Rohwert": round(metrics["tangledness"], 4), "Normiert": round(tangled_norm, 2)},
                    {"Metrik": "Annotation Richness", "Rohwert": round(metrics["annotation_richness"], 4), "Normiert": round(annotation_norm, 2)},
                ])

                st.markdown("**Strukturmetriken**")
                st.dataframe(metrics_table, use_container_width=True)

                basiszahlen_table = pd.DataFrame([
                    {"Kennzahl": "Tripel", "Wert": int(metrics.get("num_triples", 0))},
                    {"Kennzahl": "Klassen (deklariert)", "Wert": int(metrics.get("num_classes", 0))},
                    {"Kennzahl": "Object Properties", "Wert": int(metrics.get("num_object_props", 0))},
                    {"Kennzahl": "Datatype Properties", "Wert": int(metrics.get("num_data_props", 0))},
                    {"Kennzahl": "Annotation Properties", "Wert": int(metrics.get("num_annotation_props", 0))},
                    {"Kennzahl": "SubClassOf zwischen benannten Klassen", "Wert": int(metrics.get("num_subclass_relations", 0))},
                    {"Kennzahl": "owl:Restriction-Vorkommen", "Wert": int(metrics.get("num_restrictions", 0))},
                    {"Kennzahl": "Axiome gesamt", "Wert": int(metrics.get("num_axioms", 0))},
                ])

                with st.expander("Basiszahlen", expanded=False):
                    st.dataframe(basiszahlen_table, use_container_width=True)

                cached_entry = structure_score_cache.get(cache_key)
                if cached_entry is not None:
                    st.info("Verwendet gecachten Strukturscore für dieses Verfahren und diese Option.")
                    consistency = cached_entry.get("konsistenz", consistency)
                    critical_errors = int(cached_entry.get("critical_pitfalls", 0) or 0)
                    major_errors = int(cached_entry.get("major_pitfalls", 0))
                    minor_errors = int(cached_entry.get("minor_pitfalls", 0))
                    pitfall_score = cached_entry.get("pitfall_score")
                    schema_score = float(cached_entry.get("schema_score", 0.0))
                    topology_score = float(cached_entry.get("topologie_score", 0.0))
                    consistency_score = float(cached_entry.get("consistency_score", 50.0))
                    struktur_score = float(cached_entry.get("struktur_score", 0.0))
                    pitfall_quelle = cached_entry.get("pitfall_verfahren", pitfall_verfahren)
                    pitfall_ergebnis = cached_entry.get("pitfall_result") or {
                        "gefunden": 0,
                        "geprueft": 0,
                        "details": [],
                    }
                else:
                    st.markdown("---")
                    with st.spinner("Prüfe automatisch: Logische Konsistenz & Pitfalls..."):
                        consistency = run_consistency_check(onto, world=world)

                        if pitfall_verfahren == "OOPS! (extern)":
                            try:
                                with st.spinner("Frage OOPS! Webservice ab ..."):
                                    pitfall_ergebnis = run_oops_analysis(
                                        owl_text=owl_bytes.decode("utf-8", errors="replace")
                                    )
                            except Exception:
                                pitfall_ergebnis = None
                            if pitfall_ergebnis and isinstance(pitfall_ergebnis, dict):
                                critical_errors = int(pitfall_ergebnis.get("critical", 0) or 0)
                                major_errors = int(pitfall_ergebnis.get("important", 0) or 0)
                                minor_errors = int(pitfall_ergebnis.get("minor", 0) or 0)
                                pitfall_score = compute_oops_pitfall_score(pitfall_ergebnis)
                                pitfall_quelle = "OOPS! (extern)"
                            else:
                                critical_errors = 0
                                major_errors = 0
                                minor_errors = 0
                                pitfall_score = None
                                pitfall_quelle = "OOPS! (extern)"
                        else:
                            extra_definition_properties = IOF_DEFINITION_PROPERTIES if iof_annotations else None
                            pitfall_ergebnis = scan_pitfalls(
                                tmp_path,
                                extra_definition_properties=extra_definition_properties,
                            )
                            critical_errors = int(pitfall_ergebnis.get("critical", 0) or 0)
                            major_errors = int(pitfall_ergebnis.get("important", 0) or 0)
                            minor_errors = int(pitfall_ergebnis.get("minor", 0) or 0)
                            pitfall_score = compute_local_pitfall_score(
                                pitfall_ergebnis["critical"] > 0,
                                pitfall_ergebnis["important"],
                                pitfall_ergebnis["minor"],
                            )
                            pitfall_quelle = "Lokaler Pitfall-Scanner"

                    schema_score = float(np.mean([rr_norm, ir_norm, ar_norm, acr_norm]))
                    topology_score = float(np.mean([depth_norm, breadth_norm, annotation_norm]))

                    if consistency == "Konsistent":
                        consistency_score = 100.0
                    elif consistency == "Inkonsistent":
                        consistency_score = 10.0
                    else:
                        consistency_score = 50.0

                    komponenten = []
                    for component_name, weight in get_structure_component_weights(
                        include_consistency=include_consistency,
                        pitfall_available=pitfall_score is not None,
                    ):
                        if component_name == "schema":
                            komponenten.append((schema_score, weight))
                        elif component_name == "topology":
                            komponenten.append((topology_score, weight))
                        elif component_name == "consistency":
                            komponenten.append((consistency_score, weight))
                        elif component_name == "pitfall" and pitfall_score is not None:
                            komponenten.append((pitfall_score, weight))

                    gewichtssumme = sum(gewicht for _, gewicht in komponenten)
                    struktur_score = round(
                        sum(wert * gewicht for wert, gewicht in komponenten) / gewichtssumme,
                        2,
                    )

                    pitfall_cache_entry = {
                        "cache_version": STRUCTURE_SCORE_CACHE_VERSION,
                        "ontology_name": owl_file.name,
                        "metrics": metrics,
                        "struktur_score": float(struktur_score),
                        "konsistenz": consistency,
                        "critical_pitfalls": int(critical_errors),
                        "major_pitfalls": int(major_errors),
                        "minor_pitfalls": int(minor_errors),
                        "pitfall_score": float(pitfall_score) if pitfall_score is not None else None,
                        "schema_score": float(round(schema_score, 2)),
                        "topologie_score": float(round(topology_score, 2)),
                        "consistency_score": float(consistency_score),
                        "pitfall_verfahren": pitfall_quelle,
                    }
                    if pitfall_ergebnis is not None:
                        pitfall_result_for_cache = {
                            key: value for key, value in pitfall_ergebnis.items() if key != "rohantwort"
                        }
                        pitfall_cache_entry["pitfall_result"] = pitfall_result_for_cache
                    structure_score_cache[cache_key] = pitfall_cache_entry
                    save_structure_score_cache(structure_score_cache)

                if owl_file.name not in st.session_state["scores"]:
                    st.session_state["scores"][owl_file.name] = {}

                st.session_state["scores"][owl_file.name].update({
                    "struktur_score": struktur_score,
                    "konsistenz": consistency,
                    "critical_pitfalls": int(critical_errors),
                    "major_pitfalls": int(major_errors),
                    "minor_pitfalls": int(minor_errors),
                    "pitfall_score": pitfall_score,
                    "schema_score": round(schema_score, 2),
                    "topologie_score": round(topology_score, 2),
                    "manual_override": False,
                    "pitfall_verfahren": pitfall_quelle,
                })

                st.markdown("**Automatisch ermittelte Werte**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Logische Konsistenz:** {consistency}")
                    st.write(f"**Pitfalls:** kritisch={int(critical_errors)}, important={major_errors}, minor={minor_errors}")
                with col2:
                    if pitfall_score is None:
                        st.warning("nicht geprüft (OOPS! nicht erreichbar)")
                    else:
                        st.metric("Pitfall Score", f"{pitfall_score} / 100")
                    st.caption(f"Verfahren: {pitfall_quelle}")
                    if pitfall_quelle == "Lokaler Pitfall-Scanner" and pitfall_ergebnis:
                        st.caption(f"{pitfall_ergebnis.get('gefunden', 0)} von {pitfall_ergebnis.get('geprueft', 0)} geprüften Pitfalls gefunden")

                st.markdown("**Berechnete Strukturscores**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Schema Score", f"{round(schema_score, 2)} / 100")
                with col2:
                    st.metric("Topologie Score", f"{round(topology_score, 2)} / 100")
                with col3:
                    if pitfall_score is None:
                        st.warning("nicht geprüft")
                    else:
                        st.metric("Pitfall Score", f"{pitfall_score} / 100")
                    st.caption(f"Verfahren: {pitfall_quelle}")
                with col4:
                    st.metric("Strukturscore", f"{struktur_score} / 100")

                st.markdown("**Komponenten des Strukturscores**")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Schema Score", f"{round(schema_score, 2)} / 100")
                with col2:
                    st.metric("Topologie Score", f"{round(topology_score, 2)} / 100")
                with col3:
                    if pitfall_score is None:
                        st.warning("nicht geprüft")
                    else:
                        st.metric("Pitfall Score", f"{pitfall_score} / 100")
                with col4:
                    st.metric("Konsistenz Score", f"{round(consistency_score, 2)} / 100")

                st.markdown("---")
                st.markdown("**Gesamtscore**")
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.metric("Strukturscore", f"{struktur_score} / 100")
                with col2:
                    if struktur_score >= 80:
                        st.success("● ● ● ● ●")
                    elif struktur_score >= 60:
                        st.success("● ● ● ● ○")
                    elif struktur_score >= 40:
                        st.warning("● ● ● ○ ○")
                    elif struktur_score >= 20:
                        st.warning("● ● ○ ○ ○")
                    else:
                        st.error("● ○ ○ ○ ○")

                if pitfall_score is None:
                    st.info("Strukturscore ohne Pitfall-Komponente berechnet, weil OOPS! nicht erreichbar war.")

                if pitfall_ergebnis and pitfall_ergebnis.get("details"):
                    details_toggle_key = f"show_pitfall_details_{ontology_index}_{ontology_hash[:12]}"
                    if details_toggle_key not in st.session_state:
                        st.session_state[details_toggle_key] = False

                    if st.button(
                        "Pitfall-Details anzeigen/verbergen",
                        key=f"pitfall_details_btn_{ontology_index}_{ontology_hash[:12]}",
                    ):
                        st.session_state[details_toggle_key] = not st.session_state[details_toggle_key]

                    if st.session_state[details_toggle_key]:
                        detail_df = build_pitfall_details_df(pitfall_ergebnis)
                        if detail_df is not None and not detail_df.empty:
                            detail_df = detail_df.rename(columns={
                                "pitfall": "Pitfall",
                                "titel": "Titel",
                                "schweregrad": "Schweregrad",
                                "betroffene_elemente": "betroffene Elemente",
                                "beispiele": "Beispiele",
                            })
                            st.subheader("Pitfall-Details")
                            st.dataframe(detail_df, use_container_width=True)

                            if pitfall_quelle == "OOPS! (extern)" and pitfall_ergebnis:
                                with st.expander("Rohantwort des OOPS!-Webservice", expanded=False):
                                    st.code(pitfall_ergebnis.get("rohantwort", "")[:20000], language="xml")

                if struktur_score >= 80:
                    st.success("Sehr gut — Starke Ontologie-Struktur")
                elif struktur_score >= 60:
                    st.success("Gut — Solide Struktur")
                elif struktur_score >= 40:
                    st.warning("Mittel — Akzeptable Struktur")
                elif struktur_score >= 20:
                    st.warning("Schwach — Strukturverbesserung empfohlen")
                else:
                    st.error("Nicht erfüllt — Schwache Struktur")

                st.markdown("---")
                manual_key_suffix = f"{ontology_index}_{sanitized}_{hashlib.sha256(owl_bytes).hexdigest()[:12]}"
                if st.button("Manuelle Eingabe", key=f"manual_input_btn_{manual_key_suffix}"):
                    st.session_state[f"show_manual_input_{manual_key_suffix}"] = not st.session_state.get(f"show_manual_input_{manual_key_suffix}", False)

                if st.session_state.get(f"show_manual_input_{manual_key_suffix}", False):
                    st.markdown("### Manuelle Eingabe: Logische Konsistenz & Pitfalls überschreiben")

                    with st.form(key=f"manual_form_{manual_key_suffix}"):
                        consistency_manual = st.selectbox(
                            "Logische Konsistenz",
                            options=["Konsistent", "Inkonsistent", "Unbekannt"],
                            index=["Konsistent", "Inkonsistent", "Unbekannt"].index(consistency) if consistency in ["Konsistent", "Inkonsistent", "Unbekannt"] else 2,
                            key=f"manual_consistency_{manual_key_suffix}",
                        )

                        critical_manual = st.number_input(
                            "Anzahl Critical Pitfalls",
                            min_value=0,
                            step=1,
                            value=int(critical_errors),
                            key=f"manual_critical_{manual_key_suffix}",
                        )

                        major_manual = st.number_input(
                            "Anzahl Major Pitfalls",
                            min_value=0,
                            step=1,
                            value=int(major_errors),
                            key=f"manual_major_{manual_key_suffix}",
                        )

                        minor_manual = st.number_input(
                            "Anzahl Minor Pitfalls",
                            min_value=0,
                            step=1,
                            value=int(minor_errors),
                            key=f"manual_minor_{manual_key_suffix}",
                        )

                        submitted = st.form_submit_button("Speichern", use_container_width=True)

                        if submitted:
                            pitfall_score_manual = None
                            if int(critical_manual) > 0:
                                pitfall_score_manual = 10.0
                            elif major_manual or minor_manual:
                                important_score = max(0.0, 100.0 - float(major_manual) * 10.0)
                                minor_score = max(0.0, 100.0 - float(minor_manual) * 2.0)
                                pitfall_score_manual = float(min(important_score, minor_score))
                            else:
                                pitfall_score_manual = 100.0

                            if consistency_manual == "Konsistent":
                                consistency_score_manual = 100.0
                            elif consistency_manual == "Inkonsistent":
                                consistency_score_manual = 10.0
                            else:
                                consistency_score_manual = 50.0

                            manual_components = []
                            for component_name, weight in get_structure_component_weights(
                                include_consistency=include_consistency,
                                pitfall_available=pitfall_score_manual is not None,
                            ):
                                if component_name == "schema":
                                    manual_components.append((schema_score, weight))
                                elif component_name == "topology":
                                    manual_components.append((topology_score, weight))
                                elif component_name == "consistency":
                                    manual_components.append((consistency_score_manual, weight))
                                elif component_name == "pitfall" and pitfall_score_manual is not None:
                                    manual_components.append((pitfall_score_manual, weight))
                            manual_weight_sum = sum(weight for _, weight in manual_components)
                            struktur_score_manual = round(
                                sum(value * weight for value, weight in manual_components) / manual_weight_sum,
                                2,
                            )

                            st.session_state["scores"][owl_file.name].update({
                                "konsistenz": consistency_manual,
                                "critical_pitfalls": int(critical_manual),
                                "major_pitfalls": int(major_manual),
                                "minor_pitfalls": int(minor_manual),
                                "pitfall_score": pitfall_score_manual,
                                "struktur_score": struktur_score_manual,
                                "manual_override": True,
                            })

                            st.success("Manuelle Eingaben gespeichert und Strukturscore neu berechnet!")
                            st.session_state[f"show_manual_input_{manual_key_suffix}"] = False
                            st.rerun()
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)


# ============================================================
elif phase == "Phase 4: Competency Questions":
# ============================================================
    st.header("Phase 4: Competency Questions")

    if "cq_modus" not in st.session_state:
        st.session_state["cq_modus"] = "Semantisch + Lexikalisch"
    if "cq_schwelle" not in st.session_state:
        st.session_state["cq_schwelle"] = 0.45
    if "cq_top_k" not in st.session_state:
        st.session_state["cq_top_k"] = 15

    with st.expander("Erweiterte Einstellungen", expanded=False):
        cq_modus = st.radio(
            "Zuordnungsverfahren",
            options=["Semantisch", "Lexikalisch", "Semantisch + Lexikalisch"],
            index=["Semantisch", "Lexikalisch", "Semantisch + Lexikalisch"].index(st.session_state["cq_modus"]),
            horizontal=True,
            key="cq_modus",
            help=(
                "Semantisch nutzt das mehrsprachige Sentence-Transformer-Modell und "
                "findet auch englischsprachige Entitäten zu deutschen Fragen. "
                "Lexikalisch sucht nach wörtlichen Übereinstimmungen. Die kombinierte "
                "Variante vereint beide Ergebnismengen."
            ),
        )

        spalte_a, spalte_b = st.columns(2)
        with spalte_a:
            cq_schwelle = st.slider(
                "Ähnlichkeitsschwelle",
                min_value=0.20,
                max_value=0.80,
                value=st.session_state["cq_schwelle"],
                step=0.05,
                key="cq_schwelle",
                help="Ab welcher Kosinus-Ähnlichkeit eine Entität als Treffer zählt.",
            )
        with spalte_b:
            cq_top_k = st.slider(
                "Maximale Treffer pro Frage",
                min_value=5,
                max_value=50,
                value=st.session_state["cq_top_k"],
                step=5,
                key="cq_top_k",
            )

    if "cq_liste" not in st.session_state or not st.session_state.get("cq_liste"):
        st.warning("Keine Competency Questions gefunden. Bitte zuerst in Phase 1 CQs eingeben.")
    elif "owl_files" not in st.session_state or not st.session_state.get("owl_files"):
        st.warning("Keine Ontologien gefunden. Bitte zuerst in Phase 1 Ontologien hochladen.")
    else:
        cq_liste = st.session_state["cq_liste"]
        owl_files = st.session_state["owl_files"]

        if "cq_results" not in st.session_state:
            st.session_state["cq_results"] = {}
        if "scores" not in st.session_state:
            st.session_state["scores"] = {}

        def _execute_for_ontology(owl_file, modus, schwelle, top_k):
            suffix = Path(owl_file.name).suffix or ".owl"
            tmp_path = None
            results = {}
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(owl_file.getvalue())
                    tmp_path = tmp_file.name

                graph = Graph()
                parse_ok = False
                parse_errors = []
                for fmt in [None, "xml", "turtle", "n3", "nt", "trig", "json-ld"]:
                    try:
                        if fmt is None:
                            graph.parse(Path(tmp_path))
                        else:
                            graph.parse(Path(tmp_path), format=fmt)
                        parse_ok = True
                        break
                    except Exception as parse_attempt_error:
                        parse_errors.append(f"{fmt or 'auto'}: {parse_attempt_error}")

                if not parse_ok:
                    return {}, " | ".join(parse_errors)

                entity_index = _build_entity_index(graph)
                ontology_hash = hashlib.sha256(owl_file.getvalue()).hexdigest()
                embedding_cache = st.session_state.setdefault("cq_entity_embeddings", {})
                model_loader = _load_bert_model()
                use_semantic = model_loader.get("backend") == "bert" and not model_loader.get("error")

                if ontology_hash not in embedding_cache:
                    if use_semantic:
                        embeddings = _encode_texts(
                            [entry["text"] for entry in entity_index],
                            model_loader,
                            batch_size=32,
                        )
                    else:
                        embeddings = None
                    embedding_cache[ontology_hash] = {
                        "index": entity_index,
                        "embeddings": embeddings,
                    }
                else:
                    entity_index = embedding_cache[ontology_hash].get("index", entity_index)
                    embeddings = embedding_cache[ontology_hash].get("embeddings")

                if not use_semantic and modus in {"Semantisch", "Semantisch + Lexikalisch"}:
                    st.warning("Sentence-Transformer-Modell nicht verfügbar. Phase 4 fällt auf lexikalisch zurück.")

                for idx, cq_text in enumerate(cq_liste):
                    if modus == "Lexikalisch":
                        hits = _lexikalische_treffer(cq_text, entity_index, top_k=top_k)
                        applied_mode = "Lexikalisch"
                    elif modus == "Semantisch":
                        if use_semantic:
                            hits = _semantische_treffer(
                                cq_text,
                                entity_index,
                                embeddings,
                                model_loader,
                                schwelle=schwelle,
                                top_k=top_k,
                            )
                            applied_mode = "Semantisch"
                        else:
                            hits = _lexikalische_treffer(cq_text, entity_index, top_k=top_k)
                            applied_mode = "Lexikalisch (Fallback)"
                    else:
                        semantic_hits = []
                        lexical_hits = []
                        if use_semantic:
                            semantic_hits = _semantische_treffer(
                                cq_text,
                                entity_index,
                                embeddings,
                                model_loader,
                                schwelle=schwelle,
                                top_k=top_k,
                            )
                        lexical_hits = _lexikalische_treffer(cq_text, entity_index, top_k=top_k)
                        hits = _combine_treffer(semantic_hits, lexical_hits)
                        applied_mode = "Semantisch + Lexikalisch"

                    rows = _rows_from_hits(hits, applied_mode)
                    best_score = rows[0]["Ähnlichkeit/Score"] if rows else 0.0

                    klassen_kandidaten, _ = _kandidaten_aus_treffern(hits)
                    verknuepfung = {"gefunden": False, "relationen": [], "abfrage": ""}
                    geprueftes_paar = None

                    if len(klassen_kandidaten) >= 2:
                        for i in range(len(klassen_kandidaten)):
                            for j in range(i + 1, len(klassen_kandidaten)):
                                ergebnis = _pruefe_verknuepfung(
                                    graph,
                                    klassen_kandidaten[i]["iri"],
                                    klassen_kandidaten[j]["iri"],
                                )
                                if ergebnis["gefunden"]:
                                    verknuepfung = ergebnis
                                    geprueftes_paar = (
                                        klassen_kandidaten[i]["name"],
                                        klassen_kandidaten[j]["name"],
                                    )
                                    break
                            if verknuepfung["gefunden"]:
                                break
                        if geprueftes_paar is None:
                            geprueftes_paar = (
                                klassen_kandidaten[0]["name"],
                                klassen_kandidaten[1]["name"],
                            )
                            verknuepfung["abfrage"] = VERKNUEPFUNGS_QUERY % {
                                "a": klassen_kandidaten[0]["iri"],
                                "b": klassen_kandidaten[1]["iri"],
                            }

                    if not hits:
                        status = "nicht beantwortbar"
                    elif verknuepfung["gefunden"]:
                        status = "vollständig beantwortbar"
                    else:
                        status = "teilweise beantwortbar"

                    results[idx] = {
                        "rows": rows,
                        "hit_count": len(rows),
                        "status": status,
                        "verknuepfung_gefunden": verknuepfung["gefunden"],
                        "relationen": verknuepfung["relationen"],
                        "sparql_abfrage": verknuepfung["abfrage"],
                        "geprueftes_paar": geprueftes_paar,
                        "best_score": best_score,
                        "modus": applied_mode,
                    }

                return results, None
            except Exception as parse_error:
                return {}, str(parse_error)
            finally:
                if tmp_path and Path(tmp_path).exists():
                    Path(tmp_path).unlink(missing_ok=True)

        onto_names = [f.name for f in owl_files]
        selected_name = st.selectbox("Ontologie auswählen", onto_names, key="cq_selected_ontology")
        selected_file = next((f for f in owl_files if f.name == selected_name), None)
        sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", selected_name)

        if st.button("Automatisch Antworten laden", key=f"cq_run_{sanitized}") and selected_file is not None:
            run_results, parse_error = _execute_for_ontology(selected_file, cq_modus, cq_schwelle, cq_top_k)
            if parse_error:
                st.error(f"Fehler beim Laden/Ausführen von {selected_name}: {parse_error}")
            else:
                st.session_state["cq_results"][selected_name] = run_results
                st.success(f"Antworten für {selected_name} aktualisiert.")

        exec_results = st.session_state["cq_results"].get(selected_name, {})
        if not exec_results:
            st.info("Noch keine automatische Ausführung für diese Ontologie. Klicke auf 'Automatisch Antworten laden'.")

        GEWICHTE = {
            "vollständig beantwortbar": 1.0,
            "teilweise beantwortbar": 0.5,
            "nicht beantwortbar": 0.0,
        }

        details = []
        best_scores = []
        status_counts = {status: 0 for status in GEWICHTE}
        status_best_similarity = {status: [] for status in GEWICHTE}

        with st.expander("Status-Bedeutung", expanded=False):
            st.markdown(
                "**vollständig beantwortbar**\n"
                "Passende Konzepte vorhanden **und** im Graphen verknüpft\n\n"
                "**teilweise beantwortbar**\n"
                "Passende Konzepte vorhanden, aber keine Relation zwischen ihnen\n\n"
                "**nicht beantwortbar**\n"
                "Keine passenden Konzepte oberhalb der Ähnlichkeitsschwelle"
            )

        for idx, cq in enumerate(cq_liste):
            cq_result = exec_results.get(idx, {})
            hit_count = int(cq_result.get("hit_count", 0))
            result_rows = cq_result.get("rows", [])
            auto_status = cq_result.get("status", "nicht beantwortbar")
            best_score = float(cq_result.get("best_score", 0.0) or 0.0)
            modus_used = cq_result.get("modus", cq_modus)
            verknuepfung_gefunden = bool(cq_result.get("verknuepfung_gefunden", False))
            relationen = cq_result.get("relationen", [])
            sparql_abfrage = cq_result.get("sparql_abfrage", "")
            geprueftes_paar = cq_result.get("geprueftes_paar", None)

            best_scores.append(best_score)

            manual_status_key = f"cq_status_{sanitized}_{idx}"
            manual_reason_key = f"cq_reason_{sanitized}_{idx}"
            manual_status = st.session_state.get(manual_status_key, "automatisch übernehmen")
            options = ["automatisch übernehmen", "vollständig beantwortbar", "teilweise beantwortbar", "nicht beantwortbar"]
            status_index = options.index(manual_status) if manual_status in options else 0
            selected_manual_status = st.selectbox(
                "Bewertung",
                options=options,
                index=status_index,
                key=manual_status_key,
            )
            effective_status = selected_manual_status if selected_manual_status != "automatisch übernehmen" else auto_status
            manual_override = selected_manual_status != "automatisch übernehmen"

            if manual_override and selected_manual_status != auto_status:
                manual_reason = st.text_area(
                    "Begründung",
                    value=st.session_state.get(manual_reason_key, ""),
                    key=manual_reason_key,
                    height=100,
                )
            else:
                manual_reason = st.session_state.get(manual_reason_key, "")
                if selected_manual_status == "automatisch übernehmen":
                    st.session_state.pop(manual_reason_key, None)

            if manual_override:
                st.session_state[manual_reason_key] = manual_reason

            exec_results[idx] = {
                **cq_result,
                "status": effective_status,
                "verknuepfung_gefunden": verknuepfung_gefunden,
                "relationen": relationen,
                "sparql_abfrage": sparql_abfrage,
                "geprueftes_paar": geprueftes_paar,
                "best_score": best_score,
                "modus": modus_used,
                "manuell_korrigiert": manual_override,
                "manuelle_bewertung": None if not manual_override else selected_manual_status,
                "begruendung": manual_reason if manual_override else "",
            }
            st.session_state["cq_results"][selected_name] = exec_results

            status_counts[effective_status] += 1
            status_best_similarity[effective_status].append(best_score)

            with st.expander(f"CQ {idx + 1}: {cq}", expanded=(idx == 0)):
                if effective_status == "vollständig beantwortbar":
                    st.success(f"Status: {effective_status}")
                elif effective_status == "teilweise beantwortbar":
                    st.warning(f"Status: {effective_status}")
                else:
                    st.error(f"Status: {effective_status}")

                if manual_override:
                    st.caption("Manuell korrigiert")
                    if manual_reason:
                        st.write(f"**Begründung:** {manual_reason}")

                st.write(f"**Treffer:** {hit_count}")
                st.write(f"**Verfahren:** {modus_used}")
                if result_rows:
                    st.dataframe(
                        pd.DataFrame(result_rows),
                        use_container_width=True,
                    )
                else:
                    st.warning("Keine Treffer gefunden.")

                if effective_status == "vollständig beantwortbar" and verknuepfung_gefunden and relationen:
                    erste_relation = relationen[0]
                    property_name = _local_name(erste_relation.get("property", ""))
                    st.write(
                        f"Verknüpfung: {geprueftes_paar[0] if geprueftes_paar else '?'} "
                        f"--[{property_name}]--> {geprueftes_paar[1] if geprueftes_paar else '?'} "
                        f"(über {erste_relation.get('art', '')})"
                    )
                elif geprueftes_paar:
                    st.info(
                        f"Kein Pfad zwischen {geprueftes_paar[0]} und {geprueftes_paar[1]} gefunden."
                    )

                if sparql_abfrage:
                    with st.expander("SPARQL-Abfrage", expanded=False):
                        st.code(sparql_abfrage, language="sparql")

            details.append({
                "CQ Nummer": idx + 1,
                "Competency Question": cq,
                "Status": effective_status,
                "Treffer": hit_count,
                "Beste Ähnlichkeit/Score": round(best_score, 4),
                "Punkte": GEWICHTE[effective_status],
                "Manuell korrigiert": manual_override,
                "Begründung": manual_reason,
            })

        cq_score = round((sum(d["Punkte"] for d in details) / len(cq_liste)) * 100.0, 2) if cq_liste else 0.0
        mean_best_similarity = round(float(np.mean(best_scores)) if best_scores else 0.0, 4)
        status_summary = []
        for status_name in ["vollständig beantwortbar", "teilweise beantwortbar", "nicht beantwortbar"]:
            status_summary.append({
                "Status": status_name,
                "Anzahl": status_counts.get(status_name, 0),
                "Mittlere beste Ähnlichkeit": round(
                    float(np.mean(status_best_similarity.get(status_name, []))) if status_best_similarity.get(status_name) else 0.0,
                    4,
                ),
            })
        st.session_state["scores"].setdefault(selected_name, {})
        st.session_state["scores"][selected_name].update({
            "cq_score": cq_score,
            "cq_details": details,
            "cq_mode": cq_modus,
            "cq_schwelle": cq_schwelle,
            "cq_top_k": cq_top_k,
            "cq_mean_best_similarity": mean_best_similarity,
            "cq_status_counts": status_counts,
            "cq_status_summary": status_summary,
        })

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("CQ-Score", f"{cq_score} / 100")
        with c2:
            st.metric("Vollständig beantwortbar", f"{status_counts['vollständig beantwortbar']} / {len(cq_liste)}")
        with c3:
            st.metric("Mittlere beste Ähnlichkeit", f"{mean_best_similarity:.4f}")

        if details:
            st.subheader("Zusammenfassung")
            st.dataframe(pd.DataFrame(details), use_container_width=True)

            st.subheader("Statusübersicht")
            st.dataframe(pd.DataFrame(status_summary), use_container_width=True)

# ============================================================
elif phase == "Phase 5: Gesamtscore":
# ============================================================
    st.header("Phase 5: Gesamtscore & Ranking")
    st.markdown("Gewichtete Gesamtevaluation aller Ontologien basierend auf Phase 2, 3 und 4.")

    # Gewichtungen laden mit Defaults
    w1 = st.session_state.get("w1", 0.35)
    w2 = st.session_state.get("w2", 0.35)
    w3 = st.session_state.get("w3", 0.30)

    # Gewichtung anzeigen
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**w1** (Phase 2)\n{w1 * 100:.0f}%")
    with col2:
        st.info(f"**w2** (Phase 3)\n{w2 * 100:.0f}%")
    with col3:
        st.info(f"**w3** (Phase 4)\n{w3 * 100:.0f}%")

    st.markdown("---")

    # Scores aus session_state laden
    if "owl_files" not in st.session_state or not st.session_state.get("owl_files"):
        st.warning("Keine Ontologien geladen. Bitte zuerst in Phase 1 Ontologien hochladen.")
    else:
        owl_files = st.session_state["owl_files"]
        scores_dict = st.session_state.get("scores", {})

        if not scores_dict:
            st.warning("Keine Evaluierungsergebnisse vorhanden. Bitte Phase 2–4 durchlaufen.")
        else:
            # Harvey-Ball Funktion
            def get_harvey_ball(score):
                if score >= 80:
                    return "● ● ● ● ● Sehr gut"
                elif score >= 60:
                    return "● ● ● ● ○ Gut"
                elif score >= 40:
                    return "● ● ● ○ ○ Mittel"
                elif score >= 20:
                    return "● ● ○ ○ ○ Schwach"
                else:
                    return "● ○ ○ ○ ○ Nicht erfüllt"

            # Ranking berechnen
            ranking_data = []
            for owl_file in owl_files:
                onto_name = owl_file.name
                onto_scores = scores_dict.get(onto_name, {})

                coverage = onto_scores.get("coverage_score", 0)
                struktur = onto_scores.get("struktur_score", 0)
                cq = onto_scores.get("cq_score", 0)

                gesamtscore = round(
                    w1 * coverage + w2 * struktur + w3 * cq,
                    2
                )

                ranking_data.append({
                    "Ontologie": onto_name,
                    "Coverage Score": coverage,
                    "Strukturscore": struktur,
                    "CQ-Score": cq,
                    "Gesamtscore": gesamtscore
                })

            # Sortieren: Gesamtscore desc, dann Coverage Score desc (Tiebreaker)
            ranking_df = pd.DataFrame(ranking_data).sort_values(
                by=["Gesamtscore", "Coverage Score"],
                ascending=False
            ).reset_index(drop=True)

            # Rang hinzufügen
            ranking_df.insert(0, "Rang", range(1, len(ranking_df) + 1))

            # Harvey-Ball Bewertung hinzufügen
            ranking_df["Bewertung"] = ranking_df["Gesamtscore"].apply(get_harvey_ball)

            # Rangliste anzeigen
            st.subheader("Rangliste aller Ontologien")

            # Styling: Rang 1 farblich hervorheben
            def highlight_rank1(row):
                if row["Rang"] == 1:
                    return ["background-color: lightgreen"] * len(row)
                return [""] * len(row)

            display_df = ranking_df[[
                "Rang", "Ontologie", "Coverage Score", "Strukturscore",
                "CQ-Score", "Gesamtscore", "Bewertung"
            ]]

            st.dataframe(
                display_df.style.apply(highlight_rank1, axis=1),
                use_container_width=True
            )

            st.markdown("---")

            # Beste Ontologie und Empfehlung
            best_onto = ranking_df.iloc[0]
            best_name = best_onto["Ontologie"]
            best_score = best_onto["Gesamtscore"]
            best_coverage = best_onto["Coverage Score"]
            best_struktur = best_onto["Strukturscore"]
            best_cq = best_onto["CQ-Score"]

            st.subheader(f"Beste Ontologie: {best_name}")
            st.metric("Gesamtscore", f"{best_score} / 100")

            # Kombinationslogik für Empfehlung
            def get_recommendation(coverage, struktur, cq):
                if coverage >= 60 and struktur >= 60 and cq >= 60:
                    return ("Sehr geeignet", "Alle Kriterien erfüllt — Top-Kandidat")
                elif coverage >= 60 and struktur >= 60 and cq < 60:
                    return (
                        "Bedingt geeignet",
                        "Hohe Domänenabdeckung und Struktur, aber CQ-Abdeckung schwach"
                    )
                elif cq >= 60 and (coverage < 60 or struktur < 60):
                    return (
                        "Situativ geeignet",
                        "Gute CQ-Abdeckung, aber andere Kriterien schwächer"
                    )
                elif best_score >= 40:
                    return (
                        "Bedingt geeignet",
                        "Gesamtscore im mittleren Bereich — weitere Evaluierung empfohlen"
                    )
                else:
                    return (
                        "Nicht geeignet",
                        "Scores zu niedrig — erweitern oder alternative Ontologie prüfen"
                    )

            recommendation, reasoning = get_recommendation(best_coverage, best_struktur, best_cq)
            st.success(f"**{recommendation}**\n\n{reasoning}")

            st.markdown("---")

            # Detailansicht
            st.subheader("Detailansicht einzelner Ontologien")

            # Selectbox mit Rangfolge
            ontology_options = [
                f"{int(row['Rang'])}. {row['Ontologie']}"
                for _, row in ranking_df.iterrows()
            ]

            selected_display = st.selectbox(
                "Wähle eine Ontologie",
                ontology_options
            )

            # Extrahiere den Namen ohne Rangfolge
            selected_onto = selected_display.split(". ", 1)[1] if ". " in selected_display else selected_display

            if selected_onto:
                selected_scores = scores_dict.get(selected_onto, {})

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Coverage Score",
                        f"{selected_scores.get('coverage_score', 0)} / 100"
                    )
                    if selected_scores.get("coverage_score", 0):
                        st.caption(get_harvey_ball(selected_scores.get("coverage_score", 0)))

                with col2:
                    st.metric(
                        "Strukturscore",
                        f"{selected_scores.get('struktur_score', 0)} / 100"
                    )
                    if selected_scores.get("struktur_score", 0):
                        st.caption(get_harvey_ball(selected_scores.get("struktur_score", 0)))

                    # Expander für Details
                    with st.expander("Details anzeigen"):
                        st.write("**Schema Score:** " + str(selected_scores.get('schema_score', 0)) + " / 100")
                        st.write("**Topologie Score:** " + str(selected_scores.get('topologie_score', 0)) + " / 100")
                        st.write("**Pitfall Score:** " + str(selected_scores.get('pitfall_score', 0)) + " / 100")
                        st.write("**Logische Konsistenz:** " + str(selected_scores.get("konsistenz", "Unbekannt")))

                with col3:
                    st.metric(
                        "CQ-Score",
                        f"{selected_scores.get('cq_score', 0)} / 100"
                    )
                    if selected_scores.get("cq_score", 0):
                        st.caption(get_harvey_ball(selected_scores.get("cq_score", 0)))

                st.markdown("---")

                # Gesamtscore für diese Ontologie
                gesamtscore_detail = round(
                    w1 * selected_scores.get("coverage_score", 0) +
                    w2 * selected_scores.get("struktur_score", 0) +
                    w3 * selected_scores.get("cq_score", 0),
                    2
                )

                st.subheader(f"Gesamtscore: {gesamtscore_detail} / 100")
                st.caption(get_harvey_ball(gesamtscore_detail))
 