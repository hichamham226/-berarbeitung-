"""
Lokaler, deterministischer Pitfall-Scanner nach dem OOPS!-Katalog.

Prueft 21 der 41 Pitfalls aus dem OOPS!-Katalog (Poveda-Villalon et al. 2014),
und zwar genau die, die sich rein strukturell aus der Datei entscheiden lassen.
Schweregrade (Critical / Important / Minor) sind aus dem offiziellen Katalog
uebernommen: https://oops.linkeddata.es/catalogue.jsp

NICHT geprueft werden Pitfalls, die semantisches Urteil oder Web-Dereferenzierung
erfordern (P01, P02, P05, P07, P09, P14-P18, P23, P27-P31, P37, P40) sowie P12,
P20 und P21b. Fuer diese bleibt der Aufruf des echten OOPS!-Dienstes noetig.

Zaehlweise: gezaehlt wird die Anzahl der ERKANNTEN PITFALL-ARTEN je Schweregrad
(nicht die Anzahl betroffener Elemente) - konsistent zur Auswertung im
Ontologie-Evaluator.
"""

import os
import re
from collections import defaultdict
from pathlib import Path

from rdflib import Graph, RDF, RDFS, OWL, URIRef, BNode, Literal, Namespace

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
DC = Namespace("http://purl.org/dc/elements/1.1/")
DCTERMS = Namespace("http://purl.org/dc/terms/")
CC = Namespace("http://creativecommons.org/ns#")
SCHEMA = Namespace("http://schema.org/")
IOF_AV = Namespace("https://spec.industrialontologies.org/ontology/annotation/")

# Optional an scan_pitfalls() uebergebbar, damit P08 die IOF-eigenen
# Dokumentationsannotationen als Definition anerkennt.
IOF_DEFINITION_PROPERTIES = {IOF_AV.naturalLanguageDefinition,
                             IOF_AV.explanatoryNote}

LABEL_PROPERTIES = {RDFS.label, SKOS.prefLabel, SKOS.altLabel}
DEFINITION_PROPERTIES = {RDFS.comment, DC.description, DCTERMS.description,
                         SKOS.definition, DCTERMS.abstract}
LICENSE_PROPERTIES = {DCTERMS.license, DC.rights, DCTERMS.rights,
                      CC.license, SCHEMA.license}

SEVERITY = {
    "P03": "critical", "P06": "critical", "P19": "critical", "P39": "critical",
    "P10": "important", "P11": "important", "P24": "important",
    "P25": "important", "P26": "important", "P34": "important",
    "P35": "important", "P38": "important", "P41": "important",
    "P04": "minor", "P08": "minor", "P13": "minor", "P21": "minor",
    "P22": "minor", "P32": "minor", "P33": "minor", "P36": "minor",
}

TITLES = {
    "P03": 'Relationship "is" statt rdfs:subClassOf / rdf:type / owl:sameAs',
    "P04": "Unverbundene Ontologie-Elemente",
    "P06": "Zyklus in der Klassenhierarchie",
    "P08": "Fehlende Annotationen",
    "P10": "Fehlende Disjunktheit",
    "P11": "Fehlende Domain oder Range bei Properties",
    "P13": "Inverse Relationen nicht explizit deklariert",
    "P19": "Mehrfache Domains oder Ranges",
    "P21": "Sammelklasse (miscellaneous class)",
    "P22": "Uneinheitliche Namenskonventionen",
    "P24": "Rekursive Definitionen",
    "P25": "Relation invers zu sich selbst",
    "P26": "Inverse fuer eine symmetrische Relation definiert",
    "P32": "Mehrere Klassen mit identischem Label",
    "P33": "Property-Chain mit nur einer Property",
    "P34": "Untypisierte Klasse",
    "P35": "Untypisierte Property",
    "P36": "Dateiendung in der URI",
    "P38": "Keine owl:Ontology-Deklaration",
    "P39": "Mehrdeutiger Namespace",
    "P41": "Keine Lizenz deklariert",
}


def _local_name(uri):
    text = str(uri)
    for sep in ("#", "/"):
        if sep in text:
            text = text.rsplit(sep, 1)[-1]
    return text


def _naming_style(name):
    if "_" in name:
        return "snake_case"
    if "-" in name:
        return "kebab-case"
    if re.search(r"[a-z][A-Z]", name):
        return "camelCase"
    return "other"


def _load_graph(source):
    if isinstance(source, Graph):
        return source

    if hasattr(source, "read"):
        try:
            graph = Graph()
            graph.parse(source, format="xml")
            return graph
        except Exception:
            return Graph()

    if isinstance(source, (str, os.PathLike)):
        text = os.fspath(source)
        if text.startswith(("http://", "https://", "ftp://", "file://")):
            graph = Graph()
            graph.parse(text, format="xml")
            return graph

        path = Path(text)
        if path.exists():
            with path.open("rb") as handle:
                graph = Graph()
                graph.parse(handle, format="xml")
                return graph

        if "<" in text and ">" in text:
            graph = Graph()
            graph.parse(data=text, format="xml")
            return graph

    if isinstance(source, (str, os.PathLike)):
        text = os.fspath(source)
        if text.startswith(("http://", "https://", "ftp://", "file://")):
            graph = Graph()
            graph.parse(text, format="xml")
            return graph
        if os.path.exists(text):
            with open(text, "rb") as handle:
                graph = Graph()
                graph.parse(handle, format="xml")
                return graph
        if "<" in text and ">" in text:
            graph = Graph()
            graph.parse(data=text, format="xml")
            return graph

    try:
        graph = Graph()
        graph.parse(source, format="xml")
        return graph
    except Exception:
        return Graph()


def scan_pitfalls(source, ontology_namespace=None, extra_definition_properties=None):
    graph = _load_graph(source)

    # P08 kann optional zusaetzliche Definitions-Annotationen anerkennen
    # (z.B. iof-av:naturalLanguageDefinition bei IOF-Ontologien).
    definition_properties = set(DEFINITION_PROPERTIES)
    if extra_definition_properties:
        definition_properties |= set(extra_definition_properties)

    findings = {}   # pitfall-id -> Liste betroffener Elemente

    def record(pid, elements):
        elements = list(elements)
        if elements:
            findings[pid] = elements

    # ------------------------------------------------------------------
    # Entitaeten
    # ------------------------------------------------------------------
    classes = {s for s in graph.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}
    obj_props = {s for s in graph.subjects(RDF.type, OWL.ObjectProperty) if isinstance(s, URIRef)}
    data_props = {s for s in graph.subjects(RDF.type, OWL.DatatypeProperty) if isinstance(s, URIRef)}
    ann_props = {s for s in graph.subjects(RDF.type, OWL.AnnotationProperty) if isinstance(s, URIRef)}
    rdf_props = {s for s in graph.subjects(RDF.type, RDF.Property) if isinstance(s, URIRef)}
    ontologies = set(graph.subjects(RDF.type, OWL.Ontology))

    declared_properties = obj_props | data_props | ann_props | rdf_props
    own_entities = classes | obj_props | data_props

    # Eigener Namespace: haeufigster Praefix der deklarierten Klassen
    if ontology_namespace is None:
        prefix_counts = defaultdict(int)
        for entity in own_entities:
            text = str(entity)
            cut = max(text.rfind("#"), text.rfind("/"))
            if cut > 0:
                prefix_counts[text[:cut + 1]] += 1
        ontology_namespace = max(prefix_counts, key=prefix_counts.get) if prefix_counts else ""

    def is_own(uri):
        return str(uri).startswith(ontology_namespace) if ontology_namespace else True

    # ------------------------------------------------------------------
    # P38 / P39: Ontologie-Deklaration und Namespace
    # ------------------------------------------------------------------
    if not ontologies:
        record("P38", ["<keine owl:Ontology-Deklaration vorhanden>"])
    ontology_uri = next(iter(ontologies), None)
    if ontology_uri is None or isinstance(ontology_uri, BNode) or not str(ontology_uri):
        record("P39", ["<Ontologie-URI nicht eindeutig deklariert>"])

    # ------------------------------------------------------------------
    # P41: Lizenz
    # ------------------------------------------------------------------
    has_license = any(
        (subject, predicate, None) in graph
        for subject in ontologies
        for predicate in LICENSE_PROPERTIES
    )
    if ontologies and not has_license:
        record("P41", [str(ontology_uri)])

    # ------------------------------------------------------------------
    # P36: Dateiendung in der URI
    # ------------------------------------------------------------------
    extensions = (".owl", ".rdf", ".ttl", ".n3", ".rdfxml", ".xml")
    record("P36", sorted(
        str(entity) for entity in (own_entities | set(ontologies))
        if isinstance(entity, URIRef) and str(entity).lower().rstrip("/").endswith(extensions)
    ))

    # ------------------------------------------------------------------
    # P03: Relation namens "is"
    # ------------------------------------------------------------------
    record("P03", sorted(
        str(p) for p in declared_properties
        if _local_name(p).lower() in {"is", "isa", "is_a", "is-a"}
    ))

    # ------------------------------------------------------------------
    # P11 / P19: Domain und Range
    # ------------------------------------------------------------------
    missing_domain_range, multiple_domain_range = [], []
    for prop in sorted(obj_props | data_props, key=str):
        if not is_own(prop):
            continue
        domains = list(graph.objects(prop, RDFS.domain))
        ranges = list(graph.objects(prop, RDFS.range))
        if not domains or not ranges:
            missing_domain_range.append(str(prop))
        if len(domains) > 1 or len(ranges) > 1:
            multiple_domain_range.append(str(prop))
    record("P11", missing_domain_range)
    record("P19", multiple_domain_range)

    # ------------------------------------------------------------------
    # P13 / P25 / P26: inverse Relationen
    # ------------------------------------------------------------------
    symmetric = set(graph.subjects(RDF.type, OWL.SymmetricProperty))
    has_inverse, inverse_of_itself, inverse_for_symmetric = set(), [], []
    for subject, obj in graph.subject_objects(OWL.inverseOf):
        has_inverse.add(subject)
        has_inverse.add(obj)
        if subject == obj:
            inverse_of_itself.append(str(subject))
        if subject in symmetric or obj in symmetric:
            inverse_for_symmetric.append(str(subject))
    record("P13", sorted(
        str(p) for p in obj_props
        if is_own(p) and p not in has_inverse and p not in symmetric
    ))
    record("P25", sorted(set(inverse_of_itself)))
    record("P26", sorted(set(inverse_for_symmetric)))

    # ------------------------------------------------------------------
    # P10: Disjunktheit
    # ------------------------------------------------------------------
    has_disjointness = (
        any(True for _ in graph.subject_objects(OWL.disjointWith))
        or any(True for _ in graph.subjects(RDF.type, OWL.AllDisjointClasses))
    )
    if classes and not has_disjointness:
        record("P10", ["<keine Disjunktheitsaxiome in der Ontologie>"])

    # ------------------------------------------------------------------
    # P06 / P24: Zyklen und rekursive Definitionen
    # ------------------------------------------------------------------
    parents = defaultdict(set)
    for sub, sup in graph.subject_objects(RDFS.subClassOf):
        if isinstance(sub, URIRef) and isinstance(sup, URIRef):
            parents[sub].add(sup)

    def reaches(start, target, seen=None):
        seen = seen or set()
        for parent in parents.get(start, ()):
            if parent == target:
                return True
            if parent not in seen:
                seen.add(parent)
                if reaches(parent, target, seen):
                    return True
        return False

    cycles = sorted(str(c) for c in parents if reaches(c, c))
    record("P06", cycles)

    recursive = set(cycles)
    for subject, obj in graph.subject_objects(OWL.equivalentClass):
        if subject == obj:
            recursive.add(str(subject))
    for prop in obj_props | data_props:
        if prop in set(graph.objects(prop, RDFS.domain)) | set(graph.objects(prop, RDFS.range)):
            recursive.add(str(prop))
    record("P24", sorted(recursive))

    # ------------------------------------------------------------------
    # P08: fehlende Annotationen
    # ------------------------------------------------------------------
    unannotated = []
    for entity in sorted(own_entities, key=str):
        if not is_own(entity):
            continue
        has_label = any((entity, p, None) in graph for p in LABEL_PROPERTIES)
        has_definition = any((entity, p, None) in graph for p in definition_properties)
        if not (has_label and has_definition):
            unannotated.append(str(entity))
    record("P08", unannotated)

    # ------------------------------------------------------------------
    # P32: identische Labels
    # ------------------------------------------------------------------
    by_label = defaultdict(list)
    for subject, obj in graph.subject_objects(RDFS.label):
        if subject in classes:
            by_label[str(obj).strip().lower()].append(str(subject))
    record("P32", sorted(
        entity for group in by_label.values() if len(group) > 1 for entity in group
    ))

    # ------------------------------------------------------------------
    # P21: Sammelklassen
    # ------------------------------------------------------------------
    keywords = ("misc", "other", "sonstige", "various", "miscellaneous")
    record("P21", sorted(
        str(c) for c in classes
        if is_own(c) and any(k in _local_name(c).lower() for k in keywords)
    ))

    # ------------------------------------------------------------------
    # P22: Namenskonventionen
    # ------------------------------------------------------------------
    styles = defaultdict(list)
    for entity in own_entities:
        if is_own(entity):
            styles[_naming_style(_local_name(entity))].append(str(entity))
    if len([s for s in styles if s != "other"]) > 1:
        smallest = min((s for s in styles if s != "other"), key=lambda s: len(styles[s]))
        record("P22", sorted(styles[smallest]))

    # ------------------------------------------------------------------
    # P33: Property-Chain mit nur einer Property
    # ------------------------------------------------------------------
    short_chains = []
    for subject, chain in graph.subject_objects(OWL.propertyChainAxiom):
        if len(list(graph.items(chain))) < 2:
            short_chains.append(str(subject))
    record("P33", sorted(set(short_chains)))

    # ------------------------------------------------------------------
    # P34 / P35: untypisierte Elemente
    # ------------------------------------------------------------------
    used_as_class = set()
    for sub, sup in graph.subject_objects(RDFS.subClassOf):
        used_as_class.update(x for x in (sub, sup) if isinstance(x, URIRef))
    for predicate in (RDFS.domain, RDFS.range, OWL.someValuesFrom,
                      OWL.allValuesFrom, OWL.onClass):
        used_as_class.update(o for o in graph.objects(None, predicate) if isinstance(o, URIRef))
    record("P34", sorted(
        str(x) for x in used_as_class
        if is_own(x) and x not in classes and x != OWL.Thing and x != OWL.Nothing
    ))

    used_as_property = {o for o in graph.objects(None, OWL.onProperty) if isinstance(o, URIRef)}
    used_as_property.update(s for s, _ in graph.subject_objects(RDFS.subPropertyOf)
                            if isinstance(s, URIRef))
    record("P35", sorted(
        str(x) for x in used_as_property if is_own(x) and x not in declared_properties
    ))

    # ------------------------------------------------------------------
    # P04: unverbundene Elemente
    # ------------------------------------------------------------------
    connected = set(used_as_class) | set(used_as_property)
    for predicate in (RDFS.domain, RDFS.range):
        connected.update(s for s, _ in graph.subject_objects(predicate))
    connected.update(o for o in graph.objects(None, OWL.equivalentClass) if isinstance(o, URIRef))
    connected.update(s for s, _ in graph.subject_objects(OWL.equivalentClass))
    record("P04", sorted(
        str(e) for e in own_entities if is_own(e) and e not in connected
    ))

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    counts = {"critical": 0, "important": 0, "minor": 0}
    detail = []
    for pid in sorted(findings, key=lambda x: int(x[1:])):
        severity = SEVERITY[pid]
        counts[severity] += 1
        detail.append({
            "pitfall": pid,
            "titel": TITLES[pid],
            "schweregrad": severity,
            "betroffene_elemente": len(findings[pid]),
            "beispiele": [_local_name(e) for e in findings[pid][:5]],
        })

    return {
        "namespace": ontology_namespace,
        "geprueft": len(SEVERITY),
        "gefunden": len(findings),
        "critical": counts["critical"],
        "important": counts["important"],
        "minor": counts["minor"],
        "details": detail,
    }


def compute_pitfall_score(critical, important_count, minor_count):
    """Unveraenderte Formel aus dem Ontologie-Evaluator."""
    if critical:
        return 10.0
    important_score = max(0.0, 100.0 - important_count * 10.0)
    minor_score = max(0.0, 100.0 - minor_count * 2.0)
    return float(min(important_score, minor_score))


if __name__ == "__main__":
    import json
    import sys

    for path in sys.argv[1:]:
        result = scan_pitfalls(path)
        result["pitfall_score"] = compute_pitfall_score(
            result["critical"] > 0, result["important"], result["minor"]
        )
        print("=" * 70)
        print(path)
        print(json.dumps(result, indent=2, ensure_ascii=False))
