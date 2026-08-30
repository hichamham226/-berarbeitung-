"""
OOPS!-Client für den Ontologie-Evaluator.
Endpunkt und Antwortformat gemäß https://oops.linkeddata.es/webservice.html
Schweregrade gemäß https://oops.linkeddata.es/catalogue.jsp
"""

import re
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

OOPS_ENDPOINT = "https://oops.linkeddata.es/rest"
OOPS_NS = {"oops": "http://www.oeg-upm.net/oops"}

IMPORTANCE = {
    "P01": "critical",  "P02": "minor",     "P03": "critical",  "P04": "minor",
    "P05": "critical",  "P06": "critical",  "P07": "minor",     "P08": "minor",
    "P09": "minor",     "P10": "important", "P11": "important", "P12": "important",
    "P13": "minor",     "P14": "critical",  "P15": "critical",  "P16": "critical",
    "P17": "important", "P18": "important", "P19": "critical",  "P20": "minor",
    "P21": "minor",     "P22": "minor",     "P23": "important", "P24": "important",
    "P25": "important", "P26": "important", "P27": "critical",  "P28": "critical",
    "P29": "critical",  "P30": "important", "P31": "critical",  "P32": "minor",
    "P33": "minor",     "P34": "important", "P35": "important", "P36": "minor",
    "P37": "critical",  "P38": "important", "P39": "critical",  "P40": "critical",
    "P41": "important",
}

IGNORED_CODES = {"P34"}

REQUEST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<OOPSRequest>
<OntologyUrl>{url}</OntologyUrl>
<OntologyContent>{content}</OntologyContent>
<Pitfalls></Pitfalls>
<OutputFormat>XML</OutputFormat>
</OOPSRequest>"""


def _lokaler_name(element):
    """Gibt den Elementnamen ohne Namensraum zurück."""
    return element.tag.split("}")[-1].lower()


def _finde_alle(element, name):
    """Sucht rekursiv alle Kindelemente mit dem gegebenen lokalen Namen."""
    name = name.lower()
    treffer = []
    for kind in element.iter():
        if _lokaler_name(kind) == name:
            treffer.append(kind)
    return treffer


def _normalisiere_code(text):
    """'p4', 'P04.', 'Pitfall 4' -> 'P04'. Gibt None zurück, wenn nichts passt."""
    if not text:
        return None
    treffer = re.search(r"[Pp]\s*0*(\d{1,2})", text)
    if not treffer:
        return None
    return "P{:02d}".format(int(treffer.group(1)))


def _werte_antwort_aus(rohtext):
    try:
        wurzel = ET.fromstring(rohtext)
    except ET.ParseError:
        return None

    zaehlung = {"critical": 0, "important": 0, "minor": 0}
    details = []
    gesehen = set()

    for pitfall in _finde_alle(wurzel, "pitfall"):
        code = None
        for feld in ("code", "name", "title"):
            kinder = _finde_alle(pitfall, feld)
            if kinder and kinder[0].text:
                code = _normalisiere_code(kinder[0].text)
                if code:
                    break
        if code is None:
            code = _normalisiere_code(pitfall.get("code") or "")
        if code is None or code in gesehen or code in IGNORED_CODES:
            continue
        gesehen.add(code)

        schweregrad = None
        for feld in ("importance", "importancelevel", "level", "severity"):
            kinder = _finde_alle(pitfall, feld)
            if kinder and kinder[0].text:
                wert = kinder[0].text.strip().lower()
                if "critic" in wert:
                    schweregrad = "critical"
                elif "import" in wert:
                    schweregrad = "important"
                elif "minor" in wert:
                    schweregrad = "minor"
                break
        if schweregrad is None:
            schweregrad = IMPORTANCE.get(code, "minor")

        betroffen = len(_finde_alle(pitfall, "affectedelement"))
        name_kinder = _finde_alle(pitfall, "name")
        zaehlung[schweregrad] += 1
        details.append({
            "pitfall": code,
            "name": (name_kinder[0].text or "").strip() if name_kinder else "",
            "schweregrad": schweregrad,
            "betroffene_elemente": betroffen,
        })

    if not details:
        return None

    return {
        "critical": zaehlung["critical"],
        "important": zaehlung["important"],
        "minor": zaehlung["minor"],
        "details": sorted(details, key=lambda d: d["pitfall"]),
        "rohantwort": rohtext,
    }


def run_oops_analysis(owl_text=None, ontology_url=None, timeout=180):
    """Ruft den OOPS!-Webservice auf.

    Entweder owl_text (RDF/XML als str) ODER ontology_url angeben.
    Rückgabe: dict mit Zählungen und Details, oder None bei Fehlschlag.
    WICHTIG: None bedeutet "nicht geprüft", NICHT "keine Pitfalls".
    """
    if not owl_text and not ontology_url:
        raise ValueError("Entweder owl_text oder ontology_url angeben.")

    content = f"<![CDATA[{owl_text}]]>" if owl_text else ""
    body = REQUEST_TEMPLATE.format(url=ontology_url or "", content=content)

    request = urllib.request.Request(
        OOPS_ENDPOINT,
        data=body.encode("utf-8"),
        headers={
            "Content-Type": "application/xml;charset=UTF-8",
            "Accept": "application/xml",
            "User-Agent": "Ontologie-Evaluator/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        return None

    return _werte_antwort_aus(raw)


def compute_pitfall_score(oops_result):
    """Score 0–100, oder None wenn OOPS! nicht erreichbar war."""
    if oops_result is None:
        return None
    if oops_result["critical"] > 0:
        return 10.0
    important_score = max(0.0, 100.0 - oops_result["important"] * 10.0)
    minor_score = max(0.0, 100.0 - oops_result["minor"] * 2.0)
    return float(min(important_score, minor_score))


IOF_URIS = {
    "CoreCore.rdf": "https://spec.industrialontologies.org/ontology/core/Core/",
    "ProductionPlanning.rdf": "https://spec.industrialontologies.org/ontology/productionplanning/ProductionPlanning/",
    "ProductServiceSystem.rdf": "https://spec.industrialontologies.org/ontology/productservicesystem/ProductServiceSystem/",
}


if __name__ == "__main__":
    import json
    import sys

    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        result = run_oops_analysis(owl_text=text)
        if result is None:
            print(f"{path}: OOPS! nicht erreichbar - Score bleibt UNBEKANNT")
            continue
        result.pop("rohantwort")
        result["pitfall_score"] = compute_pitfall_score(result)
        print(path)
        print(json.dumps(result, indent=2, ensure_ascii=False))
