"""
td2 phase 1 - build the rdf knowledge graph from our ner csv output
domain: space exploration
"""

import csv
import re
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD
from rdflib.namespace import SKOS
from pathlib import Path

# namespaces we use throughout the graph
SPACE = Namespace("http://space-exploration.org/ontology#")
SPACE_ENT = Namespace("http://space-exploration.org/entity/")
WIKI = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")
DBP = Namespace("http://dbpedia.org/resource/")
SCHEMA = Namespace("http://schema.org/")


def slugify(text: str) -> str:
    """turn text into a safe uri slug"""
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:80]


def build_ontology(g: Graph):
    """set up the ontology classes and object properties in the graph"""
    # declare all our classes
    for cls_name in ["SpaceMission", "SpaceAgency", "Astronaut", "Telescope",
                     "Spacecraft", "Location", "Organization", "Person",
                     "SpaceProgram", "LaunchVehicle"]:
        cls_uri = SPACE[cls_name]
        g.add((cls_uri, RDF.type, OWL.Class))
        g.add((cls_uri, RDFS.label, Literal(cls_name, lang="en")))

    # subclass hierarchy
    g.add((SPACE["Astronaut"], RDFS.subClassOf, SPACE["Person"]))
    g.add((SPACE["SpaceAgency"], RDFS.subClassOf, SPACE["Organization"]))
    g.add((SPACE["Telescope"], RDFS.subClassOf, SPACE["Spacecraft"]))

    # object properties with domain/range
    props = [
        ("launchedBy", "SpaceMission", "SpaceAgency"),
        ("crewedBy", "SpaceMission", "Astronaut"),
        ("builtBy", "Spacecraft", "Organization"),
        ("operatedBy", "Spacecraft", "SpaceAgency"),
        ("locatedIn", "Organization", "Location"),
        ("partOf", "SpaceMission", "SpaceProgram"),
        ("commandedBy", "SpaceMission", "Astronaut"),
        ("fundedBy", "SpaceProgram", "Organization"),
    ]
    for prop_name, domain, range_ in props:
        prop_uri = SPACE[prop_name]
        g.add((prop_uri, RDF.type, OWL.ObjectProperty))
        g.add((prop_uri, RDFS.domain, SPACE[domain]))
        g.add((prop_uri, RDFS.range, SPACE[range_]))
        g.add((prop_uri, RDFS.label, Literal(prop_name, lang="en")))

    # datatype properties (mostly for missions)
    dataprops = ["launchDate", "foundedYear", "budget", "orbitAltitude", "missionDuration"]
    for dp in dataprops:
        g.add((SPACE[dp], RDF.type, OWL.DatatypeProperty))
        g.add((SPACE[dp], RDFS.domain, SPACE["SpaceMission"]))
        g.add((SPACE[dp], RDFS.label, Literal(dp, lang="en")))

    print("\u2713 Ontology built")


# mapping spacy entity types to our ontology classes
SPACY_TO_ONTOLOGY = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Location",
    "LOC": "Location",
    "PRODUCT": "Spacecraft",
    "DATE": None,        # dates are literals, not entities
    "MONEY": None,
    "PERCENT": None,
}

# hand-curated entities we know are in the space domain
KNOWN_ENTITIES = {
    # astronauts
    "Neil Armstrong": ("Astronaut", "Q1615"),
    "Buzz Aldrin": ("Astronaut", "Q111283"),
    "Michael Collins": ("Astronaut", "Q316852"),
    "Yuri Gagarin": ("Astronaut", "Q82438"),
    "Valentina Tereshkova": ("Astronaut", "Q62782"),
    "Edwin Hubble": ("Person", "Q44711"),
    "James Webb": ("Person", "Q465899"),
    "Chris Hadfield": ("Astronaut", "Q56002"),
    # agencies
    "NASA": ("SpaceAgency", "Q23548"),
    "ESA": ("SpaceAgency", "Q5", ),
    "European Space Agency": ("SpaceAgency", "Q5"),
    "Roscosmos": ("SpaceAgency", "Q209213"),
    "ISRO": ("SpaceAgency", "Q875744"),
    "JAXA": ("SpaceAgency", "Q167751"),
    "SpaceX": ("Organization", "Q193701"),
    "Boeing": ("Organization", "Q66"),
    "Lockheed Martin": ("Organization", "Q66526"),
    "Northrop Grumman": ("Organization", "Q1454442"),
    # missions and programs
    "Apollo 11": ("SpaceMission", "Q43653"),
    "Apollo": ("SpaceProgram", "Q183952"),
    "Artemis": ("SpaceProgram", "Q60743"),
    "Hubble Space Telescope": ("Telescope", "Q2513"),
    "James Webb Space Telescope": ("Telescope", "Q184252"),
    "International Space Station": ("Spacecraft", "Q18"),
    "ISS": ("Spacecraft", "Q18"),
    "Ariane": ("LaunchVehicle", "Q179619"),
    "Ariane 5": ("LaunchVehicle", "Q43652"),
    "Space Shuttle": ("Spacecraft", "Q183952"),
    "Voyager 1": ("Spacecraft", "Q48979"),
    "Mars Rover": ("Spacecraft", "Q208441"),
    "Curiosity": ("Spacecraft", "Q208441"),
    "Perseverance": ("Spacecraft", "Q27396516"),
}

# known triples from our dependency parsing results
KNOWN_RELATIONS = [
    ("NASA", "launchedBy", "Apollo 11"),
    ("Apollo 11", "crewedBy", "Neil Armstrong"),
    ("Apollo 11", "crewedBy", "Buzz Aldrin"),
    ("Apollo 11", "crewedBy", "Michael Collins"),
    ("Apollo 11", "commandedBy", "Neil Armstrong"),
    ("Apollo 11", "partOf", "Apollo"),
    ("Hubble Space Telescope", "operatedBy", "NASA"),
    ("Hubble Space Telescope", "builtBy", "Lockheed Martin"),
    ("James Webb Space Telescope", "operatedBy", "NASA"),
    ("James Webb Space Telescope", "builtBy", "Northrop Grumman"),
    ("International Space Station", "operatedBy", "NASA"),
    ("International Space Station", "operatedBy", "Roscosmos"),
    ("International Space Station", "operatedBy", "ESA"),
    ("ESA", "fundedBy", "European Space Agency"),
    ("Artemis", "launchedBy", "NASA"),
    ("Ariane 5", "operatedBy", "ESA"),
]


def load_entities_from_csv(csv_path: str) -> list:
    """load extracted entities from the ner output csv"""
    entities = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entities.append(row)
        print(f"\u2713 Loaded {len(entities)} entities from {csv_path}")
    except FileNotFoundError:
        print(f"\u26a0 {csv_path} not found \u2014 using known entities only")
    return entities


def add_entity_to_graph(g: Graph, name: str, etype: str, wikidata_qid: str = None):
    """add a single entity to the rdf graph, linking to wikidata if we know the qid"""
    slug = slugify(name)
    uri = SPACE_ENT[slug]

    onto_class = SPACY_TO_ONTOLOGY.get(etype, "Person")
    if onto_class is None:
        return uri  # skip dates, money, etc.

    # override class if we have a hand-coded entry for this entity
    if name in KNOWN_ENTITIES:
        onto_class, qid = KNOWN_ENTITIES[name]
        if wikidata_qid is None and qid:
            wikidata_qid = qid

    g.add((uri, RDF.type, SPACE[onto_class]))
    g.add((uri, RDFS.label, Literal(name, lang="en")))

    if wikidata_qid:
        g.add((uri, OWL.sameAs, WIKI[wikidata_qid]))

    return uri


def build_initial_kg(entities_csv: str = "extracted_knowledge.csv",
                     output_ttl: str = "kg_artifacts/space_kg.ttl"):
    """main function - build the initial private kb and save it as turtle"""
    g = Graph()
    g.bind("space", SPACE)
    g.bind("ent", SPACE_ENT)
    g.bind("wd", WIKI)
    g.bind("wdt", WDT)
    g.bind("dbp", DBP)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("schema", SCHEMA)

    # step 1: define the ontology
    build_ontology(g)

    # step 2: add all the hand-coded known entities
    entity_uris = {}
    for name, (etype, qid) in KNOWN_ENTITIES.items():
        uri = add_entity_to_graph(g, name, etype, qid)
        entity_uris[name] = uri
    print(f"\u2713 Added {len(KNOWN_ENTITIES)} known entities")

    # step 3: add whatever came out of the ner csv
    csv_entities = load_entities_from_csv(entities_csv)
    added_from_csv = 0
    seen = set(KNOWN_ENTITIES.keys())
    for row in csv_entities:
        name = row.get("text", "").strip()
        etype = row.get("type", "ORG")
        if name and name not in seen and len(name) > 3:
            uri = add_entity_to_graph(g, name, etype)
            if uri:
                entity_uris[name] = uri
                seen.add(name)
                added_from_csv += 1
    print(f"\u2713 Added {added_from_csv} entities from CSV")

    # step 4: add the known relations between entities
    for subj_name, pred, obj_name in KNOWN_RELATIONS:
        if subj_name in entity_uris and obj_name in entity_uris:
            g.add((entity_uris[subj_name], SPACE[pred], entity_uris[obj_name]))
    print(f"\u2713 Added {len(KNOWN_RELATIONS)} relations")

    # step 5: sprinkle in some literal data (dates, altitudes, etc)
    literal_facts = [
        ("Apollo 11", "launchDate", "1969-07-16", XSD.date),
        ("Apollo", "foundedYear", "1961", XSD.gYear),
        ("Hubble Space Telescope", "orbitAltitude", "547", XSD.integer),
        ("International Space Station", "orbitAltitude", "408", XSD.integer),
        ("Artemis", "foundedYear", "2017", XSD.gYear),
    ]
    for name, prop, val, dtype in literal_facts:
        if name in entity_uris:
            g.add((entity_uris[name], SPACE[prop], Literal(val, datatype=dtype)))

    # step 6: serialize to disk
    Path(output_ttl).parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=output_ttl, format="turtle")
    print(f"\n\u2713 Initial KB saved: {output_ttl}")
    print(f"  Triples: {len(g)}")
    return g


if __name__ == "__main__":
    g = build_initial_kg(
        entities_csv="extracted_knowledge.csv",
        output_ttl="kg_artifacts/space_kg.ttl"
    )
    print(f"\nTotal triples in initial KB: {len(g)}")
