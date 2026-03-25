"""
phase 2 - information extraction pipeline
ner + dependency parsing to pull out entities and relations
"""

import spacy
import json
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple

class InformationExtractor:
    """extracts entities and relations from text using spacy"""

    def __init__(self, model_name: str = "en_core_web_trf"):
        """load the spacy transformer model"""
        print(f"loading {model_name}...")
        self.nlp = spacy.load(model_name)
        print("model loaded")

        # entity types we care about
        self.target_entities = {
            'PERSON', 'ORG', 'GPE', 'LOC',
            'DATE', 'MONEY', 'PERCENT', 'PRODUCT'
        }

        self.extracted_entities = []
        self.extracted_relations = []

    def filter_common_nouns(self, ent) -> bool:
        """filter out short or generic entities we don't want"""
        if len(ent.text) < 3:
            return False

        if ent.text.lower() in {'the', 'this', 'that', 'these', 'those'}:
            return False

        return True

    def extract_entities(self, text: str, url: str) -> List[Dict]:
        """run ner on text and return the entities we care about"""
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            if ent.label_ in self.target_entities:
                if self.filter_common_nouns(ent):
                    entity_data = {
                        'text': ent.text,
                        'type': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'source_url': url
                    }
                    entities.append(entity_data)

        return entities

    def extract_relations_from_sentence(self, sent) -> List[Tuple]:
        """
        extract relations from a sentence using dependency parsing.
        looks for subject -> verb -> object patterns.
        """
        relations = []

        for token in sent:
            if token.pos_ == "VERB":
                # find the grammatical subject
                subjects = [child for child in token.children
                           if child.dep_ == "nsubj"]

                # find the object (direct, prepositional, or attribute)
                objects = [child for child in token.children
                          if child.dep_ in ("dobj", "pobj", "attr")]

                # only keep pairs where both sides are named entities
                for subj in subjects:
                    for obj in objects:
                        if subj.ent_type_ and obj.ent_type_:
                            relations.append((
                                subj.text,
                                token.lemma_,  # use lemma so "launched" and "launches" map to the same thing
                                obj.text,
                                subj.ent_type_,
                                obj.ent_type_
                            ))

        return relations

    def extract_relations(self, text: str, url: str) -> List[Dict]:
        """extract all relations from a full text by processing sentence by sentence"""
        doc = self.nlp(text)
        relations = []

        for sent in doc.sents:
            sent_relations = self.extract_relations_from_sentence(sent)

            for subj, verb, obj, subj_type, obj_type in sent_relations:
                relation_data = {
                    'subject': subj,
                    'subject_type': subj_type,
                    'predicate': verb,
                    'object': obj,
                    'object_type': obj_type,
                    'sentence': sent.text,
                    'source_url': url
                }
                relations.append(relation_data)

        return relations

    def process_jsonl(self, input_file: str):
        """run extraction on all documents from the crawler jsonl output"""
        print(f"\ninfo extraction from {input_file}")

        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                data = json.loads(line)
                url = data['url']
                text = data['text']
                title = data.get('title', 'no title')

                print(f"\n[{i}] processing: {title}")
                print(f"    URL: {url}")

                entities = self.extract_entities(text, url)
                self.extracted_entities.extend(entities)
                print(f"    \u2713 {len(entities)} entities extracted")

                relations = self.extract_relations(text, url)
                self.extracted_relations.extend(relations)
                print(f"    \u2713 {len(relations)} relations extracted")

        print(f"\nextraction finished")
        print(f"Total: {len(self.extracted_entities)} entities, "
              f"{len(self.extracted_relations)} relations")

    def save_entities_to_csv(self, output_file: str = "extracted_knowledge.csv"):
        """save extracted entities to csv"""
        df = pd.DataFrame(self.extracted_entities)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Entities saved to {output_file}")

    def save_relations_to_csv(self, output_file: str = "extracted_relations.csv"):
        """save extracted relations to csv"""
        df = pd.DataFrame(self.extracted_relations)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Relations saved to {output_file}")

    def display_statistics(self):
        """show some stats on what we extracted"""
        print("\n" + "="*60)
        print("extraction statistics")
        print("="*60)

        if self.extracted_entities:
            df_ent = pd.DataFrame(self.extracted_entities)
            print("\nentity type breakdown:")
            print(df_ent['type'].value_counts())

            print(f"\ntop 10 most frequent entities:")
            print(df_ent['text'].value_counts().head(10))

        if self.extracted_relations:
            df_rel = pd.DataFrame(self.extracted_relations)
            print(f"\ntop 10 most frequent predicates:")
            print(df_rel['predicate'].value_counts().head(10))

            print(f"\nexample triples:")
            for i, row in df_rel.head(5).iterrows():
                print(f"  ({row['subject']}) --[{row['predicate']}]--> ({row['object']})")


if __name__ == "__main__":

    extractor = InformationExtractor()

    # process the crawler output from phase 1
    extractor.process_jsonl("crawler_output.jsonl")

    extractor.save_entities_to_csv("extracted_knowledge.csv")
    extractor.save_relations_to_csv("extracted_relations.csv")

    extractor.display_statistics()

    print("\npipeline finished")
    print("generated files:")
    print("    extracted_knowledge.csv (entities)")
    print("    extracted_relations.csv (relations)")
