"""
Phase 2: Information Extraction Pipeline
NER + Dependency parsing for relationship extraction
"""

import spacy
import json
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple

class InformationExtractor:
    """
    extractor of entity  and relationships from text
    """
    
    def __init__(self, model_name: str = "en_core_web_trf"):
        """
        initialize spaCy transformer model
        """
        print(f"loading {model_name}...")
        self.nlp = spacy.load(model_name)
        print("model charged")
        
        # types of entities to extract based on cm
        self.target_entities = {
            'PERSON', 'ORG', 'GPE', 'LOC', 
            'DATE', 'MONEY', 'PERCENT', 'PRODUCT'
        }
        
        self.extracted_entities = []
        self.extracted_relations = []
    
    def filter_common_nouns(self, ent) -> bool:
        """
        filters common nouns to keep only specific entities
        """
        # avoid too short or generic entities
        if len(ent.text) < 3:
            return False
        
        # stop words
        if ent.text.lower() in {'the', 'this', 'that', 'these', 'those'}:
            return False
        
        return True
    
    def extract_entities(self, text: str, url: str) -> List[Dict]:
        """
       extract entities from text
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            # only keep pertinent entities
            if ent.label_ in self.target_entities:
                # filter common nouns
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
        extract relatoonships from a sentence via dependency parsing
        seeks patterns: subject/ verbe / object
        """
        relations = []
        
        # looks for principal verbs
        for token in sent:
            if token.pos_ == "VERB":
                # finds subject sentance (nsubj)
                subjects = [child for child in token.children 
                           if child.dep_ == "nsubj"]
                
                # finds object (dobj, pobj, attr)
                objects = [child for child in token.children 
                          if child.dep_ in ("dobj", "pobj", "attr")]
                
                # triples for each combo
                for subj in subjects:
                    for obj in objects:
                        # verify subj & obj are entities
                        if subj.ent_type_ and obj.ent_type_:
                            relations.append((
                                subj.text,
                                token.lemma_,  # lammatized form
                                obj.text,
                                subj.ent_type_,
                                obj.ent_type_
                            ))
        
        return relations
    
    def extract_relations(self, text: str, url: str) -> List[Dict]:
        """
        Extract relationships btw entities
        """
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
        """
        examine all docs from JSONL file
        """
        print(f"\n info extraction from {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                data = json.loads(line)
                url = data['url']
                text = data['text']
                title = data.get('title', 'no title')
                
                print(f"\n[{i}] loeading: {title}")
                print(f"    URL: {url}")
                
                #entities extraction
                entities = self.extract_entities(text, url)
                self.extracted_entities.extend(entities)
                print(f"    ✓ {len(entities)} entités extraites")
                
                # relationship extraction
                relations = self.extract_relations(text, url)
                self.extracted_relations.extend(relations)
                print(f"    ✓ {len(relations)} relations extraites")
        
        print(f"\ extraction finished")
        print(f"Total: {len(self.extracted_entities)} entities, "
              f"{len(self.extracted_relations)} relations")
    
    def save_entities_to_csv(self, output_file: str = "extracted_knowledge.csv"):
        """
        save entities in a csv
        """
        df = pd.DataFrame(self.extracted_entities)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f" Entities saved in {output_file}")
    
    def save_relations_to_csv(self, output_file: str = "extracted_relations.csv"):
        """
        save relations in a csv
        """
        df = pd.DataFrame(self.extracted_relations)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"relationships saved in {output_file}")
    
    def display_statistics(self):
        """
        displays statistics on extracted data
        """
        print("\n" + "="*60)
        print("extraction statistics")
        print("="*60)
        
        # entities stats
        if self.extracted_entities:
            df_ent = pd.DataFrame(self.extracted_entities)
            print("\ntype repartition:")
            print(df_ent['type'].value_counts())
            
            print(f"\n Top 10 most frquent:")
            print(df_ent['text'].value_counts().head(10))
        
        # relationship stats
        if self.extracted_relations:
            df_rel = pd.DataFrame(self.extracted_relations)
            print(f"\nTop 10 mosrt frequent:")
            print(df_rel['predicate'].value_counts().head(10))
            
            print(f"\ntriple example:")
            for i, row in df_rel.head(5).iterrows():
                print(f"  ({row['subject']}) --[{row['predicate']}]--> ({row['object']})")


# ============================================
# EX
# ============================================

if __name__ == "__main__":
    
    # extractor
    extractor = InformationExtractor()
    
    # phase 1 json file
    extractor.process_jsonl("crawler_output.jsonl")

    extractor.save_entities_to_csv("extracted_knowledge.csv")
    extractor.save_relations_to_csv("extracted_relations.csv")
    
    extractor.display_statistics()
    
    print("\n pipeline finished")
    print("generated files::")
    print("    extracted_knowledge.csv (entities)")
    print("    extracted_relations.csv (relations)")