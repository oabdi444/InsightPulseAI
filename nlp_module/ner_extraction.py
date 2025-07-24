import re
from typing import List, Dict, Any, Tuple
import pandas as pd

class AdvancedPatternNER:
    """Advanced pattern-based Named Entity Recognition"""
    
    def __init__(self):
        self.patterns = {
            'PERSON': [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
                r'\b[A-Z][a-z]+\s+[A-Z]\.[A-Z][a-z]+\b',  # First M.Last
                r'\bMr\.?\s+[A-Z][a-z]+\b',  # Mr. Last
                r'\bMrs\.?\s+[A-Z][a-z]+\b',  # Mrs. Last
                r'\bDr\.?\s+[A-Z][a-z]+\b',  # Dr. Last
                r'\bChef\s+[A-Z][a-z]+\b',  # Chef Name
            ],
            'ORGANIZATION': [
                r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\s+(?:Inc|Corp|LLC|Ltd|Co|Company|Corporation|Group|Systems|Technologies|Solutions|Services)\b',
                r'\b[A-Z][a-z]+\s+(?:Restaurant|Cafe|Hotel|Store|Shop|Market|Bank|Hospital|School|University|College)\b',
                r'\b(?:Apple|Microsoft|Google|Amazon|Facebook|Tesla|Netflix|Uber|Twitter|Instagram)\s*Inc\.?\b',
                r"\b[A-Z][a-z]+'s\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # Maria's Italian Kitchen
            ],
            'LOCATION': [
                r'\b[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)\b',
                r'\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave)\b',  # 123 Oak Street
                r'\b(?:New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose|Austin|Jacksonville|Fort Worth|Columbus|Charlotte|San Francisco|Indianapolis|Seattle|Denver|Washington|Boston|Nashville|Baltimore|Louisville|Portland|Oklahoma City|Milwaukee|Las Vegas|Albuquerque|Tucson|Fresno|Sacramento|Long Beach|Kansas City|Mesa|Virginia Beach|Atlanta|Colorado Springs|Omaha|Raleigh|Miami|Oakland|Minneapolis|Tulsa|Cleveland|Wichita|Arlington)\b',
                r'\b(?:California|Texas|Florida|New York|Pennsylvania|Illinois|Ohio|Georgia|North Carolina|Michigan|New Jersey|Virginia|Washington|Arizona|Massachusetts|Tennessee|Indiana|Missouri|Maryland|Wisconsin|Colorado|Minnesota|South Carolina|Alabama|Louisiana|Kentucky|Oregon|Oklahoma|Connecticut|Utah|Iowa|Nevada|Arkansas|Mississippi|Kansas|New Mexico|Nebraska|West Virginia|Idaho|Hawaii|New Hampshire|Maine|Montana|Rhode Island|Delaware|South Dakota|North Dakota|Alaska|Vermont|Wyoming)\b',
                r'\b(?:United States|USA|US|America|Canada|Mexico|UK|United Kingdom|France|Germany|Italy|Spain|Japan|China|India|Australia|Brazil)\b',
            ],
            'EMAIL': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'PHONE': [
                r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
                r'\b[0-9]{3}-[0-9]{3}-[0-9]{4}\b'
            ],
            'DATE': [
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
            ],
            'MONEY': [
                r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b',
                r'\b\d+(?:\.\d{2})?\s?(?:dollars?|USD|euros?|EUR|pounds?|GBP)\b'
            ],
            'TIME': [
                r'\b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\s?[AaPp][Mm])?\b'
            ],
            'URL': [
                r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w)*)?)?'
            ]
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using pattern matching"""
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group().strip()
                    
                    # Filter out common false positives
                    if self._is_valid_entity(entity_text, entity_type):
                        entities.append({
                            'text': entity_text,
                            'label': entity_type,
                            'start': match.start(),
                            'end': match.end(),
                            'confidence': 0.8  # Pattern-based confidence
                        })
        
        # Remove duplicates and overlapping entities
        return self._remove_overlaps(entities)
    
    def _is_valid_entity(self, text: str, entity_type: str) -> bool:
        """Validate extracted entities"""
        # Remove very short entities
        if len(text) < 2:
            return False
        
        # Entity-specific validation
        if entity_type == 'PERSON':
            # Avoid common false positives
            invalid_persons = {'The', 'This', 'That', 'With', 'From', 'Into', 'Over'}
            if text in invalid_persons:
                return False
        
        elif entity_type == 'ORGANIZATION':
            # Must have at least one uppercase letter
            if not any(c.isupper() for c in text):
                return False
        
        return True
    
    def _remove_overlaps(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping entities, keeping the longer ones"""
        if not entities:
            return []
        
        # Sort by start position
        entities.sort(key=lambda x: x['start'])
        
        result = []
        for entity in entities:
            # Check for overlap with existing entities
            overlap_found = False
            for existing in result:
                if (entity['start'] < existing['end'] and 
                    entity['end'] > existing['start']):
                    # There's an overlap
                    if len(entity['text']) > len(existing['text']):
                        # Replace with longer entity
                        result.remove(existing)
                        result.append(entity)
                    overlap_found = True
                    break
            
            if not overlap_found:
                result.append(entity)
        
        return result

# Initialize pattern-based NER
pattern_ner = AdvancedPatternNER()

# Try to import transformers for advanced NER
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
    
    try:
        # Initialize transformer-based NER
        ner_pipeline = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
        )
    except Exception:
        try:
            # Fallback to smaller model
            ner_pipeline = pipeline("ner", aggregation_strategy="simple")
        except Exception:
            TRANSFORMERS_AVAILABLE = False
            
except ImportError:
    TRANSFORMERS_AVAILABLE = False

def extract_entities_pattern_based(text: str) -> List[Dict[str, Any]]:
    """Extract entities using pattern-based approach (always available)"""
    return pattern_ner.extract_entities(text)

def extract_entities_transformers(text: str) -> List[Dict[str, Any]]:
    """Extract entities using transformer models (if available)"""
    if not TRANSFORMERS_AVAILABLE:
        return extract_entities_pattern_based(text)
    
    try:
        # Truncate text if too long
        if len(text) > 512:
            text = text[:512]
        
        # Get entities from transformer
        entities = ner_pipeline(text)
        
        # Convert to standard format
        formatted_entities = []
        for entity in entities:
            # Map transformer labels to our labels
            label_mapping = {
                'PER': 'PERSON',
                'PERSON': 'PERSON',
                'ORG': 'ORGANIZATION', 
                'ORGANIZATION': 'ORGANIZATION',
                'LOC': 'LOCATION',
                'LOCATION': 'LOCATION',
                'MISC': 'MISCELLANEOUS'
            }
            
            original_label = entity.get('entity_group', entity.get('label', 'MISC'))
            mapped_label = label_mapping.get(original_label.upper(), original_label)
            
            formatted_entities.append({
                'text': entity['word'].replace('##', '').strip(),
                'label': mapped_label,
                'start': entity['start'],
                'end': entity['end'],
                'confidence': entity['score']
            })
        
        return formatted_entities
    
    except Exception as e:
        print(f"Transformer NER failed: {e}")
        return extract_entities_pattern_based(text)

def extract_entities_advanced(text: str, method: str = "advanced pattern-based") -> List[Dict[str, Any]]:
    """Extract entities using specified method"""
    if "transformers" in method.lower() and TRANSFORMERS_AVAILABLE:
        return extract_entities_transformers(text)
    else:
        return extract_entities_pattern_based(text)

def format_entities(entities: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format entities into a DataFrame"""
    if not entities:
        return pd.DataFrame(columns=['text', 'label', 'start', 'end', 'confidence'])
    
    return pd.DataFrame(entities)

def get_entity_summary(entities: List[Dict[str, Any]]) -> Dict[str, int]:
    """Get summary of entity types"""
    summary = {}
    for entity in entities:
        label = entity['label']
        summary[label] = summary.get(label, 0) + 1
    return summary

def extract_specific_entities(text: str, entity_types: List[str]) -> List[Dict[str, Any]]:
    """Extract only specific types of entities"""
    all_entities = extract_entities_advanced(text)
    return [entity for entity in all_entities if entity['label'] in entity_types]

# Default function for backward compatibility
def extract_entities(text: str) -> List[str]:
    """Default entity extraction function - returns list of entity texts"""
    entities = extract_entities_advanced(text)
    return [entity['text'] for entity in entities]