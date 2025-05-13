import gradio as gr
import joblib
import pandas as pd
import re
from typing import Dict, List, Any


# Load trained pipeline
pipeline = joblib.load("models/fragma_ml.pkl")

def extract_features(text: str) -> Dict[str, bool]:
    """
    Extract linguistic features from text using regular expressions.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Dict[str, bool]: Dictionary with feature names as keys and boolean values
    """
    features = {
        'has_auxiliary': False,
        'has_fullstop': False,
        'has_question_mark': False,
        'has_exclamation_mark': False,
        'has_comma': False,
        'has_semicolon': False,
        'has_colon': False,
        'has_quotation': False,
        'has_expression': False,
        'has_conjunction': False,
        'has_temporal': False,
        'has_opinion_adverb': False,
        'has_adverb': False,
        'has_starter': False,
        'has_past_verb': False,
        'has_gerund': False,
        'starts_capitalized': False
    }
    
    if not text:
        return features
    
    # Check if text starts with capital letter
    features['starts_capitalized'] = bool(re.match(r'^[A-Z]', text))
    
    # Basic punctuation checks with regex
    features['has_fullstop'] = bool(re.search(r'\.', text))
    features['has_question_mark'] = bool(re.search(r'\?', text))
    features['has_exclamation_mark'] = bool(re.search(r'!', text))
    features['has_comma'] = bool(re.search(r',', text))
    features['has_semicolon'] = bool(re.search(r';', text))
    features['has_colon'] = bool(re.search(r':', text))
    features['has_quotation'] = bool(re.search(r'[\'"]', text))
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Common auxiliaries in English
    auxiliaries_pattern = r'\b(am|is|are|was|were|be|being|been|have|has|had|do|does|did|can|could|shall|should|will|would|may|might|must)\b'
    features['has_auxiliary'] = bool(re.search(auxiliaries_pattern, text_lower))
    
    # Common conjunctions
    conjunctions_pattern = r'\b(and|but|or|nor|for|yet|so|although|because|since|unless|while|where|if|then|than|that)\b'
    features['has_conjunction'] = bool(re.search(conjunctions_pattern, text_lower))
    
    # Common temporal expressions
    temporal_pattern = r'\b(today|tomorrow|yesterday|now|then|soon|later|before|after|while|during|until|never|always|often|sometimes|rarely)\b'
    features['has_temporal'] = bool(re.search(temporal_pattern, text_lower))
    
    # Common opinion adverbs
    opinion_adverbs_pattern = r'\b(fortunately|unfortunately|surprisingly|obviously|clearly|frankly|honestly|remarkably|interestingly|sadly|happily|strangely|certainly|undoubtedly|hopefully)\b'
    features['has_opinion_adverb'] = bool(re.search(opinion_adverbs_pattern, text_lower))
    
    # Common expressions (interjections)
    expressions_pattern = r'\b(oh|ah|wow|ouch|hey|hmm|umm|uh|oops|yay|hurray|alas|phew|whew|yikes)\b'
    features['has_expression'] = bool(re.search(expressions_pattern, text_lower))
    
    # Common starter words (discourse markers) - check at the beginning of the text or after punctuation
    starters_pattern = r'(?:^|\.\s|\?\s|!\s)(well|so|now|then|therefore|however|moreover|furthermore|nevertheless|anyway|firstly|secondly|finally|in conclusion|besides)\b'
    features['has_starter'] = bool(re.search(starters_pattern, text_lower))
    
    # Adverbs (ending in -ly is a common pattern for many adverbs)
    features['has_adverb'] = bool(re.search(r'\b\w+ly\b', text_lower))
    
    # Past tense verbs (common regular past tense pattern)
    # This is a simplification - won't catch all past verbs but will catch many regular ones
    features['has_past_verb'] = bool(re.search(r'\b\w+ed\b', text_lower))
    
    # Gerunds (verbs ending in -ing)
    features['has_gerund'] = bool(re.search(r'\b\w+ing\b', text_lower))
    
    return features


# Prediction wrapper for Gradio
def ml_predict(text_input):
    features = extract_features(text_input)
    input_data = {
        'Processed Text': [text_input],
        **{k: [v] for k, v in features.items()}
    }
    df = pd.DataFrame(input_data)
    prediction = pipeline.predict(df)[0]
    return str(prediction)

# Define input components for Gradio
ml_interface = gr.Interface(
    fn=ml_predict,
    inputs=gr.Textbox(label="Enter a sentence"),
    outputs=gr.Text(label="Prediction"),
    title="Fragment ML Classifier",
    description="Automatically extracts punctuation and verb features to predict."
)

