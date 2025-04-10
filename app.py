# -*- coding: utf-8 -*-
"""
TextPulse.ai Backend Flask Application (v13.9 - Optimized)

This script runs a Flask web server providing an API endpoint (/humanize)
to process text according to the TextPulse logic. It's designed for
deployment on a server (e.g., DigitalOcean) to backend a web application
(e.g., built with Bubble.io).

Key Optimizations Implemented:
- Models (spaCy, T5, RoBERTa) are loaded ONCE on application startup.
- Unused spaCy pipeline components ('ner') are disabled.
- The API endpoint returns only the final processed text and final metrics.

Workflow:
1. Server starts, loads all models into memory.
2. Receives POST requests at /humanize with JSON data:
   { "text": "...", "discipline": "...", "freeze_terms": [...] }
3. Processes the text using the loaded models and defined logic.
4. Returns JSON response:
   { "status": "success", "final_text": "...", "final_metrics": {...} }
   or an error message.

Deployment Note: Run using a production WSGI server like Gunicorn.
Example: gunicorn --bind 0.0.0.0:5000 app:app (replace app with your filename)
Ensure sufficient RAM on the server (8GB+ recommended).

WARNING: Contains hardcoded API key for testing - REMOVE BEFORE SHARING/COMMIT.
         Use environment variables or a secure configuration method in production. 123
"""

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import json
import logging
import os
import time
import sys
import re
import traceback # For detailed error logging
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, Tuple, Union

# Flask related imports
from flask import Flask, request, jsonify
from flask_cors import CORS
# *** Assuming python-dotenv is needed if you use .env file ***
# *** If you set env vars via systemd or other means, you can remove dotenv ***
#from dotenv import load_dotenv # To load .env file

# NLP/ML related imports
try:
    import spacy
    import torch
    import transformers
    import openai
    from textstat import flesch_kincaid_grade # Import specific function
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, set_seed
except ImportError as e:
    print(f"FATAL Error: Missing required library: {e}. Please install requirements.")
    print("Try: pip install Flask python-dotenv flask-cors spacy torch transformers openai textstat") # Added flask-cors and python-dotenv here
    # Exit if core dependencies are missing
    sys.exit(1)

# ==============================================================================
# Load .env file (immediately after imports)
# ==============================================================================
#load_dotenv() # Loads variables from .env into environment

# ==============================================================================
# 2. CONFIGURATION & CONSTANTS
# ==============================================================================
# --- Debug Flags ---
DEBUG_SPLITTING = False # Set True for detailed sentence splitting logs

# --- Model & Processing Parameters ---
CONFIG = {
    "spacy_model": "en_core_web_sm",
    "paraphraser_model": "humarin/chatgpt_paraphraser_on_T5_base",
    "ai_detector_model": "roberta-base-openai-detector",
    "default_device": "auto", # "auto", "cpu", "cuda"
    "paraphrase_num_beams": 5,
    "paraphrase_max_length": 512,
    "input_max_words": 1000, # Limit input word count (consider adjusting)
    "input_max_sentence_tokens": 480,
    "seed": 42,
    "openai_model": "gpt-3.5-turbo",
    "openai_temperature": 0.3,
    "openai_max_tokens": 1500
}

# --- Discipline Definitions ---
VALID_DISCIPLINES = {
    "comp_sci_eng", "med_health", "soc_sci", "humanities", "nat_sci",
    "biz_econ", "education", "law", "env_sci", "corporate",
    "blogger_seo", "marketing", "casual", "general"
}
DEFAULT_DISCIPLINE = "general"
ACADEMIC_DISCIPLINES = {
    "comp_sci_eng", "med_health", "soc_sci", "humanities", "nat_sci",
    "biz_econ", "education", "law", "env_sci"
}

# --- Other Constants ---
PLACEHOLDER_PATTERN = re.compile(r'__F\d+__')

# --- Set Seed for Reproducibility ---
set_seed(CONFIG["seed"])

# --- Configure Logging ---
# Basic logging setup (customize level and format as needed for production)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# 3. API KEY SETUP & OPENAI CLIENT INITIALIZATION
# ==============================================================================

# *** Reading key from environment variable ***
openai_api_key = os.environ.get("OPENAI_API_KEY")

openai_client = None
logging.info("Attempting to initialize OpenAI client...")
# Use the actual key variable here (either hardcoded or from env)
# openai_api_key = openai_api_key # This line is redundant, removed

if not openai_api_key or not openai_api_key.startswith("sk-"):
    logging.warning("OPENAI_API_KEY environment variable not found, invalid, or key format incorrect.")
    logging.warning("OpenAI refinement step will be skipped.")
else:
    try:
        openai_client = openai.OpenAI(api_key=openai_api_key)
        # Optional: Test connection (add timeout)
        # openai_client.with_options(timeout=10.0).models.list()
        logging.info("OpenAI client initialized successfully using environment variable.")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        logging.warning("OpenAI refinement step will be skipped.")
        openai_client = None

# ==============================================================================
# 4. HELPER FUNCTIONS: METRICS CALCULATION
# ==============================================================================
def calculate_burstiness(text: str) -> float:
    """Calculates the burstiness score of the text."""
    words = text.lower().split()
    if not words: return 0.0
    counts = {}
    for w in words: counts[w] = counts.get(w, 0) + 1
    freqs = list(counts.values())
    if not freqs: return 0.0
    num_freqs = len(freqs)
    mean = sum(freqs) / num_freqs
    if mean == 0: return 0.0
    variance = sum([(f - mean) ** 2 for f in freqs]) / num_freqs
    std_dev = variance ** 0.5
    return std_dev / mean

def calculate_modification_percentage(text1: str, text2: str) -> float:
    """Calculates the percentage of modification between two texts."""
    text1 = text1 or ""; text2 = text2 or ""
    if not text1 and not text2: return 0.0
    if not text1 or not text2: return 100.0
    similarity_ratio = SequenceMatcher(None, text1, text2).ratio()
    return (1.0 - similarity_ratio) * 100.0

def calculate_ttr(text: str) -> float:
    """Calculates the Type-Token Ratio (TTR) for lexical diversity."""
    if not text or not text.strip(): return 0.0
    words = [token for token in text.lower().split() if token.isalnum()]
    total_tokens = len(words)
    if total_tokens == 0: return 0.0
    unique_types = len(set(words))
    return unique_types / total_tokens

# ==============================================================================
# 5. HELPER FUNCTIONS: PARAPHRASING
# ==============================================================================
def paraphrase_sentence(
    sentence_with_placeholders: str, model: Any, tokenizer: Any, num_beams: int, max_length: int,
    no_repeat_ngram_size: int = 2, repetition_penalty: float = 1.2
) -> str:
    """Paraphrases a single sentence using the loaded T5-based model."""
    if not sentence_with_placeholders or not sentence_with_placeholders.strip():
        logging.warning("Attempted to paraphrase an empty sentence.")
        return ""
    try:
        input_text = f"paraphrase: {sentence_with_placeholders}"
        input_ids = tokenizer.encode(
            input_text, return_tensors="pt", max_length=max_length, truncation=True
        ).to(model.device)
        if input_ids.shape[1] == 0:
            logging.error(f"Encoding resulted in empty tensor for sentence: '{sentence_with_placeholders[:50]}...'")
            return "[Encoding Error]"
        outputs = model.generate(
            input_ids=input_ids, num_beams=num_beams, max_length=max_length, early_stopping=True,
            no_repeat_ngram_size=no_repeat_ngram_size, repetition_penalty=repetition_penalty
        )
        paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return paraphrased
    except Exception as e:
        logging.error(f"Paraphrasing error on sentence: '{sentence_with_placeholders[:50]}...': {e}", exc_info=True)
        return f"[Paraphrasing Error: {e}]"

# ==============================================================================
# 6. HELPER FUNCTIONS: SENTENCE SPLITTING
# ==============================================================================
def has_subject_and_verb(doc_span: spacy.tokens.Span) -> bool:
    """Checks if a spaCy span likely contains an explicit subject and a main verb."""
    if not doc_span or len(doc_span) == 0: return False
    span_has_verb, span_has_subj = False, False; verb_indices_in_span = set()
    for token in doc_span:
        if token.pos_ in ("VERB", "AUX"): span_has_verb = True; verb_indices_in_span.add(token.i)
    if not span_has_verb: return False
    for token in doc_span:
        if token.dep_ in ("nsubj", "nsubjpass") and token.head.i in verb_indices_in_span: span_has_subj = True; break
        elif token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX") and token.i in verb_indices_in_span:
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass") and child.i >= doc_span.start and child.i < doc_span.end: span_has_subj = True; break
            if span_has_subj: break
    # Debugging print (conditional)
    if DEBUG_SPLITTING:
        span_text = doc_span.text[:75] + ('...' if len(doc_span.text) > 75 else '')
        logging.debug(f"    DEBUG S/V Check: Span='{span_text}', HasSubj={span_has_subj}, HasVerb={span_has_verb}")
    return span_has_verb and span_has_subj

def find_split_token(sentence_doc: spacy.tokens.Span) -> Optional[Union[spacy.tokens.Token, Tuple[spacy.tokens.Token, Dict]]]:
    """Finds the first valid token where a sentence can be split based on prioritized rules."""
    conjunctive_adverbs = {'however', 'therefore', 'moreover', 'consequently', 'thus', 'furthermore', 'nevertheless', 'instead', 'otherwise', 'accordingly', 'subsequently', 'hence'}
    dash_chars = {'-', '—', '–'}; sent_len = len(sentence_doc)
    if DEBUG_SPLITTING: logging.debug(f"\nDEBUG Split Check: '{sentence_doc.text[:100]}...'")

    # --- Pass 1: Prioritize Semicolon ---
    if DEBUG_SPLITTING: logging.debug(f"--- Splitting Pass 1: Checking for Semicolon ---")
    for i, token in enumerate(sentence_doc):
        if token.text == ';':
            if DEBUG_SPLITTING: logging.debug(f"  DEBUG: Found ';' at index {i}. Checking S/V around it.")
            cl1_span = sentence_doc[0 : i]; cl2_span = sentence_doc[i + 1 : sent_len]
            if cl1_span and cl2_span and has_subject_and_verb(cl1_span) and has_subject_and_verb(cl2_span):
                if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Priority) - Splitting at ';' (Index {i}) based on S/V check.")
                return token
            else:
                if DEBUG_SPLITTING: logging.debug(f"  DEBUG: Found ';' (Index {i}) but FAILED S/V check around it. Ignoring this semicolon.")
                break # Stop checking semicolons in this priority pass

    # --- Pass 2: Check Other Rules ---
    if DEBUG_SPLITTING: logging.debug(f"--- Splitting Pass 2: Checking Other Rules ---")
    for i, token in enumerate(sentence_doc):
        if PLACEHOLDER_PATTERN.fullmatch(token.text): continue
        if i == 0 or i >= sent_len - 1: continue
        is_cc = (token.dep_ == "cc" and token.pos_ == "CCONJ")
        is_mid_sconj = (token.pos_ == 'SCONJ' or token.dep_ == 'mark') and i > 0
        is_comma = (token.text == ','); is_dash = (token.text in dash_chars); is_colon = (token.text == ':')

        # Rule 2.1: Mid-Sentence SCONJ
        if is_mid_sconj:
             cl1_span = sentence_doc[0 : i]; cl2_span = sentence_doc[i + 1 : sent_len]
             if cl1_span and cl2_span and has_subject_and_verb(cl1_span) and has_subject_and_verb(cl2_span):
                 if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Pass 2) - Splitting at SCONJ '{token.text}' (Index {i}) based on S/V check.")
                 return token

        # Rule 2.2: CCONJ with Context Checks
        elif is_cc:
            should_prevent_split_cc = False
            # Context Check 2a: Adj-Adj-Noun
            if i > 0 and i < sent_len - 2:
                t_before, t_after, t_after_after = sentence_doc[i-1], sentence_doc[i+1], sentence_doc[i+2]
                if t_before.pos_ == 'ADJ' and t_after.pos_ == 'ADJ' and t_after_after.pos_ in ('NOUN', 'PROPN') and \
                   t_before.head.i == t_after_after.i and t_after.head.i == t_after_after.i:
                    should_prevent_split_cc = True; logging.debug(f"    DEBUG CC Prevent: Adj-Adj-Noun") if DEBUG_SPLITTING else None
            # Context Check 2b: Noun/Pronoun-Noun/Pronoun (same head)
            if not should_prevent_split_cc and i > 0 and i < sent_len - 1:
                t_before, t_after = sentence_doc[i-1], sentence_doc[i+1]
                if t_before.pos_ in ('NOUN', 'PROPN', 'PRON') and t_after.pos_ in ('NOUN', 'PROPN', 'PRON') and \
                   t_before.head.i == t_after.head.i:
                    should_prevent_split_cc = True; logging.debug(f"    DEBUG CC Prevent: Noun/Pronoun-Noun/Pronoun") if DEBUG_SPLITTING else None
            # Context Check 2c: Verb/Aux-Verb/Aux (shared subject/head)
            if not should_prevent_split_cc and i > 0 and i < sent_len - 1:
                 t_before, t_after = sentence_doc[i-1], sentence_doc[i+1]
                 if t_before.pos_ in ('VERB', 'AUX') and t_after.pos_ in ('VERB', 'AUX'):
                     subj_before = {c.i for c in t_before.children if c.dep_ in ('nsubj', 'nsubjpass')}
                     subj_after = {c.i for c in t_after.children if c.dep_ in ('nsubj', 'nsubjpass')}
                     same_head = t_before.head.i == t_after.i or t_after.head.i == t_before.i
                     if (subj_before and subj_before == subj_after) or same_head:
                         should_prevent_split_cc = True; logging.debug(f"    DEBUG CC Prevent: Verb/Aux-Verb/Aux") if DEBUG_SPLITTING else None
            # Fallback S/V Check for CC
            if not should_prevent_split_cc:
                cl1_span = sentence_doc[0 : i]; cl2_span = sentence_doc[i + 1 : sent_len]
                if cl1_span and cl2_span and has_subject_and_verb(cl1_span) and has_subject_and_verb(cl2_span):
                    if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Pass 2) - Splitting at CC '{token.text}' (Index {i}) based on S/V check.")
                    return token
                elif DEBUG_SPLITTING: logging.debug(f"    DEBUG CC Fallback S/V check FAILED for '{token.text}' (Index {i}).")
            elif DEBUG_SPLITTING: logging.debug(f"  DEBUG: Skipping CCONJ '{token.text}' (Index {i}) due to context rule.")

        # Rule 2.3: Comma + Conjunctive Adverb
        elif is_comma and i < sent_len - 1:
            t_after_idx = i + 1
            while t_after_idx < sent_len and PLACEHOLDER_PATTERN.fullmatch(sentence_doc[t_after_idx].text): t_after_idx += 1
            if t_after_idx >= sent_len : continue
            t_after = sentence_doc[t_after_idx]
            if t_after.lower_ in conjunctive_adverbs and t_after.pos_ == 'ADV':
                cl1_span = sentence_doc[0 : i]; cl2_span_start_idx = t_after_idx + 1
                cl2_span = sentence_doc[cl2_span_start_idx : sent_len] if cl2_span_start_idx < sent_len else None
                if cl1_span and cl2_span and has_subject_and_verb(cl1_span) and has_subject_and_verb(cl2_span):
                    if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Pass 2) - Splitting at ',' (Index {i}) based on Conj. Adverb '{t_after.text}'.")
                    return token

        # Rule 2.4: Comma + Relative 'which'
        elif is_comma and i < sent_len - 1:
            t_after_idx = i + 1
            while t_after_idx < sent_len and PLACEHOLDER_PATTERN.fullmatch(sentence_doc[t_after_idx].text): t_after_idx += 1
            if t_after_idx >= sent_len : continue
            t_after = sentence_doc[t_after_idx]
            if t_after.lower_ == 'which' and t_after.tag_ == 'WDT' and t_after.dep_ in ('nsubj', 'nsubjpass', 'dobj'):
                cl1_span = sentence_doc[0 : i]; cl2_check_start_idx = t_after_idx + 1
                cl2_check_span = sentence_doc[cl2_check_start_idx : sent_len] if cl2_check_start_idx < sent_len else None
                if cl1_span and cl2_check_span and has_subject_and_verb(cl1_span) and \
                   any(t.pos_ in ("VERB", "AUX") for t in cl2_check_span if not PLACEHOLDER_PATTERN.fullmatch(t.text)):
                    antecedent = t_after.head; pronoun = "It"
                    try:
                        morph_num = antecedent.morph.get('Number')
                        if morph_num and 'Plur' in morph_num: pronoun = "They"
                        elif antecedent.pos_ == 'NOUN' and antecedent.text.lower().endswith('s') and not morph_num:
                             singular_form = antecedent.lemma_
                             if singular_form != antecedent.text.lower(): pronoun = "They"
                    except Exception: pass # Ignore morph errors
                    if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Pass 2) - Splitting at ',' (Index {i}) based on Relative Clause 'which'. Antecedent='{antecedent.text}', Pronoun='{pronoun}'.")
                    return (token, {'type': 'relative_which', 'pronoun': pronoun, 'replace_token': t_after})

        # Rule 2.5: Potential Comma Splice
        elif is_comma:
            is_handled_by_prior_rule = False # Check if handled by 2.3 or 2.4
            if i < sent_len - 1:
                t_after_idx = i + 1
                while t_after_idx < sent_len and PLACEHOLDER_PATTERN.fullmatch(sentence_doc[t_after_idx].text): t_after_idx += 1
                if t_after_idx < sent_len:
                    t_after = sentence_doc[t_after_idx]
                    if t_after.lower_ in conjunctive_adverbs and t_after.pos_ == 'ADV': is_handled_by_prior_rule = True
                    if t_after.lower_ == 'which' and t_after.tag_ == 'WDT' and t_after.dep_ in ('nsubj','nsubjpass', 'dobj'): is_handled_by_prior_rule = True
            if not is_handled_by_prior_rule:
                cl1_span = sentence_doc[0 : i]; cl2_span = sentence_doc[i + 1 : sent_len]
                if cl1_span and cl2_span and has_subject_and_verb(cl1_span) and has_subject_and_verb(cl2_span):
                    if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Pass 2) - Splitting at ',' (Index {i}). Reason: Potential Comma Splice (S/V + S/V).")
                    return token

        # Rule 2.6: Dashes
        elif is_dash:
            is_compound_word_hyphen = False
            if i > 0 and i < sent_len - 1:
                prev_token, next_token = sentence_doc[i-1], sentence_doc[i+1]
                if prev_token.idx + len(prev_token.text) == token.idx and \
                   token.idx + len(token.text) == next_token.idx and \
                   prev_token.is_alpha and next_token.is_alpha:
                    is_compound_word_hyphen = True
            if not is_compound_word_hyphen:
                cl1_span = sentence_doc[0 : i]; cl2_span = sentence_doc[i + 1 : sent_len]
                if cl1_span and cl2_span and has_subject_and_verb(cl1_span) and has_subject_and_verb(cl2_span):
                    if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Pass 2) - Splitting at dash '{token.text}' (Index {i}) based on S/V check.")
                    return token

        # Rule 2.7: Colons
        elif is_colon :
            cl1_span = sentence_doc[0:i]; cl2_span = sentence_doc[i+1:sent_len]
            if cl2_span and has_subject_and_verb(cl2_span):
                if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Pass 2) - Splitting at ':' (Index {i}) based on S/V check in second clause.")
                return token

    # --- Rule 2.8: Initial SCONJ + Comma ---
    first_real_token_idx = 0
    while first_real_token_idx < sent_len and PLACEHOLDER_PATTERN.fullmatch(sentence_doc[first_real_token_idx].text): first_real_token_idx += 1
    if first_real_token_idx < sent_len and (sentence_doc[first_real_token_idx].pos_ == 'SCONJ' or sentence_doc[first_real_token_idx].dep_ == 'mark'):
        initial_sconj_token = sentence_doc[first_real_token_idx]
        if DEBUG_SPLITTING: logging.debug(f"--- Splitting Pass 3: Checking for Initial SCONJ '{initial_sconj_token.text}' + Comma ---")
        for i, token in enumerate(sentence_doc):
            if token.i <= first_real_token_idx: continue
            if PLACEHOLDER_PATTERN.fullmatch(token.text): continue
            if token.text == ',':
                cl1_check_span = sentence_doc[first_real_token_idx + 1 : i]
                cl2_span = sentence_doc[i + 1 : sent_len]
                if cl1_check_span and cl2_span and has_subject_and_verb(cl1_check_span) and has_subject_and_verb(cl2_span):
                    is_which_case = False # Avoid splitting if handled by Rule 2.4
                    next_real_token_idx = i + 1
                    while next_real_token_idx < sent_len and PLACEHOLDER_PATTERN.fullmatch(sentence_doc[next_real_token_idx].text): next_real_token_idx += 1
                    if next_real_token_idx < sent_len:
                        next_real_token = sentence_doc[next_real_token_idx]
                        if next_real_token.lower_=='which' and next_real_token.tag_=='WDT': is_which_case=True
                    if not is_which_case:
                        if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Pass 3) - Splitting at ',' (Index {i}) following initial SCONJ '{initial_sconj_token.text}'.")
                        return token
                break # Only check first comma after initial SCONJ

    # --- No split point found ---
    if DEBUG_SPLITTING and sent_len > 10:
        logging.debug(f"DEBUG: No valid split point found for sentence: '{sentence_doc.text[:100]}...'")
    return None

# *** Start of Modified apply_sentence_splitting Function ***
def apply_sentence_splitting(text_with_placeholders: str, nlp_model: spacy.Language) -> str:
    """Applies sentence splitting logic using find_split_token, with minimum length check."""
    if not text_with_placeholders or not text_with_placeholders.strip(): return ""

    # *** Define Minimum Word Count for the Second Clause of a Split ***
    MIN_SPLIT_WORDS = 6 # Adjust this value as needed (e.g., 4 or 5)

    logging.info("Applying sentence splitting rules...")
    output_lines = []; num_splits = 0; num_splits_skipped = 0 # Track skipped splits
    try:
        doc = nlp_model(text_with_placeholders)
        for sent in doc.sents:
            original_sentence_text = sent.text
            if not original_sentence_text.strip(): continue
            original_ends_with_question = original_sentence_text.strip().endswith('?')
            original_ends_with_exclamation = original_sentence_text.strip().endswith('!')
            default_ending = "?" if original_ends_with_question else "!" if original_ends_with_exclamation else "."

            split_token_result = find_split_token(sent)
            split_details, split_token = None, None
            if isinstance(split_token_result, tuple): split_token, split_details = split_token_result
            elif split_token_result is not None: split_token = split_token_result

            if split_token is not None:
                # --- Potential Split Found - Extract Clauses First ---
                split_idx_in_sent = split_token.i - sent.start
                clause1_start_char_idx = sent.start_char; clause1_end_char_idx = split_token.idx
                # Adjust start for initial SCONJ+comma split
                first_real_token_idx_sent = 0
                while first_real_token_idx_sent < len(sent) and PLACEHOLDER_PATTERN.fullmatch(sent[first_real_token_idx_sent].text): first_real_token_idx_sent += 1
                if first_real_token_idx_sent < len(sent) and \
                   (sent[first_real_token_idx_sent].pos_ == 'SCONJ' or sent[first_real_token_idx_sent].dep_ == 'mark') and \
                   split_token.text == ',':
                    # Ensure there is a token after the SCONJ before accessing idx
                    if first_real_token_idx_sent + 1 < len(sent):
                         clause1_start_char_idx = sent[first_real_token_idx_sent + 1].idx
                    else: # Handle case where SCONJ is the last token before comma (unlikely but safe)
                         clause1_start_char_idx = split_token.idx # Effectively makes clause1 empty before cleaning

                # Adjust end if splitting at non-punct preceded by comma
                if split_token.pos_ != 'PUNCT' and split_idx_in_sent > 0:
                    preceding_token = sent[split_idx_in_sent - 1]
                    if preceding_token.text == ',': clause1_end_char_idx = preceding_token.idx
                clause1_raw_text = doc.text[clause1_start_char_idx:clause1_end_char_idx]

                clause2_start_char_idx = split_token.idx + len(split_token.text); clause2_end_char_idx = sent.end_char; clause2_prefix = ""
                # Handle 'relative_which' reconstruction
                if split_details and split_details.get('type') == 'relative_which':
                    replace_token = split_details['replace_token']
                    clause2_start_char_idx = replace_token.idx + len(replace_token.text)
                    clause2_prefix = split_details['pronoun']
                # Handle comma + conjunctive adverb reconstruction
                elif split_token.text == ',' and (split_token.i + 1) < sent.end:
                    next_token_idx = split_token.i + 1
                    while next_token_idx < sent.end and PLACEHOLDER_PATTERN.fullmatch(sent.doc[next_token_idx].text): next_token_idx+=1
                    if next_token_idx < sent.end:
                        next_token = sent.doc[next_token_idx]
                        conjunctive_adverbs_local = {'however', 'therefore', 'moreover', 'consequently', 'thus', 'furthermore', 'nevertheless', 'instead', 'otherwise', 'accordingly', 'subsequently', 'hence'}
                        if next_token.lower_ in conjunctive_adverbs_local and next_token.pos_ == 'ADV':
                            clause2_start_char_idx = next_token.idx + len(next_token.text)
                clause2_base = doc.text[clause2_start_char_idx:clause2_end_char_idx]

                clause1_cleaned = clause1_raw_text.strip().strip('.,;:')
                clause2_base_cleaned = clause2_base.strip().strip('.,;:')
                clause2_cleaned = f"{clause2_prefix} {clause2_base_cleaned}".strip() if clause2_prefix else clause2_base_cleaned

                # *** NEW: Check Length of Second Clause Before Committing to Split ***
                clause2_word_count = len(clause2_cleaned.split())
                if clause2_word_count < MIN_SPLIT_WORDS:
                    # If second clause is too short, skip the split for this sentence
                    if DEBUG_SPLITTING:
                        logging.debug(f"  DEBUG: Skipping split at '{split_token.text}' (Index {split_token.i}). Reason: Second clause '{clause2_cleaned[:30]}...' too short ({clause2_word_count} words < {MIN_SPLIT_WORDS}).")
                    output_lines.append(original_sentence_text) # Use original sentence
                    num_splits_skipped += 1
                else:
                    # If second clause is long enough, proceed with the split
                    num_splits += 1
                    if DEBUG_SPLITTING:
                         logging.debug(f"  DEBUG: Applying split at '{split_token.text}' (Index {split_token.i}). Second clause length ok ({clause2_word_count} words).")

                    def capitalize_first_letter(s: str) -> str:
                        match = re.search(r'([a-zA-Z])', s);
                        if match: start_index = match.start(); return s[:start_index] + s[start_index].upper() + s[start_index+1:]
                        return s # Return as-is if no letter found

                    clause1_capitalized = capitalize_first_letter(clause1_cleaned)
                    clause2_capitalized = capitalize_first_letter(clause2_cleaned)

                    if clause1_capitalized: output_lines.append(clause1_capitalized.rstrip('.?!') + ".")
                    if clause2_capitalized: output_lines.append(clause2_capitalized.rstrip('.?!') + default_ending)
                # *** END OF NEW LENGTH CHECK BLOCK ***

            else: # No split point found
                output_lines.append(original_sentence_text)

        logging.info(f"Sentence splitting applied. Found {num_splits} split points. Skipped {num_splits_skipped} potential splits due to short second clause.")
        final_text = " ".join(output_lines).strip()
        return re.sub(r'\s{2,}', ' ', final_text)
    except Exception as e:
        logging.error(f"Error during sentence splitting application: {e}", exc_info=True)
        return text_with_placeholders # Fallback
# *** End of Modified apply_sentence_splitting Function ***

# ==============================================================================
# 7. HELPER FUNCTIONS: OPENAI REFINEMENT
# ==============================================================================
def refine_with_openai(text_with_placeholders: str, discipline: str) -> str:
    """Uses OpenAI API (if available) for subtle refinements."""
    if not openai_client:
        logging.warning("OpenAI client not available. Skipping refinement step.")
        return text_with_placeholders
    logging.info("Applying OpenAI refinement (around placeholders)...")
    start_time = time.time()
    system_prompt = "You are a punctuation and spelling repair-bot. Your output will be displayed in a structured web app interface. Do not include any explanations or commentary—only return the modified text."
    user_prompt_parts = [
        "Repair the following text based on these rules:", "--- TEXT START ---", text_with_placeholders, "--- TEXT END ---", "--- RULES ---",
        "- Ignored freeze terms (e.g., __F0__) and leave them untouched.",
        "- For approximately 50% of the passive voice sentences in the text, change them to the active voice.",
        "- In approximately 50% of sentences, replace common modal verbs (e.g., can, could, may, might, should, would) with more natural or conversational alternatives.",
        "- In approximately 50% of sentences, replace infinitive verb forms (e.g., 'to analyze') with gerunds (e.g., 'analyzing').",
        "- Strictly remove the Oxford comma in lists of three or more items.",
        "- The input text contains sentences separated by periods (.), question marks (?), etc. DO NOT merge these sentences. Apply all other rules ONLY *within* the boundaries of each existing sentence. Treat them as separate entities which cannot be merged. Never replace a period with a comma.",
        "- Review the beginning of each sentence. If a sentence starts *exactly* with 'And ', 'But ', 'Or ', or 'So ', remove that initial word and the following space.",
        "- Correct only critical spelling or punctuation errors within the current sentence."
    ]
    user_prompt = "\n".join(user_prompt_parts)
    try:
        response = openai_client.chat.completions.create(
            model=CONFIG["openai_model"], messages=[ {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt} ],
            temperature=CONFIG["openai_temperature"], max_tokens=CONFIG["openai_max_tokens"]
        )
        end_time = time.time()
        refined_text = response.choices[0].message.content.strip()
        logging.info(f"OpenAI refinement completed in {end_time - start_time:.2f} seconds.")
        if not refined_text or "sorry" in refined_text.lower() or "cannot fulfill" in refined_text.lower():
            logging.warning("OpenAI response invalid or refusal. Returning text before refinement.")
            return text_with_placeholders
        if refined_text == text_with_placeholders: logging.info("OpenAI returned identical text.")
        else: logging.info("OpenAI returned modified text.")
        return refined_text
    except Exception as e:
        logging.error(f"Error during OpenAI API call: {e}", exc_info=True)
        return text_with_placeholders # Return original on error

# ==============================================================================
# 8. MODEL LOADING FUNCTION (Optimized)
# ==============================================================================
def load_models(device_preference: str = "auto") -> Optional[Dict[str, Any]]:
    """Loads and initializes NLP models (spaCy, Paraphraser, AI Detector), disabling unused spaCy components."""
    logging.info("--- Loading Models ---")
    models = {}
    # Determine device
    if device_preference == "auto": models["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_preference == "cuda" and torch.cuda.is_available(): models["device"] = "cuda"
    else: models["device"] = "cpu"; logging.warning("CUDA requested but not available. Using CPU.") if device_preference == "cuda" else None
    logging.info(f"Using device: {models['device']}")

    try:
        # Load spaCy model (disable 'ner' as it's likely unused)
        spacy_model_name = CONFIG['spacy_model']
        logging.info(f"Loading spaCy model: {spacy_model_name} (disabling NER)...")
        try:
            # *** Optimization: Disable unused components ***
            models["nlp"] = spacy.load(spacy_model_name, disable=['ner'])
        except OSError:
            logging.warning(f"spaCy model '{spacy_model_name}' not found. Attempting download...")
            try:
                spacy.cli.download(spacy_model_name)
                models["nlp"] = spacy.load(spacy_model_name, disable=['ner'])
                logging.info(f"spaCy model '{spacy_model_name}' downloaded and loaded.")
            except Exception as download_err:
                logging.error(f"Failed to download/load spaCy model '{spacy_model_name}': {download_err}", exc_info=True)
                return None
        logging.info("spaCy model loaded.")

        # Load Paraphraser
        paraphraser_model_name = CONFIG['paraphraser_model']
        logging.info(f"Loading paraphraser: {paraphraser_model_name}...")
        models["tokenizer"] = AutoTokenizer.from_pretrained(paraphraser_model_name)
        models["model"] = AutoModelForSeq2SeqLM.from_pretrained(paraphraser_model_name).to(models["device"])
        logging.info("Paraphraser model loaded.")

        # Load AI Detector
        ai_detector_model_name = CONFIG['ai_detector_model']
        logging.info(f"Loading AI detector: {ai_detector_model_name}...")
        previous_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error() # Suppress warnings
        models["ai_detector"] = pipeline("text-classification", model=ai_detector_model_name, device=models["device"])
        transformers.logging.set_verbosity(previous_verbosity)
        logging.info("AI detector pipeline loaded.")

        logging.info("--- All models loaded successfully ---")
        return models
    except Exception as e:
        logging.exception("Fatal error during model loading:")
        return None

# ==============================================================================
# 9. CORE PROCESSING FUNCTION
# ==============================================================================
def process_text(
    input_text: str, freeze_terms: List[str], discipline: str,
    models: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Orchestrates the text processing pipeline and returns results including metrics."""
    # --- Input Validation ---
    if not input_text or not input_text.strip():
        logging.error("Input text is empty.")
        return {'status': 'error', 'message': 'Input text cannot be empty.'}
    word_count = len(input_text.split())
    if word_count > config["input_max_words"]:
        logging.error(f"Input text exceeds word limit ({word_count}/{config['input_max_words']}).")
        return {'status': 'error', 'message': f"Error: Text exceeds word limit ({config['input_max_words']} words). Provided: {word_count}."}

    logging.info(f"--- Starting Text Processing for request ---")
    logging.info(f"Discipline: {discipline}, Input Words: {word_count}, Freeze Terms: {len(freeze_terms)}")
    start_process_time = time.time()

    try:
        # --- Placeholder Replacement ---
        placeholder_map: Dict[str, str] = {}
        text_with_placeholders = input_text
        placeholder_counter = 0
        freeze_terms_sorted = sorted([ft for ft in freeze_terms if ft and ft.strip()], key=len, reverse=True)
        if freeze_terms_sorted:
            logging.info(f"Replacing {len(freeze_terms_sorted)} freeze term(s) with placeholders...")
            processed_spans_map: Dict[int, Tuple[int, str]] = {}
            temp_text_for_search = text_with_placeholders
            for term in freeze_terms_sorted:
                # *** SyntaxError Fix Applied Below ***
                term_stripped = term.strip()
                if not term_stripped:
                    continue # Skip empty terms after stripping
                # *** End SyntaxError Fix ***
                try:
                    pattern = re.compile(re.escape(term_stripped), re.IGNORECASE); matches_this_term = []
                    for match in pattern.finditer(temp_text_for_search):
                        start, end = match.span()
                        is_overlapping = any(max(start, ps) < min(end, pe) for ps, (pe, _) in processed_spans_map.items())
                        if not is_overlapping: matches_this_term.append(match)
                    if matches_this_term:
                        original_term_matched = matches_this_term[0].group(0)
                        placeholder = f"__F{placeholder_counter}__"
                        placeholder_map[placeholder] = original_term_matched; placeholder_counter += 1
                        for match in matches_this_term: processed_spans_map[match.start()] = (match.end(), placeholder)
                except re.error as regex_err: logging.warning(f"Regex error for freeze term '{term_stripped}': {regex_err}")
                except Exception as term_err: logging.warning(f"Error processing freeze term '{term_stripped}': {term_err}")
            if processed_spans_map:
                sorted_starts = sorted(processed_spans_map.keys(), reverse=True); temp_text_list = list(text_with_placeholders)
                for start in sorted_starts: end, placeholder = processed_spans_map[start]; temp_text_list[start:end] = list(placeholder)
                text_with_placeholders = "".join(temp_text_list)
                logging.info(f"{placeholder_counter} placeholder types created for {len(processed_spans_map)} replacements.")
        else: logging.info("No freeze terms provided.")

        # --- Paraphrasing Sentences ---
        logging.info("Tokenizing text (with placeholders) into sentences...")
        try:
            doc = models["nlp"](text_with_placeholders)
            sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
            if not sentences: return {'status': 'error', 'message': 'No valid sentences found after placeholder step.'}
        except Exception as nlp_err: return {'status': 'error', 'message': f'Sentence tokenization failed: {nlp_err}'}
        try: # Check original sentence length
            original_doc = models["nlp"](input_text)
            original_sentences_map = {i: s.text.strip() for i, s in enumerate(original_doc.sents) if s.text.strip()}
        except Exception as nlp_err_orig: logging.warning(f"Could not tokenize original text: {nlp_err_orig}"); original_sentences_map = {}

        paraphrased_sentences = []; sentences_reverted = 0
        logging.info(f"Paraphrasing {len(sentences)} sentences...")
        for i, sentence_with_placeholder in enumerate(sentences):
            # Check original length if possible
            original_sentence_for_check = original_sentences_map.get(i, None)
            if original_sentence_for_check:
                 try:
                      sentence_tokens = models["tokenizer"].tokenize(original_sentence_for_check)
                      if len(sentence_tokens) > config["input_max_sentence_tokens"]:
                           msg = f"Error: Original Sentence #{i+1} exceeds token limit ({len(sentence_tokens)}/{config['input_max_sentence_tokens']})."
                           logging.error(msg)
                           return { 'status': 'error', 'message': msg }
                 except Exception as token_err: logging.warning(f"Tokenization failed for length check on sentence {i+1}: {token_err}")

            paraphrased = paraphrase_sentence(sentence_with_placeholder, models["model"], models["tokenizer"], config["paraphrase_num_beams"], config["paraphrase_max_length"])
            if "[Paraphrasing Error:" in paraphrased or "[Encoding Error]" in paraphrased:
                logging.error(f"Paraphrasing failed for sentence {i+1}: {paraphrased}")
                return {'status': 'error', 'message': f"Error paraphrasing sentence #{i+1}: {paraphrased}"}

            # Verify placeholder preservation
            placeholders_in_original = set(PLACEHOLDER_PATTERN.findall(sentence_with_placeholder))
            if placeholders_in_original:
                placeholders_in_paraphrased = set(PLACEHOLDER_PATTERN.findall(paraphrased))
                if not placeholders_in_original.issubset(placeholders_in_paraphrased):
                    missing = placeholders_in_original - placeholders_in_paraphrased
                    logging.warning(f"Placeholder(s) {missing} dropped in sentence {i+1}. Reverting.")
                    paraphrased_sentences.append(sentence_with_placeholder); sentences_reverted += 1
                else: paraphrased_sentences.append(paraphrased)
            else: paraphrased_sentences.append(paraphrased)
        logging.info(f"Paraphrasing complete. {sentences_reverted} sentence(s) reverted.")
        t5_paraphrased_text_with_placeholders = " ".join(paraphrased_sentences)

        # --- Apply Sentence Splitting ---
        text_after_splitting_with_placeholders = apply_sentence_splitting(t5_paraphrased_text_with_placeholders, models["nlp"])

        # --- Optional Filter Trailing Question ---
        unwanted_phrase = "Can you provide some examples?"
        temp_text_lower = text_after_splitting_with_placeholders.rstrip('.?! ').lower()
        phrase_lower = unwanted_phrase.rstrip('.?! ').lower()
        text_to_refine_with_placeholders = text_after_splitting_with_placeholders
        if temp_text_lower.endswith(phrase_lower):
            logging.info(f"Filtering detected unwanted trailing phrase: '{unwanted_phrase}'")
            try:
                phrase_start_index = text_after_splitting_with_placeholders.lower().rindex(phrase_lower)
                text_to_refine_with_placeholders = text_after_splitting_with_placeholders[:phrase_start_index].rstrip()
            except Exception as filter_err: logging.warning(f"Error during filtering: {filter_err}")

        # --- Refine with OpenAI ---
        refined_text_with_placeholders = refine_with_openai(text_to_refine_with_placeholders, discipline)

        # --- Re-insert Original Freeze Terms ---
        logging.info("Re-inserting original freeze terms...")
        final_text = refined_text_with_placeholders

        # *** FIX 1: Clean up potential extra spaces around placeholders BEFORE replacing ***
        if placeholder_map: # Only clean if placeholders were used
            try:
                # Remove space(s) immediately BEFORE __F...__
                final_text = re.sub(r'\s+(__F\d+__)', r'\1', final_text)
                # Remove space(s) immediately AFTER __F...__
                final_text = re.sub(r'(__F\d+__)\s+', r'\1', final_text)
                logging.info("Cleaned potential whitespace around placeholders.")
            except Exception as clean_err:
                logging.warning(f"Could not clean whitespace around placeholders: {clean_err}")
        # *** END FIX 1 ***

        if placeholder_map:
            try: sorted_placeholders = sorted(placeholder_map.keys(), key=lambda p: int(PLACEHOLDER_PATTERN.fullmatch(p).group(0).split('__F')[1].split('__')[0]))
            except Exception as sort_err: logging.error(f"Error sorting placeholders: {sort_err}"); sorted_placeholders = sorted(placeholder_map.keys())
            num_reinserted = 0
            for placeholder in sorted_placeholders:
                original_term = placeholder_map[placeholder]; occurrences = final_text.count(placeholder)
                if occurrences > 0:
                    # *** FIX 2: Add spaces around original_term during replacement ***
                    final_text = final_text.replace(placeholder, f" {original_term} ")
                    num_reinserted += occurrences
            logging.info(f"{num_reinserted} placeholder instance(s) re-inserted.")

            # *** FIX 3: Clean up potential double spaces AFTER the loop ***
            final_text = re.sub(r'\s{2,}', ' ', final_text).strip()
            logging.info("Cleaned up potential double spaces after re-insertion.")
            # *** END FIX 3 ***

        else: logging.info("No placeholders used, skipping re-insertion.")


        # --- Final Metrics Calculation ---
        logging.info("Calculating final metrics...")
        paraphrased_ai_score, paraphrased_fk_score, paraphrased_burstiness, paraphrased_ttr, paraphrased_word_count, paraphrased_mod_percentage = 0.0, 0.0, 0.0, 0.0, 0, 0.0
        paraphrased_ai_result = {'label': 'N/A', 'score': 0.0}
        if final_text and final_text.strip():
            try: paraphrased_ai_result = models["ai_detector"](final_text)[0]; paraphrased_ai_score = paraphrased_ai_result.get('score', 0.0)
            except Exception as ai_err: logging.error(f"Error calculating final AI score: {ai_err}"); paraphrased_ai_result = {'label': 'Error', 'score': 0.0}
            try: paraphrased_fk_score = flesch_kincaid_grade(final_text)
            except Exception as fk_err: logging.error(f"Error calculating final FK score: {fk_err}")
            paraphrased_burstiness = calculate_burstiness(final_text)
            paraphrased_ttr = calculate_ttr(final_text)
            paraphrased_word_count = len(final_text.split())
            paraphrased_mod_percentage = calculate_modification_percentage(input_text, final_text)
            logging.info("Final metrics calculated.")
        else: logging.warning("Final text is empty. Setting final metrics to default values.")

        # --- Final Freeze Term Preservation Check ---
        # *** Using Improved Regex Check ***
        final_freeze_terms_preserved = True
        if placeholder_map:
            logging.info(f"Verifying placeholder re-insertion in final output...")
            if PLACEHOLDER_PATTERN.search(final_text):
                remaining_found = PLACEHOLDER_PATTERN.findall(final_text)
                logging.warning(f"Placeholder(s) still found in final output after re-insertion: {set(remaining_found)}")
                final_freeze_terms_preserved = False
            else:
                logging.info("All placeholders appear to have been re-inserted successfully.")
        else:
            logging.info("No freeze terms were used, skipping final verification.")
            final_freeze_terms_preserved = True


        # --- Structure Output Data ---
        output_data = {
            "status": "success",
            # Only include final results for the API response
            "Final Output Text": final_text,
            "final_metrics": {
                "ai_score_label": paraphrased_ai_result.get('label', 'N/A'),
                "ai_score_value": round(paraphrased_ai_score, 4),
                "flesch_kincaid_grade": round(paraphrased_fk_score, 1),
                "burstiness_score": round(paraphrased_burstiness, 4),
                "type_token_ratio": round(paraphrased_ttr, 4),
                "modification_percentage": round(paraphrased_mod_percentage, 2),
                "word_count": paraphrased_word_count,
                "freeze_terms_preserved": final_freeze_terms_preserved # Use the result of the improved check
            }
            # Optional: Add back intermediate steps here if needed for debugging
            # "T5_Output_Placeholders": t5_paraphrased_text_with_placeholders,
            # "Split_Output_Placeholders": text_after_splitting_with_placeholders,
            # "GPT_Input_Placeholders": text_to_refine_with_placeholders,
            # "GPT_Output_Placeholders": refined_text_with_placeholders,
        }
        end_process_time = time.time()
        logging.info(f"--- Processing Finished (Request Time: {end_process_time - start_process_time:.2f} seconds) ---")
        return output_data

    except Exception as e:
        logging.exception("An unexpected error occurred during core text processing:")
        return {'status': 'error', 'message': f"An unexpected processing error occurred: {str(e)}"}

# ==============================================================================
# 10. FLASK APP INITIALIZATION
# ==============================================================================
app = Flask(__name__)
CORS(app) # Enable CORS for all routes
# ==============================================================================
# 11. GLOBAL MODEL LOADING (Executed once when Flask starts)
# ==============================================================================
# Load models globally when the Flask app starts
logging.info("Initializing Flask app and loading models...")
# Use the function defined above
loaded_models_global = load_models(device_preference=CONFIG.get("default_device", "auto"))
if not loaded_models_global:
    logging.critical("FATAL: Models failed to load. Flask app cannot serve requests properly.")
    # In a production scenario, you might want the app to exit or signal an error state.
    # For now, it will just log the error, and the endpoint will return an error message.
else:
    logging.info("Global models loaded successfully. Flask app ready.")

# ==============================================================================
# 12. FLASK API ENDPOINT
# ==============================================================================
@app.route('/humanize', methods=['POST'])
def humanize_endpoint():
    """API endpoint to process text."""
    start_request_time = time.time()
    logging.info(f"Received request for /humanize from {request.remote_addr}")

    # Check if models loaded correctly during startup
    if not loaded_models_global:
        logging.error("Models not loaded, cannot process request.")
        return jsonify({"status": "error", "message": "Backend models are not loaded or failed to load."}), 503 # Service Unavailable

    try:
        # --- Get and Validate Input Data ---
        data = request.get_json()
        if not data:
            logging.warning("Received empty request data.")
            return jsonify({"status": "error", "message": "No input data received"}), 400

        input_text = data.get('text')
        discipline = data.get('discipline', DEFAULT_DISCIPLINE)
        freeze_terms = data.get('freeze_terms', [])

        if not input_text:
            logging.warning("Request missing 'text' field.")
            return jsonify({"status": "error", "message": "Input 'text' is required"}), 400
        if not isinstance(freeze_terms, list):
             logging.warning("Received 'freeze_terms' is not a list.")
             return jsonify({"status": "error", "message": "'freeze_terms' must be a list"}), 400
        if discipline not in VALID_DISCIPLINES:
             logging.warning(f"Received invalid discipline '{discipline}', using default '{DEFAULT_DISCIPLINE}'.")
             discipline = DEFAULT_DISCIPLINE

        # --- Call Core Processing Logic ---
        # Pass the globally loaded models
        result_dict = process_text(
            input_text=input_text,
            freeze_terms=freeze_terms,
            discipline=discipline,
            models=loaded_models_global,
            config=CONFIG
        )

        # --- Prepare and Return Response ---
        if result_dict.get("status") == "success":
            # *** Optimization: Return only final text and metrics ***
            response_data = {
                "status": "success",
                "final_text": result_dict.get("Final Output Text", ""), # Default to empty string if key missing
                "final_metrics": result_dict.get("final_metrics", {})   # Default to empty dict if key missing
            }
            status_code = 200
        else:
            # Pass through the error message from process_text
            response_data = {
                "status": "error",
                "message": result_dict.get('message', 'Unknown processing error')
            }
            # Use 500 for internal server errors, maybe 422 (Unprocessable Entity) for validation errors caught in process_text
            status_code = 500 if "unexpected" in result_dict.get('message', '').lower() else 422

        end_request_time = time.time()
        logging.info(f"Request processed in {end_request_time - start_request_time:.2f} seconds. Status: {status_code}")
        return jsonify(response_data), status_code

    except Exception as e:
        # Catch-all for errors within the endpoint logic itself
        logging.error("An error occurred processing the /humanize request:", exc_info=True)
        return jsonify({"status": "error", "message": f"An internal server error occurred processing the request."}), 500

# ==============================================================================
# 13. MAIN EXECUTION BLOCK (For Running the Server)
# ==============================================================================
if __name__ == '__main__':
    # Instructions for running in production:
    print("\n--- TextPulse Flask Server ---")
    print("To run in production (on DigitalOcean, etc.), use Gunicorn:")
    print("Example: gunicorn --workers 2 --bind 0.0.0.0:5000 app:app")
    print("  (Replace 'app:app' with 'your_filename:app')")
    print("  Adjust '--workers' based on your server's CPU cores and RAM.")
    print("---------------------------------")

    # Run the Flask development server (for local testing ONLY)
    # NOTE: debug=True automatically reloads on code changes but can consume more memory.
    #       Set debug=False for more realistic performance/memory testing locally.
    print("\nStarting Flask development server (for testing only)...")
    app.run(debug=False, host='0.0.0.0', port=5001) # Use a specific port

# ==============================================================================
# 14. FINAL SCRIPT DOCSTRING (Summary - Optional but good practice)
# ==============================================================================
"""
(See detailed feature summary in the main docstring at the top of the file)
"""
