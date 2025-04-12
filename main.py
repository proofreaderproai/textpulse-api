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
from flask_caching import Cache # Note: Cache is imported but not used in the provided snippet. Remove if unused.

# Flask related imports
from flask import Flask, request, jsonify
from flask_cors import CORS
# *** Assuming python-dotenv is needed if you use .env file ***
# *** If you set env vars via systemd or other means, you can remove dotenv ***
# from dotenv import load_dotenv # To load .env file

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
    print("Try: pip install Flask flask-cors spacy torch transformers openai textstat python-dotenv") # Added flask-cors and python-dotenv here
    # Exit if core dependencies are missing
    sys.exit(1)

# ==============================================================================
# Load .env file (immediately after imports) - Uncomment if needed
# ==============================================================================
# load_dotenv() # Loads variables from .env into environment

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

if not openai_api_key or not openai_api_key.startswith("sk-"):
    logging.warning("OPENAI_API_KEY environment variable not found, invalid, or key format incorrect.")
    logging.warning("OpenAI refinement step will be skipped.")
else:
    try:
        openai_client = openai.OpenAI(api_key=openai_api_key)
        # Optional: Test connection (add timeout)
        # You might want to remove this test call in production to avoid unnecessary API calls on startup.
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
    # Avoid division by zero if std_dev is 0 (text is uniform)
    return (std_dev / mean) if mean > 0 else 0.0

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

# >>> START OF PART 2 <<<

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

        # Check if input_ids exceed max_length *after* encoding (shouldn't happen with truncation=True, but safety check)
        if input_ids.shape[1] > max_length:
             logging.warning(f"Input sentence encoding length ({input_ids.shape[1]}) exceeded max_length ({max_length}) even after truncation. Output might be poor. Sentence: '{sentence_with_placeholders[:50]}...'")
             # Optionally truncate again, though tokenizer should handle this
             input_ids = input_ids[:, :max_length]


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
# 6. HELPER FUNCTIONS: SENTENCE SPLITTING (Part 1 - Dependency Checks)
# ==============================================================================
def has_subject_and_verb(doc_span: spacy.tokens.Span) -> bool:
    """Checks if a spaCy span likely contains an explicit subject and a main verb."""
    if not doc_span or len(doc_span) == 0: return False
    span_has_verb, span_has_subj = False, False; verb_indices_in_span = set()
    for token in doc_span:
        if token.pos_ in ("VERB", "AUX"): span_has_verb = True; verb_indices_in_span.add(token.i)
    if not span_has_verb: return False
    # Check for nominal subjects attached to verbs within the span
    for token in doc_span:
        if token.dep_ in ("nsubj", "nsubjpass") and token.head.i in verb_indices_in_span: span_has_subj = True; break
        # Check if a root verb in the span has a nominal subject within the span
        elif token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX") and token.i in verb_indices_in_span:
            for child in token.children:
                # Ensure child is within the span's boundaries
                if child.dep_ in ("nsubj", "nsubjpass") and child.i >= doc_span.start and child.i < doc_span.end:
                    span_has_subj = True; break
            if span_has_subj: break # Found subject for this root verb

    if DEBUG_SPLITTING:
        span_text = doc_span.text[:75] + ('...' if len(doc_span.text) > 75 else '')
        logging.debug(f"       DEBUG S/V Check: Span='{span_text}', HasSubj={span_has_subj}, HasVerb={span_has_verb}")
    return span_has_verb and span_has_subj


def find_split_token(sentence_doc: spacy.tokens.Span) -> Optional[Union[spacy.tokens.Token, Tuple[spacy.tokens.Token, Dict]]]:
    """
    Finds the first valid token where a sentence can be split based on prioritized rules.
    Includes targeted checks within Rule 2.1 (SCONJ context), Rule 2.2 (CCONJ context),
    and Rule 2.5 (Comma list context).
    """
    global DEBUG_SPLITTING # Ensure DEBUG_SPLITTING flag is accessible

    conjunctive_adverbs = {'however', 'therefore', 'moreover', 'consequently', 'thus', 'furthermore', 'nevertheless', 'instead', 'otherwise', 'accordingly', 'subsequently', 'hence'}
    dash_chars = {'-', '—', '–'} # Using em dash and en dash as well
    sent_len = len(sentence_doc)
    if DEBUG_SPLITTING: logging.debug(f"\nDEBUG Split Check: '{sentence_doc.text[:100]}...'")

    # Word sets/POS tags for specific rule checks
    interrogative_explanatory_sconjs = {'how', 'if', 'whether', 'why', 'when', 'where'}
    preventing_prev_pos_for_target_sconj = {'VERB', 'SCONJ', 'ADP', 'PREP'} # Note: PREP might be too restrictive, test
    list_pattern_pos = {'NUM', 'NOUN', 'PROPN', 'ADJ'} # Added ADJ for lists like "red, white, and blue"
    noun_like_pos = {'NOUN', 'PROPN', 'PRON'} # Added PRON
    noun_gerund_like_pos = {'NOUN', 'PROPN', 'VERB', 'PRON'} # Added PRON

    # --- Pass 1: Prioritize Semicolon ---
    if DEBUG_SPLITTING: logging.debug(f"--- Splitting Pass 1: Checking for Semicolon ---")
    for i, token in enumerate(sentence_doc):
        if token.text == ';':
            if DEBUG_SPLITTING: logging.debug(f"  DEBUG: Found ';' at index {i}. Checking S/V around it.")
            # Check only non-empty spans
            cl1_span = sentence_doc[0 : i] if i > 0 else None
            cl2_span = sentence_doc[i + 1 : sent_len] if i + 1 < sent_len else None
            if cl1_span and cl2_span and has_subject_and_verb(cl1_span) and has_subject_and_verb(cl2_span):
                if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Priority) - Splitting at ';' (Index {i}) based on S/V check.")
                return token
            else:
                if DEBUG_SPLITTING: logging.debug(f"  DEBUG: Found ';' (Index {i}) but FAILED S/V check around it or resulted in empty clause. Ignoring this semicolon.")

    # --- Pass 2: Check Other Rules ---
    if DEBUG_SPLITTING: logging.debug(f"--- Splitting Pass 2: Checking Other Rules ---")
    for i, token in enumerate(sentence_doc):
        # Skip placeholders and tokens too close to the edge
        if PLACEHOLDER_PATTERN.fullmatch(token.text): continue
        if i == 0 or i >= sent_len - 1: continue

        # Pre-calculate token properties
        is_cc = (token.dep_ == "cc" and token.pos_ == "CCONJ")
        is_mid_sconj = (token.pos_ == 'SCONJ' or token.dep_ == 'mark') # Removed i > 0 check, handled by loop range
        is_comma = (token.text == ',')
        is_dash = (token.text in dash_chars)
        is_colon = (token.text == ':')

        # Rule 2.1: Mid-Sentence SCONJ
        if is_mid_sconj:
            cl1_span = sentence_doc[0 : i] # Always valid as i > 0
            cl2_span = sentence_doc[i + 1 : sent_len] # Always valid as i < sent_len - 1
            if cl1_span and cl2_span and has_subject_and_verb(cl1_span) and has_subject_and_verb(cl2_span):
                is_target_sconj = token.lower_ in interrogative_explanatory_sconjs
                if is_target_sconj: # No need for i > 0 check here again
                    prev_token = sentence_doc[i-1]
                    prev_pos = prev_token.pos_
                    if DEBUG_SPLITTING: logging.debug(f"    DEBUG SCONJ Check: Target SCONJ '{token.text}' preceded by '{prev_token.text}' (POS='{prev_pos}')")
                    if prev_pos in preventing_prev_pos_for_target_sconj:
                        if DEBUG_SPLITTING: logging.debug(f"    DEBUG SCONJ Prevent: Scenario DETECTED. Skipping split.")
                        continue # Skip split for this SCONJ
                if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Pass 2) - Splitting at SCONJ '{token.text}' (Index {i}) based on S/V and context.")
                return token # Return the SCONJ token as the split point

        # Rule 2.2: CCONJ (e.g., and, but, or) with Context Checks
        elif is_cc:
            should_prevent_split_cc = False

            # Context Check 2a: Adj-Adj-Noun (e.g., "red, white, and blue")
            if i > 0 and i < sent_len - 2: # Check boundaries: Need token before, after, and after-after
                t_before_cc = sentence_doc[i-1]; t_after_cc = sentence_doc[i+1]; t_after_after_cc = sentence_doc[i+2]
                if t_before_cc.pos_ == 'ADJ' and t_after_cc.pos_ == 'ADJ' and t_after_after_cc.pos_ in ('NOUN', 'PROPN') and \
                   t_before_cc.head.i == t_after_after_cc.i and t_after_cc.head.i == t_after_after_cc.i:
                    should_prevent_split_cc = True; logging.debug(f"    DEBUG CC Prevent (2a): Adj-Adj-Noun") if DEBUG_SPLITTING else None

            # Context Check 2b: Noun/Gerund/Pronoun Coordination (e.g., "research and outlining", "he and I")
            if not should_prevent_split_cc and i > 0 and i < sent_len - 1:
                t_before_cc = sentence_doc[i-1]; t_after_cc = sentence_doc[i+1]
                try:
                    if (t_before_cc.pos_ in noun_like_pos and
                        t_after_cc.pos_ in noun_gerund_like_pos and
                        t_after_cc.dep_ == 'conj' and t_after_cc.head.i == t_before_cc.i):
                        should_prevent_split_cc = True
                        logging.debug(f"    DEBUG CC Prevent (2b): Likely Noun/Gerund/Pronoun coord ('{t_before_cc.text} {token.text} {t_after_cc.text}')") if DEBUG_SPLITTING else None
                except AttributeError: logging.warning(f"Attribute error during CC context check 2b near index {i}")

            # Context Check 2c: Noun/Pronoun-Noun/Pronoun (same non-verb head)
            if not should_prevent_split_cc and i > 0 and i < sent_len - 1:
                t_before_cc = sentence_doc[i-1]; t_after_cc = sentence_doc[i+1]
                if t_before_cc.pos_ in ('NOUN', 'PROPN', 'PRON') and t_after_cc.pos_ in ('NOUN', 'PROPN', 'PRON') and \
                   t_before_cc.head.i == t_after_cc.head.i:
                   if t_before_cc.head.pos_ not in ('VERB', 'AUX'):
                       should_prevent_split_cc = True; logging.debug(f"    DEBUG CC Prevent (2c): Noun/Pronoun-Noun/Pronoun same non-verb head") if DEBUG_SPLITTING else None

            # Context Check 2d: Verb/Aux-Verb/Aux (shared subject/head)
            if not should_prevent_split_cc and i > 0 and i < sent_len - 1:
                t_before_cc = sentence_doc[i-1]; t_after_cc = sentence_doc[i+1]
                if t_before_cc.pos_ in ('VERB', 'AUX') and t_after_cc.pos_ in ('VERB', 'AUX'):
                    # Check if they share the same subject (nsubj points to the same token index)
                    subj_before = {c.i for c in t_before_cc.children if c.dep_ in ('nsubj', 'nsubjpass')}
                    subj_after = {c.i for c in t_after_cc.children if c.dep_ in ('nsubj', 'nsubjpass')}
                    verbs_share_subject = bool(subj_before and subj_before == subj_after)

                    # Check if they share the same head OR if one is aux of the other
                    verbs_share_head = (t_before_cc.head.i == t_after_cc.head.i or \
                                        t_before_cc.head.i == t_after_cc.i or \
                                        t_after_cc.head.i == t_before_cc.i or \
                                        (t_before_cc.dep_ == 'aux' and t_before_cc.head.i == t_after_cc.i) or \
                                        (t_after_cc.dep_ == 'aux' and t_after_cc.head.i == t_before_cc.i))

                    if verbs_share_subject or verbs_share_head:
                        should_prevent_split_cc = True; logging.debug(f"    DEBUG CC Prevent (2d): Verb/Aux-Verb/Aux (Shared Subj or Head/Aux)") if DEBUG_SPLITTING else None

            # Context Check 2e: List Coordination with Comma before CCONJ (Oxford Comma or just list item)
            if not should_prevent_split_cc and i > 1 and i < sent_len - 1: # Need token before comma, comma, CCONJ, token after
                t_comma = sentence_doc[i-1]        # Token before CCONJ (should be comma)
                t_before_comma = sentence_doc[i-2] # Item before comma
                t_after_conj = sentence_doc[i+1]   # Item after CCONJ

                if t_comma.text == ',':
                    # Check if items around ", CCONJ" are likely list items (using list_pattern_pos)
                    if (t_before_comma.pos_ in list_pattern_pos and
                        t_after_conj.pos_ in list_pattern_pos):
                        # Optional stricter check: POS tags must match (e.g., NOUN, CCONJ NOUN or ADJ, CCONJ ADJ)
                        # if t_before_comma.pos_ == t_after_conj.pos_:
                        should_prevent_split_cc = True
                        logging.debug(f"    DEBUG CC Prevent (2e): Likely list coord with comma ('{t_before_comma.text}, {token.text} {t_after_conj.text}')") if DEBUG_SPLITTING else None


            # Fallback S/V Check for CC if no context rule prevented it
            if not should_prevent_split_cc:
                cl1_span_cc = sentence_doc[0 : i]; cl2_span_cc = sentence_doc[i + 1 : sent_len]
                if cl1_span_cc and cl2_span_cc and has_subject_and_verb(cl1_span_cc) and has_subject_and_verb(cl2_span_cc):
                    if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Pass 2) - Splitting at CC '{token.text}' (Index {i}) based on S/V check.")
                    return token
                elif DEBUG_SPLITTING: logging.debug(f"    DEBUG CC Fallback S/V check FAILED for '{token.text}' (Index {i}).")
            elif DEBUG_SPLITTING: logging.debug(f"  DEBUG: Skipping CCONJ '{token.text}' (Index {i}) due to context rule (2a, 2b, 2c, 2d, or 2e).")


        # Rule 2.3: Comma + Conjunctive Adverb
        elif is_comma and i < sent_len - 1:
            t_after_idx = i + 1
            # Skip over any placeholders immediately after the comma
            while t_after_idx < sent_len and PLACEHOLDER_PATTERN.fullmatch(sentence_doc[t_after_idx].text): t_after_idx += 1
            if t_after_idx >= sent_len : continue # Reached end while skipping placeholders

            t_after = sentence_doc[t_after_idx]
            if t_after.lower_ in conjunctive_adverbs and t_after.pos_ == 'ADV':
                cl1_span_conj = sentence_doc[0 : i] # Clause before comma
                cl2_span_start_idx = t_after_idx + 1 # Clause starts AFTER the conjunctive adverb
                cl2_span_conj = sentence_doc[cl2_span_start_idx : sent_len] if cl2_span_start_idx < sent_len else None
                # Check S/V in both clauses
                if cl1_span_conj and cl2_span_conj and has_subject_and_verb(cl1_span_conj) and has_subject_and_verb(cl2_span_conj):
                    if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Pass 2) - Splitting at ',' (Index {i}) based on Conj. Adverb '{t_after.text}'.")
                    return token # Return the comma as split point

        # Rule 2.4: Comma + Relative 'which' (Non-restrictive clause)
        elif is_comma and i < sent_len - 1:
            t_after_idx = i + 1
            # Skip over any placeholders immediately after the comma
            while t_after_idx < sent_len and PLACEHOLDER_PATTERN.fullmatch(sentence_doc[t_after_idx].text): t_after_idx += 1
            if t_after_idx >= sent_len : continue # Reached end while skipping placeholders

            t_after = sentence_doc[t_after_idx]
            # Check for 'which' used as a relative pronoun (WDT) introducing a clause (common dependencies)
            if t_after.lower_ == 'which' and t_after.tag_ == 'WDT' and t_after.dep_ in ('relcl', 'nsubj', 'nsubjpass', 'dobj', 'pobj', 'acomp', 'advcl'): # Added more potential deps for clauses
                cl1_span_which = sentence_doc[0 : i] # Clause before comma
                cl2_check_start_idx = t_after_idx + 1 # Clause check starts AFTER 'which'
                cl2_check_span = sentence_doc[cl2_check_start_idx : sent_len] if cl2_check_start_idx < sent_len else None
                # Check S/V in first clause AND that the 'which' clause contains a verb
                if cl1_span_which and cl2_check_span and has_subject_and_verb(cl1_span_which) and \
                   any(t.pos_ in ("VERB", "AUX") for t in cl2_check_span if not PLACEHOLDER_PATTERN.fullmatch(t.text)):

                    # Determine pronoun replacement based on 'which' antecedent's number
                    antecedent = t_after.head ; pronoun = "It" # Default to 'It'
                    try:
                        morph_num = antecedent.morph.get('Number')
                        if morph_num and 'Plur' in morph_num: pronoun = "They"
                        # Heuristic for nouns ending in 's' that are likely plural
                        elif antecedent.pos_ == 'NOUN' and antecedent.text.lower().endswith('s') and not morph_num:
                             singular_form = antecedent.lemma_
                             if singular_form != antecedent.text.lower(): pronoun = "They"
                    except Exception as morph_err: logging.warning(f"Morph analysis failed for antecedent '{antecedent.text}': {morph_err}")

                    if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Pass 2) - Splitting at ',' (Index {i}) based on Relative Clause 'which'. Antecedent='{antecedent.text}', Pronoun='{pronoun}'.")
                    # Return the comma and details for replacement
                    return (token, {'type': 'relative_which', 'pronoun': pronoun, 'replace_token': t_after})


        # Rule 2.5: Potential Comma Splice (Independent Clause + Comma + Independent Clause)
        elif is_comma:
            # Check if this comma was already handled by Rule 2.3 (Conj Adverb) or 2.4 (Relative Which)
            is_handled_by_prior_rule = False
            if i < sent_len - 1:
                t_after_idx = i + 1
                while t_after_idx < sent_len and PLACEHOLDER_PATTERN.fullmatch(sentence_doc[t_after_idx].text): t_after_idx += 1
                if t_after_idx < sent_len:
                    t_after = sentence_doc[t_after_idx]
                    if t_after.lower_ in conjunctive_adverbs and t_after.pos_ == 'ADV': is_handled_by_prior_rule = True
                    if t_after.lower_ == 'which' and t_after.tag_ == 'WDT' and t_after.dep_ in ('relcl','nsubj','nsubjpass', 'dobj','pobj','acomp','advcl'): is_handled_by_prior_rule = True

            if not is_handled_by_prior_rule:
                cl1_span_comma = sentence_doc[0 : i]
                cl2_span_comma = sentence_doc[i + 1 : sent_len]
                # Check basic S+V condition first for both clauses
                if cl1_span_comma and cl2_span_comma and has_subject_and_verb(cl1_span_comma) and has_subject_and_verb(cl2_span_comma):

                    # Context check for lists to prevent splitting lists like "..., item1, and item2" at the comma
                    skip_comma_splice = False
                    # Check for pattern: ITEM + COMMA + CCONJ + ITEM
                    if i > 0 and i < sent_len - 2: # Need context: item before, comma, CCONJ, item after
                        t_before = sentence_doc[i-1]        # Item before comma
                        t_after_comma = sentence_doc[i+1] # Token immediately after comma
                        t_after_conj = sentence_doc[i+2]    # Token after that

                        # Check if token after comma is a coordinating conjunction
                        if t_after_comma.pos_ == 'CCONJ':
                            # Check if items around "COMMA CCONJ" are likely list items
                            if (t_before.pos_ in list_pattern_pos and
                                t_after_conj.pos_ in list_pattern_pos):
                                skip_comma_splice = True
                                logging.debug(f"    DEBUG Comma Splice Prevent (2.5 List): Likely list pattern around '{t_before.text}, {t_after_comma.text} {t_after_conj.text}'") if DEBUG_SPLITTING else None

                    # Proceed only if the list check did NOT set the skip flag
                    if not skip_comma_splice:
                        if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Rule 2.5) - Splitting at ',' (Index {i}). Reason: Potential Comma Splice (S/V + S/V).")
                        return token # Return the comma as the split point
                    else:
                        # Skipped due to list pattern detection
                        continue # Continue to next token in the main loop


        # Rule 2.6: Dashes (em dash, en dash, hyphen) likely separating clauses
        elif is_dash:
            # Heuristic: Check if it's likely a hyphen connecting parts of a word (no spaces around)
            is_compound_word_hyphen = False
            if token.text == '-' and i > 0 and i < sent_len - 1:
                prev_token, next_token = sentence_doc[i-1], sentence_doc[i+1]
                # Check if tokens are alphabetic and there are no spaces around the hyphen
                if prev_token.is_alpha and next_token.is_alpha and \
                   prev_token.idx + len(prev_token.text) == token.idx and \
                   token.idx + len(token.text) == next_token.idx:
                    is_compound_word_hyphen = True
                    if DEBUG_SPLITTING: logging.debug(f"    DEBUG Dash Skip: Likely compound word hyphen: '{prev_token.text}{token.text}{next_token.text}'")

            if not is_compound_word_hyphen:
                cl1_span_dash = sentence_doc[0 : i]; cl2_span_dash = sentence_doc[i + 1 : sent_len]
                # Check S/V on both sides
                if cl1_span_dash and cl2_span_dash and has_subject_and_verb(cl1_span_dash) and has_subject_and_verb(cl2_span_dash):
                    if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Pass 2) - Splitting at dash '{token.text}' (Index {i}) based on S/V check.")
                    return token # Return dash token

        # Rule 2.7: Colons introducing an independent clause
        elif is_colon :
            cl1_span_colon = sentence_doc[0:i]; cl2_span_colon = sentence_doc[i+1:sent_len]
            # Check if the part *after* the colon forms an independent clause (has S/V)
            if cl1_span_colon and cl2_span_colon and has_subject_and_verb(cl2_span_colon):
                if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Pass 2) - Splitting at ':' (Index {i}) based on S/V check in second clause.")
                return token # Return colon token


    # --- Rule 2.8 (Check after Pass 2): Initial SCONJ + Comma separating clauses ---
    first_real_token_idx = 0
    # Skip initial placeholders
    while first_real_token_idx < sent_len and PLACEHOLDER_PATTERN.fullmatch(sentence_doc[first_real_token_idx].text): first_real_token_idx += 1

    # Check if the first *actual* token is a subordinating conjunction or marker
    if first_real_token_idx < sent_len and (sentence_doc[first_real_token_idx].pos_ == 'SCONJ' or sentence_doc[first_real_token_idx].dep_ == 'mark'):
        initial_sconj_token = sentence_doc[first_real_token_idx]
        if DEBUG_SPLITTING: logging.debug(f"--- Splitting Rule 2.8: Checking for Initial SCONJ '{initial_sconj_token.text}' + Comma ---")

        comma_found = False
        for i_rule28, token_rule28 in enumerate(sentence_doc):
            # Start search *after* the initial SCONJ
            if token_rule28.i <= first_real_token_idx: continue
            if PLACEHOLDER_PATTERN.fullmatch(token_rule28.text): continue

            # Found a potential separating comma
            if token_rule28.text == ',':
                comma_found = True
                # Clause 1 is between SCONJ and Comma
                cl1_check_span_rule28 = sentence_doc[first_real_token_idx + 1 : i_rule28]
                # Clause 2 is after Comma
                cl2_span_rule28 = sentence_doc[i_rule28 + 1 : sent_len]

                # Check S/V in both potential main clauses (after SCONJ, after comma)
                if cl1_check_span_rule28 and cl2_span_rule28 and \
                   has_subject_and_verb(cl1_check_span_rule28) and \
                   has_subject_and_verb(cl2_span_rule28):

                    # Avoid splitting if the comma is introducing a relative 'which' clause (handled by Rule 2.4)
                    is_which_case = False
                    next_real_token_idx = i_rule28 + 1
                    while next_real_token_idx < sent_len and PLACEHOLDER_PATTERN.fullmatch(sentence_doc[next_real_token_idx].text): next_real_token_idx += 1
                    if next_real_token_idx < sent_len:
                        next_real_token = sentence_doc[next_real_token_idx]
                        if next_real_token.lower_=='which' and next_real_token.tag_=='WDT': is_which_case=True

                    if not is_which_case:
                        if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Rule 2.8) - Splitting at ',' (Index {i_rule28}) following initial SCONJ '{initial_sconj_token.text}'.")
                        return token_rule28 # Return the comma

                break # Found the first comma after the SCONJ, decision made based on S/V


    # --- No split point found by any rule ---
    if DEBUG_SPLITTING and sent_len > 10: # Only log if sentence is reasonably long
        logging.debug(f"DEBUG: No valid split point found for sentence: '{sentence_doc.text[:100]}...'")
    return None

# >>> START OF PART 3 <<<

# ==============================================================================
# 6. HELPER FUNCTIONS: SENTENCE SPLITTING (Part 2 - Application Logic)
# ==============================================================================

def apply_sentence_splitting(text_with_placeholders: str, nlp_model: spacy.Language) -> str:
    """
    Applies sentence splitting logic using find_split_token.
    Includes checks to prevent splits if:
    - The second clause starts with a common dependent marker.
    - Either resulting clause is too short (MIN_SPLIT_WORDS, excluding placeholders).
    """
    global DEBUG_SPLITTING # Make sure it's accessible

    if not text_with_placeholders or not text_with_placeholders.strip(): return ""

    # Minimum number of *actual words* (not placeholders) required in each resulting clause
    MIN_SPLIT_WORDS = 7 # Adjust as needed based on testing

    # Common words/phrases (lowercase, with trailing space) that often start dependent clauses
    # Making this more specific to avoid overly aggressive prevention
    dependent_clause_starters = {
        'how ', 'why ', 'if ', 'whether ', 'when ', 'where ', 'because ',
        'although ', 'though ', 'while ', 'since ', 'unless ', 'until ',
        'after ', 'before ', 'so that ', 'such that ', 'whenever ',
        'wherever ', 'whereas ', 'even if ', 'even though ', 'as if ', 'as though '
        # Omitting 'as ' and 'that ' as they are too ambiguous
    }

    logging.info("Applying sentence splitting rules...")
    output_lines = []; num_splits = 0; num_splits_skipped_length = 0; num_splits_skipped_dependent = 0
    try:
        # Process the text with placeholders using the provided spaCy model
        doc = nlp_model(text_with_placeholders)
        for sent_idx, sent in enumerate(doc.sents):
            original_sentence_text = sent.text
            if not original_sentence_text or not original_sentence_text.strip(): continue

            if DEBUG_SPLITTING: logging.debug(f"\nProcessing Sent #{sent_idx}: '{original_sentence_text[:100]}...'")

            # Preserve original punctuation for the second clause if split occurs
            original_ends_with_question = original_sentence_text.strip().endswith('?')
            original_ends_with_exclamation = original_sentence_text.strip().endswith('!')
            default_ending = "?" if original_ends_with_question else "!" if original_ends_with_exclamation else "."

            # Find a potential split point using the detailed rules
            split_token_result = find_split_token(sent)
            split_details, split_token = None, None
            if isinstance(split_token_result, tuple): split_token, split_details = split_token_result
            elif split_token_result is not None: split_token = split_token_result

            if split_token is not None:
                # --- Potential Split Found - Extract Clauses ---
                split_idx_in_sent = split_token.i - sent.start # Index of split token within the sentence span

                # Determine Clause 1 Text
                clause1_start_char_idx = sent.start_char
                clause1_end_char_idx = split_token.idx # End *before* the split token

                # Handle case: Initial SCONJ ("Because X, Y" -> split at comma -> "X")
                first_real_token_idx_sent = 0
                while first_real_token_idx_sent < len(sent) and PLACEHOLDER_PATTERN.fullmatch(sent[first_real_token_idx_sent].text): first_real_token_idx_sent += 1
                if first_real_token_idx_sent < len(sent) and \
                   (sent[first_real_token_idx_sent].pos_ == 'SCONJ' or sent[first_real_token_idx_sent].dep_ == 'mark') and \
                   split_token.text == ',':
                   if first_real_token_idx_sent + 1 < len(sent):
                       # Start Clause 1 *after* the initial SCONJ
                       clause1_start_char_idx = sent[first_real_token_idx_sent + 1].idx
                   else: # Should not happen if comma exists, but safety check
                       clause1_start_char_idx = split_token.idx

                # Handle case: Split at SCONJ/CCONJ preceeded by comma ("..., and X" -> split at 'and' -> "...")
                if split_token.pos_ != 'PUNCT' and split_idx_in_sent > 0:
                    preceding_token = sent[split_idx_in_sent - 1]
                    if preceding_token.text == ',':
                        clause1_end_char_idx = preceding_token.idx # End clause 1 before the comma

                clause1_raw_text = doc.text[clause1_start_char_idx : clause1_end_char_idx]

                # Determine Clause 2 Text
                clause2_start_char_idx = split_token.idx + len(split_token.text) # Start *after* the split token
                clause2_end_char_idx = sent.end_char
                clause2_prefix = ""

                # Handle case: Relative 'which' replacement
                if split_details and split_details.get('type') == 'relative_which':
                    replace_token = split_details['replace_token']
                    # Start clause 2 *after* the 'which' token that gets replaced
                    clause2_start_char_idx = replace_token.idx + len(replace_token.text)
                    clause2_prefix = split_details['pronoun'] # Add pronoun prefix (e.g., "It", "They")

                # Handle case: Comma followed by Conjunctive Adverb ("..., however X" -> split at comma -> "However X")
                # (Need to check token after split_token)
                elif split_token.text == ',' and (split_token.i + 1) < sent.end:
                    next_token_idx = split_token.i + 1
                    # Skip placeholders after comma
                    while next_token_idx < sent.end and PLACEHOLDER_PATTERN.fullmatch(sent.doc[next_token_idx].text): next_token_idx+=1
                    if next_token_idx < sent.end:
                        next_token = sent.doc[next_token_idx]
                        conjunctive_adverbs_local = {'however', 'therefore', 'moreover', 'consequently', 'thus', 'furthermore', 'nevertheless', 'instead', 'otherwise', 'accordingly', 'subsequently', 'hence'}
                        if next_token.lower_ in conjunctive_adverbs_local and next_token.pos_ == 'ADV':
                            # Start clause 2 *after* the conjunctive adverb
                            clause2_start_char_idx = next_token.idx + len(next_token.text)

                clause2_base = doc.text[clause2_start_char_idx : clause2_end_char_idx]

                # --- Clean and Check Clauses ---
                clause1_cleaned = clause1_raw_text.strip().strip('.,;:')
                clause2_base_cleaned = clause2_base.strip().strip('.,;:')
                clause2_cleaned = f"{clause2_prefix} {clause2_base_cleaned}".strip() if clause2_prefix else clause2_base_cleaned

                # Calculate Word Counts (excluding placeholders)
                clause1_word_count = len([word for word in clause1_cleaned.split() if not PLACEHOLDER_PATTERN.fullmatch(word)])
                clause2_word_count = len([word for word in clause2_cleaned.split() if not PLACEHOLDER_PATTERN.fullmatch(word)])

                if DEBUG_SPLITTING:
                    logging.debug(f"    DEBUG Split Check (Sent #{sent_idx}): Token='{split_token.text}' @ idx {split_token.i}, "
                                   f"C1 words={clause1_word_count} ('{clause1_cleaned[:30]}...'), "
                                   f"C2 words={clause2_word_count} ('{clause2_cleaned[:30]}...'), "
                                   f"MIN_SPLIT={MIN_SPLIT_WORDS}")

                # Check if Clause 2 starts with a dependent marker
                starts_with_dependent = False
                clause2_lower = clause2_cleaned.lower()
                for starter in dependent_clause_starters:
                    if clause2_lower.startswith(starter):
                        starts_with_dependent = True
                        break

                # --- Decision Logic ---
                if not clause1_cleaned or not clause2_cleaned:
                    if DEBUG_SPLITTING: logging.debug(f"    -> SPLIT DECISION (Sent #{sent_idx}): Skipping (Empty Clause)")
                    output_lines.append(original_sentence_text) # Keep original
                    # num_splits_skipped_empty += 1 # Optional: track reason
                elif starts_with_dependent:
                    if DEBUG_SPLITTING: logging.debug(f"    -> SPLIT DECISION (Sent #{sent_idx}): Skipping (Clause 2 starts with dependent marker: '{clause2_cleaned[:30]}...')")
                    output_lines.append(original_sentence_text) # Keep original
                    num_splits_skipped_dependent += 1
                elif clause1_word_count < MIN_SPLIT_WORDS or clause2_word_count < MIN_SPLIT_WORDS:
                    if DEBUG_SPLITTING: logging.debug(f"    -> SPLIT DECISION (Sent #{sent_idx}): Skipping (Too Short: C1={clause1_word_count}, C2={clause2_word_count} < MIN={MIN_SPLIT_WORDS})")
                    output_lines.append(original_sentence_text) # Keep original
                    num_splits_skipped_length += 1
                else:
                    # Proceed with Split
                    if DEBUG_SPLITTING: logging.debug(f"    -> SPLIT DECISION (Sent #{sent_idx}): Proceeding with Split.")
                    num_splits += 1

                    # Capitalization helper
                    def capitalize_first_letter(s: str) -> str:
                        match = re.search(r'([a-zA-Z])', s); # Find first alphabetical character
                        if match:
                            start_index = match.start();
                            return s[:start_index] + s[start_index].upper() + s[start_index+1:]
                        return s # Return original if no letter found

                    clause1_final = capitalize_first_letter(clause1_cleaned).rstrip('.?!') + "." # End first clause with period
                    clause2_final = capitalize_first_letter(clause2_cleaned).rstrip('.?!') + default_ending # Use original ending for second

                    # Append clauses only if they are not empty after cleaning
                    if clause1_final.strip('.'): output_lines.append(clause1_final)
                    if clause2_final.strip(default_ending): output_lines.append(clause2_final)

            else: # No split point found by find_split_token
                if DEBUG_SPLITTING: logging.debug(f"No split point found for Sent #{sent_idx}.")
                output_lines.append(original_sentence_text) # Keep original sentence


        logging.info(f"Sentence splitting applied. Performed {num_splits} splits. Skipped: {num_splits_skipped_length} (length), {num_splits_skipped_dependent} (dependent marker).")

        # Join sentences back, clean up spacing around punctuation
        final_text = " ".join(output_lines).strip()
        final_text = re.sub(r'\s{2,}', ' ', final_text) # Condense multiple spaces
        final_text = re.sub(r'\s+([.,;?!:])', r'\1', final_text) # Remove space before punctuation
        # Ensure space after punctuation where needed (e.g., ".Next" -> ". Next") - handles most cases
        final_text = re.sub(r'([.,;?!:])([^\s\d__])', r'\1 \2', final_text) # Add space after punct if followed by non-space, non-digit, non-placeholder

        return final_text

    except Exception as e:
        logging.error(f"Error during sentence splitting application: {e}", exc_info=True)
        return text_with_placeholders # Fallback to text before splitting on error


# ==============================================================================
# 7. HELPER FUNCTIONS: OPENAI REFINEMENT
# ==============================================================================
def refine_with_openai(text_with_placeholders: str, discipline: str) -> str:
    """Uses OpenAI API (if available) for subtle refinements."""
    if not openai_client:
        logging.warning("OpenAI client not available. Skipping refinement step.")
        return text_with_placeholders

    # Basic check if text is reasonably processable
    if not text_with_placeholders or not text_with_placeholders.strip():
         logging.warning("Input text to OpenAI refinement is empty. Skipping.")
         return text_with_placeholders

    logging.info("Applying OpenAI refinement (around placeholders)...")
    start_time = time.time()

    # Define prompts (Consider making discipline influence the prompt more if needed)
    system_prompt = "You are an expert academic editor specializing in refining texts while preserving specific formatting like placeholders (e.g., __F0__). Your task is to subtly improve fluency, grammar, and style based on the user's rules, outputting only the modified text without explanations."

    user_prompt_parts = [
        "Carefully refine the following text according to the rules provided. Preserve the exact placeholders (like __F0__, __F1__) without modification.",
        "--- TEXT START ---",
        text_with_placeholders,
        "--- TEXT END ---",
        "--- REFINEMENT RULES ---",
        # Note: Applying rules probabilistically (e.g., "50%") is hard for the LLM to do accurately and consistently.
        # Stating the rules directly might be more effective. Consider if these rules are essential or if simpler
        # grammar/style correction is the main goal.
        "1. Passive to Active: Where appropriate and natural, convert passive voice sentences to active voice.",
        "2. Modal Verbs: Replace common modal verbs (can, could, may, might, should, would) with alternatives if it improves naturalness, avoiding overuse.",
        "3. Infinitives to Gerunds: Convert infinitive verb forms ('to analyze') to gerunds ('analyzing') if it enhances flow, but not excessively.",
        "4. Oxford Comma: Consistently remove the Oxford comma in lists of three or more items (e.g., 'A, B and C' NOT 'A, B, and C').",
        "5. Sentence Boundaries: Maintain existing sentence boundaries marked by periods (.), question marks (?), or exclamation points (!). DO NOT merge sentences or replace end punctuation with commas.",
        "6. Initial Conjunctions: If a sentence starts *exactly* with 'And ', 'But ', 'Or ', or 'So ', remove that initial word and the following space.",
        "7. Error Correction: Correct obvious, critical spelling or grammatical errors ONLY. Prioritize preserving the original meaning and structure.",
        f"8. Discipline Context: Maintain a formal, academic tone suitable for the '{discipline}' discipline (unless discipline is 'casual', 'blogger_seo', or 'marketing').",
        "9. Placeholders: VERY IMPORTANT - Do not change, add, or remove any placeholders like __F0__, __F1__, etc. They must remain exactly as they appear in the input text."
    ]
    user_prompt = "\n".join(user_prompt_parts)

    try:
        response = openai_client.chat.completions.create(
            model=CONFIG["openai_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=CONFIG["openai_temperature"],
            max_tokens=CONFIG["openai_max_tokens"] # Ensure this is large enough for the output
        )
        end_time = time.time()

        refined_text = response.choices[0].message.content.strip()

        # Basic validation of the response
        if not refined_text or len(refined_text) < 0.5 * len(text_with_placeholders): # Check if response is suspiciously short
             logging.warning("OpenAI response seems invalid (empty or too short). Returning text before refinement.")
             return text_with_placeholders
        if "sorry" in refined_text.lower() or "cannot fulfill" in refined_text.lower():
             logging.warning("OpenAI response indicates refusal. Returning text before refinement.")
             return text_with_placeholders

        # Placeholder check (Simple version: count should ideally match)
        original_ph_count = len(PLACEHOLDER_PATTERN.findall(text_with_placeholders))
        refined_ph_count = len(PLACEHOLDER_PATTERN.findall(refined_text))
        if original_ph_count > 0 and original_ph_count != refined_ph_count:
             logging.warning(f"OpenAI Placeholder count mismatch! Original: {original_ph_count}, Refined: {refined_ph_count}. Returning text before refinement to preserve placeholders.")
             return text_with_placeholders

        logging.info(f"OpenAI refinement completed in {end_time - start_time:.2f} seconds.")
        if refined_text == text_with_placeholders:
            logging.info("OpenAI returned identical text.")
        else:
            logging.info("OpenAI returned modified text.")
        return refined_text

    except Exception as e:
        logging.error(f"Error during OpenAI API call: {e}", exc_info=True)
        return text_with_placeholders # Return original on error

# ==============================================================================
# 8. MODEL LOADING FUNCTION (Optimized - Called once at startup)
# ==============================================================================
def load_models(device_preference: str = "auto") -> Optional[Dict[str, Any]]:
    """Loads and initializes NLP models (spaCy, Paraphraser, AI Detector), disabling unused spaCy components."""
    logging.info("--- Loading Models ---")
    models = {}

    # Determine compute device (CUDA if available, otherwise CPU)
    try:
        if device_preference == "auto":
            models["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_preference == "cuda" and torch.cuda.is_available():
            models["device"] = torch.device("cuda")
        else:
            models["device"] = torch.device("cpu")
            if device_preference == "cuda": # Log warning only if CUDA was explicitly requested but unavailable
                logging.warning("CUDA requested but not available. Using CPU.")
        logging.info(f"Using device: {models['device']}")
    except Exception as device_err:
        logging.error(f"Error detecting torch device: {device_err}. Defaulting to CPU.")
        models["device"] = torch.device("cpu")


    try:
        # Load spaCy model
        spacy_model_name = CONFIG['spacy_model']
        # *** Optimization: Disable unused components like 'ner' ***
        # Adjust 'disable' list based on actual components needed by your splitting logic (parser, tagger are likely needed)
        disabled_spacy_components = ['ner']
        logging.info(f"Loading spaCy model: {spacy_model_name} (disabling: {disabled_spacy_components})...")
        try:
            models["nlp"] = spacy.load(spacy_model_name, disable=disabled_spacy_components)
        except OSError:
            logging.warning(f"spaCy model '{spacy_model_name}' not found locally. Attempting download...")
            try:
                spacy.cli.download(spacy_model_name)
                models["nlp"] = spacy.load(spacy_model_name, disable=disabled_spacy_components)
                logging.info(f"spaCy model '{spacy_model_name}' downloaded and loaded.")
            except Exception as download_err:
                logging.error(f"Failed to download/load spaCy model '{spacy_model_name}': {download_err}", exc_info=True)
                return None # Critical failure if spaCy model cannot be loaded
        logging.info("spaCy model loaded.")

        # Load Paraphraser (T5 based)
        paraphraser_model_name = CONFIG['paraphraser_model']
        logging.info(f"Loading paraphraser: {paraphraser_model_name}...")
        # Load tokenizer first
        models["tokenizer"] = AutoTokenizer.from_pretrained(paraphraser_model_name)
        # Load model and move to the determined device
        models["model"] = AutoModelForSeq2SeqLM.from_pretrained(paraphraser_model_name).to(models["device"])
        models["model"].eval() # Set model to evaluation mode
        logging.info("Paraphraser model loaded.")

        # Load AI Detector (RoBERTa based)
        ai_detector_model_name = CONFIG['ai_detector_model']
        logging.info(f"Loading AI detector: {ai_detector_model_name}...")
        # Use device mapping for pipeline (-1 for CPU, 0 for first GPU, etc.)
        pipeline_device = 0 if models["device"].type == 'cuda' else -1
        # Temporarily reduce transformers logging during pipeline load if desired
        previous_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        models["ai_detector"] = pipeline("text-classification", model=ai_detector_model_name, device=pipeline_device)
        transformers.logging.set_verbosity(previous_verbosity) # Restore verbosity
        logging.info("AI detector pipeline loaded.")

        logging.info("--- All models loaded successfully ---")
        return models

    except Exception as e:
        logging.exception("Fatal error during model loading:") # Log the full traceback
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

    # Word count check (using simple split)
    word_count = len(input_text.split())
    max_words = config.get("input_max_words", 1000) # Default to 1000 if not in config
    if word_count > max_words:
        logging.error(f"Input text exceeds word limit ({word_count}/{max_words}).")
        # Consider customizing error message based on plan limits in a real app
        return {'status': 'error', 'message': f"Error: Text exceeds word limit ({max_words} words). Provided: {word_count}."}

    logging.info(f"--- Starting Text Processing for request ---")
    logging.info(f"Discipline: {discipline}, Input Words: {word_count}, Freeze Terms: {len(freeze_terms)}")
    start_process_time = time.time()

    try:
        # --- Placeholder Replacement ---
        placeholder_map: Dict[str, str] = {}
        text_with_placeholders = input_text
        placeholder_counter = 0
        # Filter and sort freeze terms (longest first to avoid partial replacements)
        freeze_terms_sorted = sorted([ft for ft in freeze_terms if ft and ft.strip()], key=len, reverse=True)

        if freeze_terms_sorted:
            logging.info(f"Replacing {len(freeze_terms_sorted)} freeze term type(s) with placeholders...")
            processed_spans_map: Dict[int, Tuple[int, str]] = {} # Store start -> (end, placeholder)

            # Iterate through sorted terms to find and mark replacements
            for term in freeze_terms_sorted:
                term_stripped = term.strip()
                if not term_stripped: continue # Skip empty terms

                try:
                    # Use word boundaries for more precise matching, case-insensitive
                    # Note: \b might not work well with punctuation attached. Simple escape might be safer.
                    # pattern = re.compile(r'\b' + re.escape(term_stripped) + r'\b', re.IGNORECASE)
                    pattern = re.compile(re.escape(term_stripped), re.IGNORECASE) # Safer default
                    matches_this_term = []
                    # Find all non-overlapping matches for *this specific term*
                    for match in pattern.finditer(text_with_placeholders): # Search in the current state of text_with_placeholders
                        start, end = match.span()
                        # Check for overlaps with already processed spans
                        is_overlapping = any(max(start, ps) < min(end, pe) for ps, (pe, _) in processed_spans_map.items())
                        if not is_overlapping:
                            matches_this_term.append(match)

                    # If matches found, create placeholder and mark spans
                    if matches_this_term:
                        original_term_matched = matches_this_term[0].group(0) # Use the case from the first match
                        placeholder = f"__F{placeholder_counter}__"
                        placeholder_map[placeholder] = original_term_matched
                        placeholder_counter += 1
                        # Mark all non-overlapping matches for this term with the *same* placeholder
                        for match in matches_this_term:
                            processed_spans_map[match.start()] = (match.end(), placeholder)

                except re.error as regex_err: logging.warning(f"Regex error for freeze term '{term_stripped}': {regex_err}")
                except Exception as term_err: logging.warning(f"Error processing freeze term '{term_stripped}': {term_err}")

            # Perform replacements using the marked spans (from right to left to avoid index issues)
            if processed_spans_map:
                sorted_starts = sorted(processed_spans_map.keys(), reverse=True)
                temp_text_list = list(text_with_placeholders)
                for start in sorted_starts:
                    end, placeholder = processed_spans_map[start]
                    temp_text_list[start:end] = list(placeholder) # Replace slice with placeholder chars
                text_with_placeholders = "".join(temp_text_list)
                logging.info(f"{placeholder_counter} placeholder type(s) created, corresponding to {len(processed_spans_map)} replacements.")

        else:
            logging.info("No valid freeze terms provided or found.")


        # --- Paraphrasing Sentences ---
        logging.info("Tokenizing text (with placeholders) into sentences...")
        try:
            # Use spaCy model loaded earlier
            doc = models["nlp"](text_with_placeholders)
            # Filter empty sentences after stripping whitespace
            sentences = [s.text.strip() for s in doc.sents if s.text and s.text.strip()]
            if not sentences:
                 logging.error("No valid sentences found after placeholder replacement and tokenization.")
                 return {'status': 'error', 'message': 'No valid sentences found to process.'}
        except Exception as nlp_err:
            logging.error(f"Sentence tokenization failed: {nlp_err}", exc_info=True)
            return {'status': 'error', 'message': f'Sentence tokenization failed: {nlp_err}'}

        # Optional: Check length of original sentences before paraphrasing (can be slow)
        # try:
        #     original_doc = models["nlp"](input_text)
        #     original_sentences_map = {i: s.text.strip() for i, s in enumerate(original_doc.sents) if s.text.strip()}
        # except Exception as nlp_err_orig:
        #     logging.warning(f"Could not tokenize original text for length check: {nlp_err_orig}"); original_sentences_map = {}

        paraphrased_sentences = []; sentences_reverted = 0; sentences_failed = 0
        logging.info(f"Paraphrasing {len(sentences)} sentences...")
        max_sentence_tokens = config.get("input_max_sentence_tokens", 480)

        for i, sentence_with_placeholder in enumerate(sentences):
             # Check token length *before* paraphrasing (using the paraphraser tokenizer)
            try:
                # Include the "paraphrase: " prefix in length check if applicable to model
                check_text = f"paraphrase: {sentence_with_placeholder}"
                sentence_tokens = models["tokenizer"].encode(check_text, max_length=max_sentence_tokens + 10, truncation=False) # Check without truncating initially
                if len(sentence_tokens) > max_sentence_tokens:
                    msg = f"Error: Sentence #{i+1} (with placeholders) exceeds token limit for paraphraser ({len(sentence_tokens)}/{max_sentence_tokens}). Skipping paraphrase for this sentence."
                    logging.warning(msg)
                    # Option 1: Skip paraphrase, keep original (with placeholders)
                    paraphrased_sentences.append(sentence_with_placeholder); sentences_reverted += 1
                    # Option 2: Return error immediately (uncomment below)
                    # return { 'status': 'error', 'message': msg.replace("Error: ","") }
                    continue # Move to next sentence
            except Exception as token_err:
                logging.warning(f"Tokenization failed for length check on sentence {i+1}: {token_err}")
                # Decide how to handle - proceed cautiously or skip/error


            # Proceed with paraphrasing
            paraphrased = paraphrase_sentence(
                sentence_with_placeholder,
                models["model"], models["tokenizer"],
                config["paraphrase_num_beams"], config["paraphrase_max_length"]
            )

            # Check for paraphrasing errors
            if "[Paraphrasing Error:" in paraphrased or "[Encoding Error]" in paraphrased:
                logging.error(f"Paraphrasing failed for sentence {i+1}: {paraphrased}")
                # Option 1: Keep original sentence
                paraphrased_sentences.append(sentence_with_placeholder); sentences_failed += 1
                # Option 2: Return error immediately (uncomment below)
                # return {'status': 'error', 'message': f"Error paraphrasing sentence #{i+1}: {paraphrased}"}
                continue # Move to next sentence

            # Verify placeholder preservation (crucial!)
            placeholders_in_original = set(PLACEHOLDER_PATTERN.findall(sentence_with_placeholder))
            if placeholders_in_original:
                placeholders_in_paraphrased = set(PLACEHOLDER_PATTERN.findall(paraphrased))
                if not placeholders_in_original.issubset(placeholders_in_paraphrased):
                    missing = placeholders_in_original - placeholders_in_paraphrased
                    extra = placeholders_in_paraphrased - placeholders_in_original
                    logging.warning(f"Placeholder mismatch in sentence {i+1}. Missing: {missing or 'None'}. Extra: {extra or 'None'}. Reverting sentence.")
                    paraphrased_sentences.append(sentence_with_placeholder); sentences_reverted += 1
                else:
                    paraphrased_sentences.append(paraphrased) # Placeholders look okay
            else:
                paraphrased_sentences.append(paraphrased) # No placeholders to check

        logging.info(f"Paraphrasing complete. Reverted: {sentences_reverted}, Failed: {sentences_failed}.")
        t5_paraphrased_text_with_placeholders = " ".join(paraphrased_sentences)


        # --- Apply Sentence Splitting ---
        # Pass the spaCy model loaded earlier
        text_after_splitting_with_placeholders = apply_sentence_splitting(t5_paraphrased_text_with_placeholders, models["nlp"])

        # --- Optional Filter Trailing Question ---
        # This seems very specific, ensure it's still needed. Can be removed if not.
        unwanted_phrase = "Can you provide some examples?"
        text_to_refine_with_placeholders = text_after_splitting_with_placeholders # Default
        if unwanted_phrase in text_after_splitting_with_placeholders: # Check if phrase exists anywhere first
            temp_text_lower = text_after_splitting_with_placeholders.rstrip('.?! ').lower()
            phrase_lower = unwanted_phrase.rstrip('.?! ').lower()
            if temp_text_lower.endswith(phrase_lower):
                logging.info(f"Filtering detected unwanted trailing phrase: '{unwanted_phrase}'")
                try:
                    # Find the last occurrence and slice before it
                    phrase_start_index = text_after_splitting_with_placeholders.lower().rindex(phrase_lower)
                    text_to_refine_with_placeholders = text_after_splitting_with_placeholders[:phrase_start_index].rstrip()
                except Exception as filter_err:
                     logging.warning(f"Error during trailing phrase filtering: {filter_err}")


        # --- Refine with OpenAI (if client initialized) ---
        refined_text_with_placeholders = refine_with_openai(text_to_refine_with_placeholders, discipline)


        # --- Re-insert Original Freeze Terms ---
        logging.info("Re-inserting original freeze terms...")
        final_text = refined_text_with_placeholders
        num_reinserted = 0

        if placeholder_map: # Only proceed if placeholders were created
            # Clean up potential extra spaces *around* placeholders BEFORE replacing
            try:
                # Remove space(s) immediately BEFORE __F...__ unless preceded by sentence start or another placeholder
                final_text = re.sub(r'(?<!^)(?<!__F\d+__)\s+(__F\d+__)', r'\1', final_text)
                # Remove space(s) immediately AFTER __F...__ unless followed by sentence end or another placeholder
                final_text = re.sub(r'(__F\d+__)\s+(?!$)(?!__F\d+__)', r'\1', final_text)
                logging.info("Attempted cleaning whitespace around placeholders.")
            except Exception as clean_err:
                logging.warning(f"Could not clean whitespace around placeholders: {clean_err}")

            # Replace placeholders with original terms (sorted by number for potential order dependency)
            try:
                # Sort keys numerically: extract number from __F<num>__
                sorted_placeholders = sorted(placeholder_map.keys(), key=lambda p: int(re.search(r'\d+', p).group()))
            except Exception as sort_err:
                logging.error(f"Error sorting placeholders numerically: {sort_err}. Using default sort.")
                sorted_placeholders = sorted(placeholder_map.keys())

            final_text_list = list(final_text) # Work with list for efficient replacement? Or use string replace..
            current_offset = 0 # Track index changes if using list replace

            # Use string replace for simplicity, iterate multiple times if needed for overlaps (though shouldn't happen with sorted longest first)
            for placeholder in sorted_placeholders:
                original_term = placeholder_map[placeholder]
                # Replace placeholder with " original_term " to ensure spacing, will clean later
                # Count occurrences before replacing for logging
                occurrences = final_text.count(placeholder)
                if occurrences > 0:
                    final_text = final_text.replace(placeholder, f" {original_term} ")
                    num_reinserted += occurrences

            logging.info(f"{num_reinserted} placeholder instance(s) re-inserted.")

            # Clean up potential double spaces and spaces around punctuation AFTER all replacements
            final_text = re.sub(r'\s{2,}', ' ', final_text).strip()
            final_text = re.sub(r'\s+([.,;?!:])', r'\1', final_text) # Space before punct
            final_text = re.sub(r'([.,;?!:])([^\s])', r'\1 \2', final_text) # Punct followed by non-space

        else:
            logging.info("No placeholders were used, skipping re-insertion.")


        # --- Final Metrics Calculation ---
        logging.info("Calculating final metrics...")
        paraphrased_ai_score, paraphrased_fk_score, paraphrased_burstiness, paraphrased_ttr, paraphrased_word_count, paraphrased_mod_percentage = 0.0, 0.0, 0.0, 0.0, 0, 0.0
        paraphrased_ai_result = {'label': 'N/A', 'score': 0.0}

        if final_text and final_text.strip():
            try:
                # Use the AI detector pipeline loaded earlier
                # Truncate input to AI detector if necessary (check model's max length)
                # detector_max_len = models["ai_detector"].model.config.max_position_embeddings
                # truncated_final_text = models["ai_detector"].tokenizer.decode(models["ai_detector"].tokenizer.encode(final_text, max_length=detector_max_len, truncation=True))
                # paraphrased_ai_result = models["ai_detector"](truncated_final_text)[0]

                # Assuming detector handles truncation internally or text is within limits:
                paraphrased_ai_result = models["ai_detector"](final_text)[0]
                paraphrased_ai_score = paraphrased_ai_result.get('score', 0.0)
            except Exception as ai_err: logging.error(f"Error calculating final AI score: {ai_err}", exc_info=True); paraphrased_ai_result = {'label': 'Error', 'score': 0.0}

            try: paraphrased_fk_score = flesch_kincaid_grade(final_text)
            except Exception as fk_err: logging.error(f"Error calculating final FK score: {fk_err}")

            paraphrased_burstiness = calculate_burstiness(final_text)
            paraphrased_ttr = calculate_ttr(final_text)
            paraphrased_word_count = len(final_text.split()) # Simple word count
            paraphrased_mod_percentage = calculate_modification_percentage(input_text, final_text)
            logging.info("Final metrics calculated.")
        else:
            logging.warning("Final text is empty. Setting final metrics to default values.")


        # --- Final Freeze Term Preservation Check ---
        final_freeze_terms_preserved = True
        if placeholder_map: # Only check if placeholders were supposed to be there
            logging.info(f"Verifying placeholder removal in final output...")
            if PLACEHOLDER_PATTERN.search(final_text):
                remaining_found = PLACEHOLDER_PATTERN.findall(final_text)
                logging.warning(f"Placeholder(s) still found in final output after re-insertion: {set(remaining_found)}")
                final_freeze_terms_preserved = False
            else:
                logging.info("All placeholders appear to have been removed/replaced successfully.")
        else:
            logging.info("No freeze terms were used, skipping final verification.")
            # final_freeze_terms_preserved remains True


        # --- Structure Output Data ---
        output_data = {
            "status": "success",
            # Renamed key for clarity based on your variable name
            "final_text": final_text,
            "final_metrics": {
                "ai_score_label": paraphrased_ai_result.get('label', 'N/A'),
                "ai_score_value": round(paraphrased_ai_score, 4),
                "flesch_kincaid_grade": round(paraphrased_fk_score, 1),
                "burstiness_score": round(paraphrased_burstiness, 4),
                "type_token_ratio": round(paraphrased_ttr, 4),
                "modification_percentage": round(paraphrased_mod_percentage, 2),
                "word_count": paraphrased_word_count,
                "freeze_terms_preserved": final_freeze_terms_preserved
            }
            # Removed intermediate results keys as per your optimization comment
        }
        end_process_time = time.time()
        logging.info(f"--- Processing Finished (Request Time: {end_process_time - start_process_time:.2f} seconds) ---")
        return output_data

    except Exception as e:
        logging.exception("An unexpected error occurred during core text processing:") # Log full traceback
        # Return a structured error message
        return {'status': 'error', 'message': f"An unexpected processing error occurred: {str(e)}"}


# >>> START OF PART 4 <<<

# ==============================================================================
# 10. FLASK APP INITIALIZATION
# ==============================================================================
app = Flask(__name__)
CORS(app) # Enable CORS for all routes - configure origins properly for production

# Optional: Configure Flask settings if needed (e.g., SECRET_KEY for sessions)
# app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'default-secret-key-change-me')


# ==============================================================================
# 11. GLOBAL MODEL LOADING (Executed once when Flask app/worker starts)
# ==============================================================================
# >>> THIS IS WHERE MODELS ARE LOADED GLOBALLY <<<
logging.info("Initializing Flask app and loading models...")
# Call the model loading function defined in Section 8
loaded_models_global = load_models(device_preference=CONFIG.get("default_device", "auto"))

if not loaded_models_global:
    logging.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    logging.critical("FATAL: Models failed to load during application startup.")
    logging.critical("The /humanize endpoint will return an error until models load successfully.")
    logging.critical("Check logs above for specific model loading errors.")
    logging.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # The application will start, but the endpoint will check loaded_models_global
    # and return an error if it's None.
else:
    logging.info("********************************************************")
    logging.info("Global models loaded successfully. Flask app is ready.")
    logging.info(f"Using device: {loaded_models_global.get('device', 'UNKNOWN')}")
    logging.info("********************************************************")


# ==============================================================================
# 12. FLASK API ENDPOINTS
# ==============================================================================

# Basic root endpoint for testing if the server is reachable
@app.route('/', methods=['GET', 'POST'])
def index():
    # Simple check endpoint
    method = request.method
    logging.info(f"Root endpoint '/' received a {method} request.")
    return jsonify({"status": "ok", "message": f"TextPulse backend alive. Received {method}."}), 200


@app.route('/humanize', methods=['POST'])
def humanize_endpoint():
    """API endpoint to process text using pre-loaded models."""
    start_request_time = time.time()
    logging.info(f"Received request for /humanize from {request.remote_addr}")

    # --- CRITICAL CHECK: Ensure models loaded successfully at startup ---
    if not loaded_models_global:
        logging.error("Models were not loaded during startup, cannot process request.")
        # Return 503 Service Unavailable, as the service is fundamentally not ready
        return jsonify({"status": "error", "message": "Service Unavailable: Backend models failed to load."}), 503

    # --- Get and Validate Input Data ---
    if not request.is_json:
        logging.warning("Request Content-Type is not application/json.")
        return jsonify({"status": "error", "message": "Invalid request format: Content-Type must be application/json"}), 415 # Unsupported Media Type

    data = request.get_json()
    if not data:
        logging.warning("Received empty JSON payload.")
        return jsonify({"status": "error", "message": "No input data received"}), 400 # Bad Request

    input_text = data.get('text')
    discipline = data.get('discipline', DEFAULT_DISCIPLINE)
    freeze_terms = data.get('freeze_terms', [])

    # --- Basic Input Validation ---
    if not input_text: # Check if text exists and is not just whitespace
        logging.warning("Request missing 'text' field or text is empty.")
        return jsonify({"status": "error", "message": "Input 'text' is required and cannot be empty"}), 400
    if not isinstance(freeze_terms, list):
        logging.warning(f"Received 'freeze_terms' is not a list (Type: {type(freeze_terms)}).")
        return jsonify({"status": "error", "message": "'freeze_terms' must be a list of strings"}), 400
    # Ensure all freeze terms are strings (or handle potential errors)
    if not all(isinstance(term, str) for term in freeze_terms):
         logging.warning("Received 'freeze_terms' contains non-string elements.")
         return jsonify({"status": "error", "message": "'freeze_terms' must be a list of strings"}), 400

    if discipline not in VALID_DISCIPLINES:
        logging.warning(f"Received invalid discipline '{discipline}', using default '{DEFAULT_DISCIPLINE}'.")
        discipline = DEFAULT_DISCIPLINE


    # --- Call Core Processing Logic ---
    try:
        # Pass the globally loaded models dictionary
        result_dict = process_text(
            input_text=input_text,
            freeze_terms=freeze_terms,
            discipline=discipline,
            models=loaded_models_global, # Use the models loaded at startup
            config=CONFIG
        )

        # --- Prepare and Return Response ---
        if result_dict.get("status") == "success":
            # Return only the final text and metrics as specified
            response_data = {
                "status": "success",
                "final_text": result_dict.get("final_text", ""), # Use the correct key from process_text output
                "final_metrics": result_dict.get("final_metrics", {})
            }
            status_code = 200
        else:
            # Pass through the error message from process_text
            response_data = {
                "status": "error",
                "message": result_dict.get('message', 'Unknown processing error')
            }
            # Determine status code based on error type if possible
            # 422: Unprocessable Entity (validation errors within process_text)
            # 500: Internal Server Error (unexpected errors)
            msg_lower = response_data['message'].lower()
            if "word limit" in msg_lower or "token limit" in msg_lower or "no valid sentences" in msg_lower:
                 status_code = 422
            else:
                 status_code = 500

        end_request_time = time.time()
        logging.info(f"Request processed in {end_request_time - start_request_time:.2f} seconds. Status: {status_code}")
        return jsonify(response_data), status_code

    except Exception as e:
        # Catch-all for unexpected errors *within the endpoint logic itself*
        # Errors within process_text should be caught and returned by that function.
        logging.exception("An unexpected error occurred within the /humanize endpoint:") # Log full traceback
        return jsonify({"status": "error", "message": f"An internal server error occurred while handling the request."}), 500


# ==============================================================================
# 13. MAIN EXECUTION BLOCK (For Running the Server Directly)
# ==============================================================================
if __name__ == '__main__':
    # This block runs only when the script is executed directly (e.g., `python your_script.py`)
    # It's primarily for local development and testing.

    print("\n" + "="*60)
    print("--- TextPulse Flask Server ---")
    print("="*60)
    print("\n >> Running in direct execution mode (for development/testing) <<\n")

    print(" --- Deployment Instructions ---")
    print(" For production (e.g., on DigitalOcean with Gunicorn):")
    print(" 1. Ensure Gunicorn is installed: pip install gunicorn")
    print(" 2. Run using a command like:")
    # Assuming your script is named 'app.py'. Change 'app:app' if filename or Flask app variable name differs.
    print("    gunicorn --workers 2 --threads 2 --bind 0.0.0.0:5000 app:app --timeout 120")
    print("    - Adjust '--workers' (processes) based on CPU cores (e.g., 2*cores + 1).")
    print("    - Adjust '--threads' (per worker) based on workload (start with 2-4).")
    print("    - Adjust '--timeout' (seconds) if requests take longer (default is 30s).")
    print("    - Ensure sufficient RAM for the number of workers * model size.")
    print("    - Consider running behind a reverse proxy like Nginx.")
    print(" -----------------------------")

    # Run the Flask development server
    # Set debug=False to mimic production loading behavior more closely (no auto-reload)
    # Set debug=True for auto-reloading on code changes during active development (will reload models on change).
    flask_debug_mode = False # Set to True only for active code modification testing
    flask_port = 5001 # Use a distinct port for development server

    print(f"\nStarting Flask development server on http://0.0.0.0:{flask_port}/")
    print(f" --> DEBUG MODE: {'ON (Auto-reload enabled)' if flask_debug_mode else 'OFF (Production-like loading)'} <--")
    print(" (Use Ctrl+C to stop)")
    print("="*60 + "\n")

    # Note: app.run() is NOT suitable for production deployment.
    app.run(debug=flask_debug_mode, host='0.0.0.0', port=flask_port)

# ==============================================================================
# 14. FINAL SCRIPT DOCSTRING (Redundant if top one is detailed)
# ==============================================================================
"""
End of TextPulse.ai Flask Backend Script.
Provides the /humanize API endpoint with pre-loaded ML models.
"""
