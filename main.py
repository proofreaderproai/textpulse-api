# -*- coding: utf-8 -*-
"""
TextPulse.ai Backend Flask Application (v17.1 - Improved Splitter Logic)

Reverted changes related to citation extraction/re-insertion.
Processing Flow: Input -> Preprocess -> T5 (Sentence-by-Sentence) -> Splitter -> Filter -> OpenAI.
Uses humarin/t5-base, num_beams=5.
Logging reduced to show key intermediate outputs at INFO level.
Configured for local testing (Flask debug=True).
**MODIFIED: Sentence splitter logic improved based on safety analysis.**
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
import traceback
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, Tuple, Union
# REMOVED: from collections import defaultdict

# Flask related imports
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# NLP/ML related imports
try:
    import spacy
    import torch
    import transformers # Still needed for T5
    import openai
    from textstat import flesch_kincaid_grade
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
except ImportError as e:
    logging.critical(f"Missing required library: {e}. Please install requirements.", exc_info=True)
    print(f"FATAL Error: Missing required library: {e}.")
    print("Try: pip install Flask flask-cors spacy torch transformers openai textstat python-dotenv")
    sys.exit(1)

# ==============================================================================
# Load .env file
# ==============================================================================
load_dotenv()

# ==============================================================================
# 2. CONFIGURATION & CONSTANTS
# ==============================================================================
# --- Debug Flags ---
# Set DEBUG_SPLITTING to False to minimize splitter logs if needed
DEBUG_SPLITTING = False # Set to False for reduced verbosity

# --- Model & Processing Parameters ---
CONFIG = {
    "spacy_model": "en_core_web_sm",
    "paraphraser_model": "humarin/chatgpt_paraphraser_on_T5_base", #  t5-small # humarin/chatgpt_paraphraser_on_T5_base
    "default_device": "auto",
    "paraphrase_num_beams": 5, # Using beam search=5
    "paraphrase_max_length": 512,
    "input_max_words": 1000,
    "input_max_sentence_tokens": 480, # Max tokens for a *single sentence* input to T5
    "seed": 42,
    "openai_model": "gpt-4o-mini", # gpt-3.5-turbo
    "openai_temperature": 0.3,
    "openai_max_tokens": 1500
}

# --- Other Constants ---
# REMOVED: CITATION_PATTERN_DETECT = re.compile(r'\[\s*\d+(?:\s*[,/-]\s*\d+)*\s*\]')

# --- Set Seed for Reproducibility ---
set_seed(CONFIG["seed"])

# --- Configure Logging for Local Testing ---
# Set level to INFO for reduced verbosity
logging.basicConfig(
    level=logging.INFO, # Changed to INFO
    format='%(asctime)s [%(levelname)s] [%(process)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Suppress INFO messages from underlying libraries if needed (optional)
# logging.getLogger("transformers").setLevel(logging.WARNING)
# logging.getLogger("openai").setLevel(logging.WARNING)
# logging.getLogger("spacy").setLevel(logging.WARNING)


# ==============================================================================
# 3. API KEY SETUP & OPENAI CLIENT INITIALIZATION
# ==============================================================================
# (This section remains unchanged)
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_client = None
logging.info("Attempting to initialize OpenAI client...")
if not openai_api_key or not openai_api_key.startswith("sk-"):
    logging.warning("OpenAI API Key not found/invalid. OpenAI refinement skipped.")
else:
    try:
        openai_client = openai.OpenAI(api_key=openai_api_key)
        logging.info("OpenAI client initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        logging.warning("OpenAI refinement step will be skipped.")
        openai_client = None

# ==============================================================================
# 4. HELPER FUNCTIONS: METRICS CALCULATION
# ==============================================================================
# (Metric functions remain unchanged)
def calculate_burstiness(text: str) -> float:
    words = text.lower().split(); counts = {};
    if not words: return 0.0
    for w in words: counts[w] = counts.get(w, 0) + 1
    freqs = list(counts.values())
    if not freqs: return 0.0
    num_freqs = len(freqs); mean = sum(freqs) / num_freqs
    if mean == 0: return 0.0
    variance = sum([(f - mean) ** 2 for f in freqs]) / num_freqs
    std_dev = variance ** 0.5
    return (std_dev / mean) if mean > 0 else 0.0

def calculate_modification_percentage(text1: str, text2: str) -> float:
    text1 = text1 or ""; text2 = text2 or ""
    if not text1 and not text2: return 0.0
    if not text1 or not text2: return 100.0
    similarity_ratio = SequenceMatcher(None, text1, text2).ratio()
    return (1.0 - similarity_ratio) * 100.0

def calculate_ttr(text: str) -> float:
    if not text or not text.strip(): return 0.0
    words = [token for token in text.lower().split() if token.isalnum()]
    total_tokens = len(words)
    if total_tokens == 0: return 0.0
    unique_types = len(set(words))
    return unique_types / total_tokens

# ==============================================================================
# 5. HELPER FUNCTIONS: PARAPHRASING (Takes one sentence)
# ==============================================================================
def paraphrase_sentence(
    sentence: str, # Input is a single sentence
    model: Any, tokenizer: Any, num_beams: int, max_length: int,
    no_repeat_ngram_size: int = 2, repetition_penalty: float = 1.2
) -> str:
    """Paraphrases a single sentence using the loaded T5-based model (Beam Search)."""
    if not sentence or not sentence.strip():
        # Logged as warning previously, keeping it brief
        # logging.warning("Attempted to paraphrase an empty sentence.")
        return ""
    try:
        input_text = f"paraphrase: {sentence}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=max_length, truncation=True).to(model.device)
        if input_ids.shape[1] == 0:
            logging.error(f"Encoding resulted in empty tensor for sentence: '{sentence[:50]}...'")
            return "[Encoding Error]"
        # Warning about truncation is less critical with INFO level logging
        # if input_ids.shape[1] > max_length:
        #       logging.warning(f"Input sentence encoding length ({input_ids.shape[1]}) > max_length ({max_length}). Truncating.")

        outputs = model.generate(
            input_ids=input_ids, num_beams=num_beams, max_length=max_length, early_stopping=True,
            no_repeat_ngram_size=no_repeat_ngram_size, repetition_penalty=repetition_penalty
        )
        paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return paraphrased
    except Exception as e:
        logging.error(f"Paraphrasing error on sentence: '{sentence[:50]}...': {e}", exc_info=True)
        return f"[Paraphrasing Error: {e}]"

# ==============================================================================
# 6. HELPER FUNCTIONS: SENTENCE SPLITTING (Part 1 - Dependency Checks)
# ==============================================================================
# Note: This section reverts to the version *before* citation handling was added.
# It operates on text directly from T5.
# **MODIFIED: Improved unsafe splitting logic.**

def has_subject_and_verb(doc_span: spacy.tokens.Span) -> bool:
    # (Same as previous versions, DEBUG log depends on DEBUG_SPLITTING flag)
    # Basic check for presence of a verb and a nominal subject related to a verb within the span.
    if not doc_span or len(doc_span) == 0: return False
    span_has_verb, span_has_subj = False, False; verb_indices_in_span = set()
    for token in doc_span:
        if token.pos_ in ("VERB", "AUX"): span_has_verb = True; verb_indices_in_span.add(token.i)
    if not span_has_verb: return False
    for token in doc_span:
        if token.dep_ in ("nsubj", "nsubjpass") and token.head.i in verb_indices_in_span: span_has_subj = True; break
        elif token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX") and token.i in verb_indices_in_span:
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass") and child.i >= doc_span.start and child.i < doc_span.end:
                    span_has_subj = True; break
            if span_has_subj: break
    if DEBUG_SPLITTING:
        span_text = doc_span.text[:75] + ('...' if len(doc_span.text) > 75 else '')
        logging.debug(f"             DEBUG S/V Check: Span='{span_text}', HasSubj={span_has_subj}, HasVerb={span_has_verb}")
    return span_has_verb and span_has_subj

def span_is_likely_single_independent_clause(span: spacy.tokens.Span) -> bool:
    """
    Enhanced check: Checks for S/V and tries to rule out spans containing
    multiple independent clauses joined internally. Very basic heuristic.
    """
    if not span or len(span) < 3: return False # Too short to be independent clause typically
    if not has_subject_and_verb(span): return False

    # Heuristic: Check if the span itself contains strong internal separators
    # that might indicate multiple clauses within this *single* span.
    internal_clause_separators = {";"} # Semicolon is a strong indicator
    # Mid-sentence CCONJ preceded by comma is also strong indicator if not part of list
    # Mid-sentence SCONJ (if rule was active) would also be indicator
    # This is a simplified check
    has_internal_separator = False
    for i, token in enumerate(span):
        if i == 0 or i == len(span) - 1: continue # Ignore start/end tokens of the span
        if token.text in internal_clause_separators:
            has_internal_separator = True
            break
        # Check for internal comma + CCONJ
        # if token.pos_ == 'CCONJ' and span[i-1].text == ',':
             # Further checks needed to distinguish list from clause joining here...
             # Keeping it simple for now, primarily relying on semicolon check.
             # has_internal_separator = True
             # break

    if DEBUG_SPLITTING:
        span_text = span.text[:75] + ('...' if len(span.text) > 75 else '')
        logging.debug(f"             DEBUG SingleClause Check: Span='{span_text}', HasInternalSeparator={has_internal_separator}")

    return not has_internal_separator # If no strong internal separators found, assume single clause


def is_likely_list_content(span: spacy.tokens.Span) -> bool:
    """ Basic heuristic to check if a span looks like list content rather than a clause. """
    if not span or len(span) == 0: return False

    # Check 1: Starts with list markers
    first_token_text = span[0].text
    if first_token_text.isdigit() and span[0].nbor().text == '.': return True # e.g., "1."
    if first_token_text in {'*', '-', '•'}: return True # Bullet points

    # Check 2: Multiple commas separating primarily non-verb phrases
    # (Simple version: count commas vs verbs)
    comma_count = 0
    verb_count = 0
    noun_adj_count = 0
    for token in span:
        if token.text == ',': comma_count += 1
        if token.pos_ in ('VERB', 'AUX'): verb_count += 1
        if token.pos_ in ('NOUN', 'PROPN', 'ADJ', 'NUM'): noun_adj_count +=1

    # Heuristic: if many commas and few verbs relative to nouns/adjectives, likely a list
    if comma_count >= 2 and verb_count <= 1 and noun_adj_count > comma_count:
        if DEBUG_SPLITTING: logging.debug(f"             DEBUG List Check: Likely List (commas={comma_count}, verbs={verb_count}, nouns={noun_adj_count})")
        return True

    if DEBUG_SPLITTING: logging.debug(f"             DEBUG List Check: Not detected as list.")
    return False


def find_split_token(sentence_doc: spacy.tokens.Span) -> Optional[Union[spacy.tokens.Token, Tuple[spacy.tokens.Token, Dict]]]:
    # (Reverted to version before is_likely_citation_token helper and checks)
    # **MODIFIED: Rules 2.1, 2.4 disabled. Rules 2.6, 2.7 enhanced.**
    global DEBUG_SPLITTING
    conjunctive_adverbs = {'however', 'therefore', 'moreover', 'consequently', 'thus', 'furthermore', 'nevertheless', 'instead', 'otherwise', 'accordingly', 'subsequently', 'hence'}
    dash_chars = {'-', '—', '–'}
    sent_len = len(sentence_doc)
    if DEBUG_SPLITTING: logging.debug(f"\nDEBUG Split Check: '{sentence_doc.text[:100]}...'")
    # REMOVED interrogative_explanatory_sconjs as Rule 2.1 is disabled
    # REMOVED preventing_prev_pos_for_target_sconj as Rule 2.1 is disabled
    list_pattern_pos = {'NUM', 'NOUN', 'PROPN', 'ADJ'}
    noun_like_pos = {'NOUN', 'PROPN', 'PRON'}
    noun_gerund_like_pos = {'NOUN', 'PROPN', 'VERB', 'PRON'}

    # Pass 1: Semicolon (Generally Safe)
    if DEBUG_SPLITTING: logging.debug(f"--- Splitting Pass 1: Checking for Semicolon ---")
    for i, token in enumerate(sentence_doc):
        # REMOVED: if is_likely_citation_token(token): continue
        if token.text == ';':
            if DEBUG_SPLITTING: logging.debug(f"  DEBUG: Found ';' at index {i}. Checking S/V.")
            # Get spans simply
            cl1_span = sentence_doc[0 : i] if i > 0 else None
            cl2_span = sentence_doc[i + 1 : sent_len] if i + 1 < sent_len else None
            if cl1_span and cl2_span and has_subject_and_verb(cl1_span) and has_subject_and_verb(cl2_span):
                if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Priority) - Splitting at ';'.")
                return token
            elif DEBUG_SPLITTING: logging.debug(f"  DEBUG: ';' FAILED S/V check. Ignoring.")

    # Pass 2: Other Rules
    if DEBUG_SPLITTING: logging.debug(f"--- Splitting Pass 2: Checking Other Rules ---")
    for i, token in enumerate(sentence_doc):
        # REMOVED: Citation token checks
        if i == 0 or i >= sent_len - 1: continue
        # is_mid_sconj = (token.pos_=='SCONJ' or token.dep_=='mark') # Keep flag for logic below if needed, but rule disabled
        is_cc, is_comma, is_dash, is_colon = (token.dep_=="cc" and token.pos_=="CCONJ"), (token.text==','), (token.text in dash_chars), (token.text==':')

        # --- Simplified get_clause_spans (no citation logic) ---
        def get_clause_spans(split_idx: int, subtract_token_from_c1: bool = False) -> Tuple[Optional[spacy.tokens.Span], Optional[spacy.tokens.Span]]:
            c1_end_idx = split_idx
            if subtract_token_from_c1 and split_idx > 0:
                c1_end_idx = split_idx -1
            c2_start_idx = split_idx + 1
            cl1_s = sentence_doc[0 : c1_end_idx] if c1_end_idx > 0 else None
            cl2_s = sentence_doc[c2_start_idx : sent_len] if c2_start_idx < sent_len else None
            return cl1_s, cl2_s
        # --- End Span Helper ---

        # Rule 2.1: Mid-Sentence SCONJ -- DISABLED (Highly Unsafe)
        # if is_mid_sconj:
        #    if DEBUG_SPLITTING: logging.debug(f"  DEBUG: Skipping potential split at Mid-Sentence SCONJ '{token.text}' (Rule Disabled).")
        #    continue # Explicitly skip this rule

        # Rule 2.2: CCONJ (Conditionally Safe - Relies on Comma + S/V Fallback)
        if is_cc: # Changed from elif to if, as Rule 2.1 is gone
            should_prevent_split_cc = False
            # Context checks to prevent splitting within phrases (same as before)
            if i > 0 and i < sent_len - 2:
                 t_before_cc, t_cc, t_after_cc, t_after_after_cc = sentence_doc[i-1], token, sentence_doc[i+1], sentence_doc[i+2]
                 if t_before_cc.pos_ == 'ADJ' and t_after_cc.pos_ == 'ADJ' and t_after_after_cc.pos_ in ('NOUN', 'PROPN'):
                      if t_after_cc.dep_ == 'conj' and t_after_cc.head.i == t_before_cc.i and t_before_cc.head.i == t_after_after_cc.i:
                           should_prevent_split_cc = True; logging.debug(f"     DEBUG CC Prevent (2a REVISED): Adj-Adj-Noun.") if DEBUG_SPLITTING else None
            if not should_prevent_split_cc and i > 0 and i < sent_len - 1:
                 t_before_cc, t_after_cc = sentence_doc[i-1], sentence_doc[i+1]
                 try:
                      if (t_before_cc.pos_ in noun_like_pos and t_after_cc.pos_ in noun_gerund_like_pos and t_after_cc.dep_ == 'conj' and t_after_cc.head.i == t_before_cc.i):
                           should_prevent_split_cc = True; logging.debug(f"     DEBUG CC Prevent (2b): Noun/Gerund coord") if DEBUG_SPLITTING else None
                 except AttributeError: logging.warning(f"Attr error CC check 2b")
            if not should_prevent_split_cc and i > 0 and i < sent_len - 1:
                 t_before_cc, t_after_cc = sentence_doc[i-1], sentence_doc[i+1]
                 if t_before_cc.pos_ in ('NOUN','PROPN','PRON') and t_after_cc.pos_ in ('NOUN','PROPN','PRON') and t_before_cc.head.i == t_after_cc.head.i and t_before_cc.head.pos_ not in ('VERB', 'AUX'):
                      should_prevent_split_cc = True; logging.debug(f"     DEBUG CC Prevent (2c): Noun/Pronoun same head.") if DEBUG_SPLITTING else None
            if not should_prevent_split_cc and i > 0 and i < sent_len - 1:
                 t_before_cc, t_after_cc = sentence_doc[i-1], sentence_doc[i+1]
                 if t_before_cc.pos_ in ('VERB', 'AUX') and t_after_cc.pos_ in ('VERB', 'AUX'):
                      verb2_has_own_subject = any(child.dep_ in ('nsubj', 'nsubjpass') for child in t_after_cc.children)
                      is_direct_conj = (t_after_cc.dep_ == 'conj' and t_after_cc.head.i == t_before_cc.i)
                      if DEBUG_SPLITTING and not verb2_has_own_subject: logging.debug(f"     DEBUG CC Check (2d v3): Verb2 lacks own subject.")
                      if DEBUG_SPLITTING and is_direct_conj: logging.debug(f"     DEBUG CC Check (2d v3): Verb2 direct conjunct.")
                      if not verb2_has_own_subject or is_direct_conj:
                           should_prevent_split_cc = True; logging.debug(f"     DEBUG CC Prevent (2d v3): Verb coord detected.") if DEBUG_SPLITTING else None
            if not should_prevent_split_cc and i > 1 and i < sent_len - 1:
                 t_comma, t_before_comma, t_after_conj = sentence_doc[i-1], sentence_doc[i-2], sentence_doc[i+1]
                 if t_comma.text==',' and t_before_comma.pos_ in list_pattern_pos and t_after_conj.pos_ in list_pattern_pos:
                      should_prevent_split_cc = True; logging.debug(f"     DEBUG CC Prevent (2e): List coord w/ comma.") if DEBUG_SPLITTING else None

            # Fallback S/V Check (Only split if preceded by comma and S/V check passes)
            if not should_prevent_split_cc:
                preceded_by_comma = (i > 0 and sentence_doc[i-1].text == ',')
                if DEBUG_SPLITTING and not preceded_by_comma: logging.debug(f"     DEBUG CC Split Prevent: CCONJ '{token.text}' not preceded by comma.")
                if preceded_by_comma:
                    cl1_span_cc, cl2_span_cc = get_clause_spans(i, subtract_token_from_c1=True) # Use simplified spans
                    if cl1_span_cc and cl2_span_cc and has_subject_and_verb(cl1_span_cc) and has_subject_and_verb(cl2_span_cc):
                        if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Pass 2) - Splitting at CCONJ '{token.text}' (Comma+S/V passed).")
                        return token
                    elif DEBUG_SPLITTING: logging.debug(f"     DEBUG CC Fallback S/V check FAILED despite comma.")
            elif DEBUG_SPLITTING: logging.debug(f"  DEBUG: Skipping CCONJ '{token.text}' due to context rule.")

        # Rule 2.3 & 2.4 Combined Check for Comma
        elif is_comma and i < sent_len - 1:
            t_after_idx = i + 1
            if t_after_idx < sent_len:
                 t_after = sentence_doc[t_after_idx]
                 # Rule 2.3 (Conj Adverb - Generally Safe)
                 if t_after.lower_ in conjunctive_adverbs and t_after.pos_ == 'ADV':
                      cl2_span_start_idx = t_after_idx + 1
                      cl1_span_conj = sentence_doc[0 : i] if i > 0 else None
                      cl2_span_conj = sentence_doc[cl2_span_start_idx : sent_len] if cl2_span_start_idx < sent_len else None
                      # REMOVED: Citation refinement for cl2_span_conj
                      if cl1_span_conj and cl2_span_conj and has_subject_and_verb(cl1_span_conj) and has_subject_and_verb(cl2_span_conj):
                           if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Pass 2) - Splitting at ',' based on Conj Adverb.")
                           return token
                 # Rule 2.4 (Relative 'which') -- DISABLED (Highly Unsafe)
                 # elif t_after.lower_ == 'which' and t_after.tag_ == 'WDT' and t_after.dep_ in ('relcl', 'nsubj', 'nsubjpass', 'dobj', 'pobj', 'acomp', 'advcl'):
                 #    if DEBUG_SPLITTING: logging.debug(f"  DEBUG: Skipping potential split at Comma + 'which' (Rule Disabled).")
                 #    continue # Explicitly skip


        # Rule 2.6: Dashes (Moderately Unsafe - Enhanced Logic)
        elif is_dash:
            is_compound_word_hyphen = False # Compound word check (same as before)
            if token.text == '-' and i > 0 and i < sent_len - 1:
                 prev_token, next_token = sentence_doc[i-1], sentence_doc[i+1]
                 if prev_token.is_alpha and next_token.is_alpha and \
                    prev_token.idx+len(prev_token.text)==token.idx and token.idx+len(token.text)==next_token.idx:
                      is_compound_word_hyphen = True
                      if DEBUG_SPLITTING: logging.debug(f"     DEBUG Dash Skip: Compound word hyphen.")
            if not is_compound_word_hyphen:
                cl1_span_dash, cl2_span_dash = get_clause_spans(i) # Use simplified spans
                # **ENHANCED CHECK:** Require both sides to look like single independent clauses
                if cl1_span_dash and cl2_span_dash and \
                   span_is_likely_single_independent_clause(cl1_span_dash) and \
                   span_is_likely_single_independent_clause(cl2_span_dash):
                    if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Pass 2) - Splitting at dash '{token.text}' (Enhanced Check Passed).")
                    return token
                elif DEBUG_SPLITTING:
                    logging.debug(f"  DEBUG: Dash split FAILED enhanced check (e.g., S/V missing, internal complexity detected).")


        # Rule 2.7: Colons (Unsafe - Enhanced Logic)
        elif is_colon :
            cl1_span_colon, cl2_span_colon = get_clause_spans(i) # Use simplified spans
            # **ENHANCED CHECK:** Require S/V BEFORE colon, S/V AFTER colon, and check if content AFTER looks like a list.
            if cl1_span_colon and cl2_span_colon and \
               has_subject_and_verb(cl1_span_colon) and \
               has_subject_and_verb(cl2_span_colon) and \
               not is_likely_list_content(cl2_span_colon):
                 if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Pass 2) - Splitting at ':' (Enhanced Check Passed).")
                 return token
            elif DEBUG_SPLITTING:
                 # Log reason for failure
                 reason = []
                 if not (cl1_span_colon and cl2_span_colon): reason.append("Invalid spans")
                 else:
                     if not has_subject_and_verb(cl1_span_colon): reason.append("No S/V before")
                     if not has_subject_and_verb(cl2_span_colon): reason.append("No S/V after")
                     if is_likely_list_content(cl2_span_colon): reason.append("Looks like list after")
                 logging.debug(f"  DEBUG: Colon split FAILED enhanced check. Reason(s): {', '.join(reason) or 'Unknown'}.")

    # Rule 2.8: Initial SCONJ + Comma (Generally Safe)
    # This rule remains the same - it's considered relatively safe.
    first_real_token_idx = 0
    # Basic check for initial SCONJ or mark
    if first_real_token_idx < sent_len and (sentence_doc[first_real_token_idx].pos_ == 'SCONJ' or sentence_doc[first_real_token_idx].dep_ == 'mark'):
        initial_sconj_token = sentence_doc[first_real_token_idx]
        if DEBUG_SPLITTING: logging.debug(f"--- Splitting Rule 2.8: Check Initial '{initial_sconj_token.text}' + Comma ---")
        for i_rule28, token_rule28 in enumerate(sentence_doc):
            if token_rule28.i <= first_real_token_idx: continue
            # REMOVED: if is_likely_citation_token(token_rule28): continue
            if token_rule28.text == ',':
                # Spans relative to this comma
                cl1_check_span_rule28 = sentence_doc[first_real_token_idx + 1 : i_rule28]
                cl2_span_rule28 = sentence_doc[i_rule28 + 1 : sent_len]
                # REMOVED: Citation refinement for cl2_span_rule28
                if cl1_check_span_rule28 and cl2_span_rule28 and has_subject_and_verb(cl1_check_span_rule28) and has_subject_and_verb(cl2_span_rule28):
                    # Avoid splitting if the part after comma starts with 'which' (handled by disabled rule 2.4 anyway)
                    is_which_case = False; next_real_token_idx = i_rule28 + 1
                    if next_real_token_idx < sent_len:
                         next_real_token = sentence_doc[next_real_token_idx]
                         if next_real_token.lower_=='which' and next_real_token.tag_=='WDT': is_which_case=True
                    if not is_which_case:
                         if DEBUG_SPLITTING: logging.debug(f"  DEBUG: SUCCESS (Rule 2.8) - Splitting at ',' after initial SCONJ.")
                         return token_rule28 # Return the comma token
                break # Only check the first comma after the initial SCONJ

    # No split point found
    if DEBUG_SPLITTING and sent_len > 10: logging.debug(f"DEBUG: No valid split point found by any rule.")
    return None

# --- END OF PART 1 ---

# --- START OF PART 2 ---

# ==============================================================================
# 6. HELPER FUNCTIONS: SENTENCE SPLITTING (Part 2 - Application Logic)
# ==============================================================================

def apply_sentence_splitting(text_to_split: str, nlp_model: spacy.Language) -> str:
    """
    Applies sentence splitting logic using find_split_token.
    Operates on text directly from T5 output (no citation handling here).
    **Uses the modified find_split_token with improved logic.**
    """
    global DEBUG_SPLITTING
    if not text_to_split or not text_to_split.strip(): return ""
    MIN_SPLIT_WORDS = 5
    dependent_clause_starters = {
        'how ', 'why ', 'if ', 'whether ', 'when ', 'where ', 'because ',
        'although ', 'though ', 'while ', 'since ', 'unless ', 'until ',
        'after ', 'before ', 'so that ', 'such that ', 'whenever ',
        'wherever ', 'whereas ', 'even if ', 'even though ', 'as if ', 'as though '
        # Note: 'which ' might be needed if Rule 2.4 was active and produced fragments
    }
    # Use INFO level for this summary message
    logging.info("Applying sentence splitting rules...")
    output_lines = []; num_splits = 0; num_splits_skipped_length = 0; num_splits_skipped_dependent = 0
    try:
        # Ensure spacy max_length is sufficient
        required_length = len(text_to_split) + 100 # Add buffer
        if nlp_model.max_length < required_length:
             # Increase incrementally rather than based on text length directly
             # to avoid excessive memory allocation if not needed often.
             # This simple increase might need refinement if very long texts are common.
             new_max_length = max(required_length, nlp_model.max_length * 2)
             nlp_model.max_length = new_max_length
             logging.warning(f"Increased spacy max_length to {nlp_model.max_length} for splitting input.")

        doc = nlp_model(text_to_split)
        for sent_idx, sent in enumerate(doc.sents):
            original_sentence_text = sent.text
            if not original_sentence_text or not original_sentence_text.strip(): continue
            # DEBUG log depends on flag
            if DEBUG_SPLITTING: logging.debug(f"\nProcessing Sent #{sent_idx} for splitting: '{original_sentence_text[:100]}...'")

            # REMOVED: Citation extraction logic

            sent_to_check = sent # Check the original sentence span directly

            # Determine ending based on the original sentence
            original_ends_with_question = original_sentence_text.strip().endswith('?')
            original_ends_with_exclamation = original_sentence_text.strip().endswith('!')
            default_ending = "?" if original_ends_with_question else "!" if original_ends_with_exclamation else "."

            split_token_result = find_split_token(sent_to_check) # Check the sentence directly
            split_details, split_token = None, None
            # Handle tuple return (though disabled Rule 2.4 was main user)
            if isinstance(split_token_result, tuple): split_token, split_details = split_token_result
            elif split_token_result is not None: split_token = split_token_result

            if split_token is not None:
                split_idx_in_sent = split_token.i - sent_to_check.start

                # --- Determine Clause Texts directly from sentence ---
                clause1_start_char_idx = sent_to_check.start_char
                clause1_end_char_idx = split_token.idx # End *before* the split token char

                clause2_start_char_idx = split_token.idx + len(split_token.text) # Start *after* the split token char
                clause2_end_char_idx = sent_to_check.end_char
                clause2_prefix = ""

                # --- Adjust boundaries based on split type ---
                # Case 1: Initial SCONJ + Comma (Rule 2.8) - Remove initial SCONJ from Clause 1
                # Check if the split token is a comma AND the sentence starts with SCONJ/mark
                if split_token.text == ',' and (sent_to_check[0].pos_ == 'SCONJ' or sent_to_check[0].dep_ == 'mark'):
                    # Start clause 1 text after the initial SCONJ token
                    if 1 < len(sent_to_check): clause1_start_char_idx = sent_to_check[1].idx
                    else: clause1_start_char_idx = split_token.idx # Should not happen if S/V checks pass

                # Case 2: CCONJ preceded by Comma (Rule 2.2 Fallback) - Remove comma before CCONJ from Clause 1
                # If split token is CCONJ and previous token is comma
                if split_token.pos_ == 'CCONJ' and split_idx_in_sent > 0 and sent_to_check[split_idx_in_sent - 1].text == ',':
                    preceding_comma_token = sent_to_check[split_idx_in_sent - 1]
                    clause1_end_char_idx = preceding_comma_token.idx # End before the comma

                # Case 3: Comma + Conjunctive Adverb (Rule 2.3) - Start Clause 2 after Adverb
                if split_token.text == ',' and (split_token.i + 1) < sent_to_check.end:
                    next_token_idx = split_token.i + 1
                    if next_token_idx < sent_to_check.end:
                         next_token = sent_to_check.doc[next_token_idx]
                         conjunctive_adverbs_local = {'however', 'therefore', 'moreover', 'consequently', 'thus', 'furthermore', 'nevertheless', 'instead', 'otherwise', 'accordingly', 'subsequently', 'hence'}
                         if next_token.lower_ in conjunctive_adverbs_local and next_token.pos_ == 'ADV':
                              clause2_start_char_idx = next_token.idx + len(next_token.text) # Start after the adverb

                # Case 4: Relative 'which' (Rule 2.4) -- DISABLED, logic removed
                # if split_details and split_details.get('type') == 'relative_which':
                #      # This case should no longer occur as the rule is disabled in find_split_token
                #      pass

                # --- Get clause texts based on adjusted indices ---
                clause1_raw_text = doc.text[clause1_start_char_idx : clause1_end_char_idx]
                clause2_base = doc.text[clause2_start_char_idx : clause2_end_char_idx]

                # Clean up whitespace and punctuation
                clause1_cleaned = clause1_raw_text.strip().strip('.,;:')
                clause2_base_cleaned = clause2_base.strip().strip('.,;:')
                clause2_cleaned = f"{clause2_prefix} {clause2_base_cleaned}".strip() if clause2_prefix else clause2_base_cleaned

                # --- Word Count & Final Checks ---
                clause1_word_count = len(clause1_cleaned.split())
                clause2_word_count = len(clause2_cleaned.split())

                if DEBUG_SPLITTING: logging.debug(f"   DEBUG Final Split Check: C1='{clause1_cleaned[:50]}...'({clause1_word_count}w), C2='{clause2_cleaned[:50]}...'({clause2_word_count}w), MIN={MIN_SPLIT_WORDS}")

                starts_with_dependent = False
                clause2_lower = clause2_cleaned.lower()
                for starter in dependent_clause_starters:
                    if clause2_lower.startswith(starter): starts_with_dependent = True; break

                # Perform final checks before accepting the split
                if not clause1_cleaned or not clause2_cleaned:
                    if DEBUG_SPLITTING: logging.debug(f"     -> Final Decision: Skipping (Empty Clause after cleaning)")
                    output_lines.append(original_sentence_text)
                elif starts_with_dependent:
                    if DEBUG_SPLITTING: logging.debug(f"     -> Final Decision: Skipping (Clause 2 starts dependent marker: '{clause2_lower.split()[0]}')")
                    output_lines.append(original_sentence_text); num_splits_skipped_dependent += 1
                elif clause1_word_count < MIN_SPLIT_WORDS or clause2_word_count < MIN_SPLIT_WORDS:
                    if DEBUG_SPLITTING: logging.debug(f"     -> Final Decision: Skipping (Too Short)")
                    output_lines.append(original_sentence_text); num_splits_skipped_length += 1
                else:
                    # Split accepted
                    if DEBUG_SPLITTING: logging.debug(f"     -> Final Decision: Proceeding with Split.")
                    num_splits += 1
                    def capitalize_first_letter(s: str) -> str:
                         match = re.search(r'([a-zA-Z])', s);
                         if match: return s[:match.start()] + s[match.start()].upper() + s[match.start()+1:]
                         return s

                    # Capitalize first letter of each new sentence and add punctuation
                    clause1_final = capitalize_first_letter(clause1_cleaned).rstrip('.?!') + "."
                    # REMOVED: Citation appending to clause 2
                    clause2_final = capitalize_first_letter(clause2_cleaned).rstrip('.?!') + default_ending # Use original ending for second part

                    if clause1_final.strip('.'): output_lines.append(clause1_final)
                    if clause2_final.strip(default_ending).strip(): output_lines.append(clause2_final)
            else: # No split point found by find_split_token
                 output_lines.append(original_sentence_text) # Keep original sentence

        # Log summary at INFO level
        logging.info(f"Sentence splitting summary: Performed {num_splits} splits. Skipped: {num_splits_skipped_length} (length), {num_splits_skipped_dependent} (dependent marker).")
        final_text = " ".join(output_lines).strip()
        # Basic post-processing cleanup (same as before)
        final_text = re.sub(r'\s{2,}', ' ', final_text)
        final_text = re.sub(r'\s+([.,;?!:])', r'\1', final_text)
        # Ensure space after punctuation if followed by letter/number (more robust)
        final_text = re.sub(r'([.,;?!])([a-zA-Z0-9])', r'\1 \2', final_text)
        return final_text.strip()
    except Exception as e:
        logging.error(f"Error during sentence splitting application: {e}", exc_info=True)
        logging.warning("An error occurred in apply_sentence_splitting. Returning original text.")
        return text_to_split # Return original text on error


# ==============================================================================
# 7. HELPER FUNCTIONS: OPENAI REFINEMENT (Reverted Prompt)
# ==============================================================================
# (This function remains unchanged from the input)
def refine_with_openai(text_to_refine: str) -> str:
    """Uses OpenAI API (if available) for subtle refinements on plain text."""
    if not openai_client:
        logging.warning("OpenAI client not available. Skipping refinement step.")
        return text_to_refine
    if not text_to_refine or not text_to_refine.strip():
        logging.warning("Input text to OpenAI refinement is empty. Skipping.")
        return text_to_refine

    # Use INFO level for this message
    logging.info("Applying OpenAI refinement...")
    start_time = time.time()

    # REVERTED: Removed citation preservation rule from prompt
    system_prompt = "You are a punctuation and spelling repair-bot. Follow the strict rules given. Your output will be displayed in a structured web app interface. Do not include any explanations or commentary—only return the modified text."
    user_prompt_parts_cleaned = [
        "TEXT TO MODIFY:",
        text_to_refine,
        "STRICT RULES TO FOLLOW:",
        """
- The input text contains sentences. DO NOT merge these sentences. Apply fixes ONLY *within* the boundaries of each existing sentence. Treat them as separate entities which cannot be merged. Never replace a period with a comma. 
- If a critical misspelling or punctuation error is encountered, fix it, while keeping wording and structural changes to a bare minimum.  
- For about half of the passive voice sentences in the text, change them to the active voice, while keeping wording and structural changes to a bare minimum.  
- For about half of the sentences in the text, Replace infinitive verb forms (e.g., 'to analyze') with gerunds (e.g., 'analyzing'), while keeping wording and structural changes to a bare minimum.
- For about half of the  sentences in the text, replace common modal verbs (e.g., can, could, may, might, should, would) with more natural or conversational alternatives.
- Remove the Oxford comma in lists of three or more items. Whenever a list of items is provided, do not include a comma before the final 'and' or 'or'.
        """
    ]

    user_prompt = "\n".join(user_prompt_parts_cleaned)

    try:
        response = openai_client.chat.completions.create(
            model=CONFIG["openai_model"], messages=[ {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt} ],
            temperature=CONFIG["openai_temperature"], max_tokens=CONFIG["openai_max_tokens"]
        )
        end_time = time.time()
        refined_text = response.choices[0].message.content.strip()
        if not refined_text or len(refined_text) < 0.5 * len(text_to_refine):
             logging.warning("OpenAI response seems invalid (too short). Reverting.")
             return text_to_refine
        if "sorry" in refined_text.lower() or "cannot fulfill" in refined_text.lower():
             logging.warning("OpenAI response indicates refusal. Reverting.")
             return text_to_refine
        # REMOVED: Citation preservation check

        # Log summary at INFO level
        logging.info(f"OpenAI refinement completed in {end_time - start_time:.2f} seconds.")
        # Optionally log if text was modified or not
        # if refined_text == text_to_refine: logging.info("OpenAI returned identical text.")
        # else: logging.info("OpenAI returned modified text.")
        return refined_text
    except Exception as e:
        logging.error(f"Error during OpenAI API call: {e}", exc_info=True)
        return text_to_refine

# ==============================================================================
# 8. MODEL LOADING FUNCTION (AI Detector Removed)
# ==============================================================================
# (Function remains the same, loads spaCy and T5)
def load_models(device_preference: str = "auto") -> Optional[Dict[str, Any]]:
    logging.info("--- Loading Models ---")
    models = {}
    try:
        if device_preference == "auto": models["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_preference == "cuda" and torch.cuda.is_available(): models["device"] = torch.device("cuda")
        else:
            models["device"] = torch.device("cpu")
            if device_preference == "cuda": logging.warning("CUDA requested but not available. Using CPU.")
        logging.info(f"Using device: {models['device']}")
    except Exception as device_err:
        logging.error(f"Error detecting torch device: {device_err}. Defaulting to CPU.")
        models["device"] = torch.device("cpu")
    try:
        spacy_model_name = CONFIG['spacy_model']
        disabled_spacy_components = ['ner'] # Keep NER disabled unless needed
        logging.info(f"Loading spaCy model: {spacy_model_name} (disabling: {disabled_spacy_components})...")
        try: models["nlp"] = spacy.load(spacy_model_name, disable=disabled_spacy_components)
        except OSError:
            logging.warning(f"spaCy model '{spacy_model_name}' not found. Attempting download...")
            try: spacy.cli.download(spacy_model_name); models["nlp"] = spacy.load(spacy_model_name, disable=disabled_spacy_components); logging.info(f"spaCy model downloaded and loaded.")
            except Exception as download_err: logging.error(f"Failed download/load spaCy: {download_err}", exc_info=True); return None
        # Set a high max_length, ensure it's done *after* loading
        try: models["nlp"].max_length = 2000000 # Increased limit for potentially long inputs before splitting
        except Exception as max_len_err: logging.error(f"Failed to set spacy max_length: {max_len_err}")

        logging.info("spaCy model loaded.")
        paraphraser_model_name = CONFIG['paraphraser_model']
        logging.info(f"Loading paraphraser: {paraphraser_model_name}...")
        models["tokenizer"] = AutoTokenizer.from_pretrained(paraphraser_model_name)
        models["model"] = AutoModelForSeq2SeqLM.from_pretrained(paraphraser_model_name).to(models["device"])
        models["model"].eval()
        logging.info("Paraphraser model loaded.")
        logging.info("--- Required models (spaCy, T5) loaded successfully ---")
        return models
    except Exception as e:
        logging.exception("Fatal error during model loading:")
        return None

# --- END OF PART 2 ---

# --- START OF PART 3 ---

# ==============================================================================
# 9. CORE PROCESSING FUNCTION (Reverted to simpler flow)
# ==============================================================================
# (This function remains unchanged from the input, but will use the modified splitter)
def process_text(
    input_text: str,
    models: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Orchestrates the text processing pipeline (No citation handling)."""
    req_id = f"{time.time():.0f}-{os.getpid()}"
    # Basic start log
    logging.info(f"[{req_id}] === START TEXT PROCESSING ===")

    # --- Input Validation ---
    if not input_text or not input_text.strip():
        logging.error(f"[{req_id}] PROCESS_TEXT: Input text is empty.")
        return {'status': 'error', 'message': 'Input text cannot be empty.'}
    word_count = len(input_text.split())
    max_words = config.get("input_max_words", 1000)
    if word_count > max_words:
        logging.error(f"[{req_id}] PROCESS_TEXT: Input text exceeds word limit ({word_count}/{max_words}).")
        return {'status': 'error', 'message': f"Error: Text exceeds word limit ({max_words} words)."}

    # Log Original Input (INFO Level)
    logging.info(f"[{req_id}] --- Original Input Text ---\n{input_text}\n" + "-"*26)

    overall_start_time = time.time()
    nlp_model = models["nlp"]

    try:
        # --- Step 1: Pre-processing ---
        # Simplified pre-processing
        processed_text_step0 = input_text.replace('。', '.')
        processed_text_step0 = re.sub(r'\.(?=[a-zA-Z0-9\[\(])', '. ', processed_text_step0)
        # REMOVED: Citation extraction loop

        # --- Step 2: Placeholder Replacement REMOVED ---
        # Text going into T5 step is processed_text_step0

        # --- Step 3: Paraphrasing Sentences ---
        step_start_time = time.time()
        logging.info(f"[{req_id}] --- Step 3: Paraphrasing Sentences ---")
        try:
            # Ensure spacy max_length is sufficient for the text
            required_length_t5 = len(processed_text_step0) + 100 # Add buffer
            if nlp_model.max_length < required_length_t5:
                 new_max_length_t5 = max(required_length_t5, nlp_model.max_length) # Use existing max_length if already larger
                 if new_max_length_t5 > nlp_model.max_length:
                     nlp_model.max_length = new_max_length_t5
                     logging.warning(f"[{req_id}] Increased spacy max_length to {nlp_model.max_length} for T5 input tokenization.")
            doc = nlp_model(processed_text_step0)
            sentences = [s.text.strip() for s in doc.sents if s.text and s.text.strip()]
            if not sentences: raise ValueError("No valid sentences found after tokenization for T5.")
        except Exception as nlp_err:
            logging.error(f"[{req_id}] Sentence tokenization for T5 failed: {nlp_err}", exc_info=True)
            return {'status': 'error', 'message': f'Sentence tokenization for T5 failed: {nlp_err}'}

        paraphrased_sentences = []; sentences_failed = 0
        num_beams_param = config.get("paraphrase_num_beams", 5)
        # Log summary info only
        logging.info(f"[{req_id}] Paraphrasing {len(sentences)} sentences (num_beams={num_beams_param})...")
        max_sentence_tokens = config.get("input_max_sentence_tokens", 480)

        for i, sentence in enumerate(sentences):
            if not sentence or not sentence.strip():
                 paraphrased_sentences.append("")
                 continue

            # Length check logic (keep this)
            token_length_ok = True
            try:
                 check_text = f"paraphrase: {sentence}"
                 # Use max_length slightly larger for the check to be sure
                 sentence_tokens = models["tokenizer"].encode(check_text, max_length=max_sentence_tokens + 10, truncation=False)
                 if len(sentence_tokens) > max_sentence_tokens:
                      logging.warning(f"[{req_id}] Sent #{i} exceeds token limit ({len(sentence_tokens)}/{max_sentence_tokens}). Using original.");
                      token_length_ok = False
            except Exception as token_err:
                 logging.warning(f"[{req_id}] Tokenization check failed sent {i}: {token_err}")
                 token_length_ok = False

            if token_length_ok:
                 paraphrased = paraphrase_sentence(
                      sentence, models["model"], models["tokenizer"],
                      num_beams_param, config["paraphrase_max_length"]
                 )
                 if "[Paraphrasing Error:" in paraphrased or "[Encoding Error]" in paraphrased:
                      logging.error(f"[{req_id}] Paraphrasing failed sent {i}: {paraphrased}. Using original.")
                      paraphrased_sentences.append(sentence) # Use original sentence on error
                      sentences_failed += 1
                 else:
                      paraphrased_sentences.append(paraphrased) # Use paraphrased version
            else:
                 paraphrased_sentences.append(sentence) # Use original if too long/token error
                 sentences_failed += 1

        # Log summary info only
        if sentences_failed > 0:
            logging.warning(f"[{req_id}] Paraphrasing completed. Used original for {sentences_failed} sentences due to length/errors.")
        t5_paraphrased_text = " ".join(filter(None, paraphrased_sentences))

        # Log Full T5 Output (INFO Level)
        logging.info(f"[{req_id}] --- Full T5 Output ---\n{t5_paraphrased_text}\n" + "-"*20)
        # REMOVED: Step 3b (Citation Re-insertion)

        # --- Step 4: Apply Sentence Splitting ---
        step_start_time = time.time()
        logging.info(f"[{req_id}] --- Step 4: Applying Sentence Splitting ---")
        # Pass the direct T5 output to the modified splitter
        text_after_splitting = apply_sentence_splitting(t5_paraphrased_text, nlp_model)
        # Log Full Splitter Output (INFO Level)
        logging.info(f"[{req_id}] --- Full Splitter Output ---\n{text_after_splitting}\n" + "-"*26)

        # --- Step 5: Optional Filter Trailing Phrase ---
        step_start_time = time.time()
        # Keep this step, but logging can be minimal if not needed
        # logging.info(f"[{req_id}] --- Step 5: Filtering Trailing Phrase ---")
        unwanted_phrase="Can you provide some examples?"; text_to_refine = text_after_splitting
        if unwanted_phrase in text_after_splitting:
              text_lower = text_after_splitting.lower()
              phrase_lower = unwanted_phrase.lower()
              try:
                  last_occurrence_index = text_lower.rindex(phrase_lower)
                  text_after_occurrence = text_after_splitting[last_occurrence_index + len(phrase_lower):].strip()
                  # Simpler check: remove only if it's right at the end (allowing punctuation)
                  if re.fullmatch(r'[.?!]*\s*', text_after_occurrence): # Allow trailing whitespace too
                       logging.info(f"[{req_id}] Filtering detected unwanted trailing phrase.")
                       text_to_refine = text_after_splitting[:last_occurrence_index].rstrip()
                  # else:
                  #     logging.info(f"[{req_id}] Trailing phrase found, but not at the very end.")
              except ValueError: pass
              except Exception as filter_err: logging.warning(f"[{req_id}] Error during filtering check: {filter_err}")
        # else: logging.info(f"[{req_id}] Trailing phrase not present.")

        # --- Step 6: Refine with OpenAI ---
        step_start_time = time.time()
        # Logging handled within refine_with_openai
        # Call refine_with_openai WITHOUT citation preservation
        refined_text = refine_with_openai(text_to_refine)
        # Log Full GPT Output (INFO Level)
        logging.info(f"[{req_id}] --- Full GPT Refinement Output ---\n{refined_text}\n" + "-"*31)


        # --- Step 7: Re-insert Original Freeze Terms REMOVED ---
        final_text = refined_text

        # --- Step 8: Final Metrics Calculation ---
        # Keep metrics calculation, but logging can be minimal
        # logging.info(f"[{req_id}] --- Step 8: Calculating Final Metrics ---")
        paraphrased_fk_score,paraphrased_burstiness,paraphrased_ttr,paraphrased_word_count,paraphrased_mod_percentage = 0.0,0.0,0.0,0,0.0
        if final_text and final_text.strip():
            try: paraphrased_fk_score = flesch_kincaid_grade(final_text)
            except Exception as fk_err: logging.error(f"[{req_id}] Error calculating FK score: {fk_err}")
            paraphrased_burstiness = calculate_burstiness(final_text)
            paraphrased_ttr = calculate_ttr(final_text)
            paraphrased_word_count = len(final_text.split())
            paraphrased_mod_percentage = calculate_modification_percentage(input_text, final_text)
            # logging.info(f"[{req_id}] Final metrics calculated.")
        else: logging.warning(f"[{req_id}] Final text empty, skipping metrics.")

        # --- Step 9: Final Freeze Term Preservation Check REMOVED ---

        # --- Step 10: Structure Output Data ---
        # logging.info(f"[{req_id}] --- Step 10: Structuring Final Output ---")
        output_data = {
            "status": "success", "final_text": final_text,
            "final_metrics": {
                 "flesch_kincaid_grade": round(paraphrased_fk_score, 1), "burstiness_score": round(paraphrased_burstiness, 4),
                 "type_token_ratio": round(paraphrased_ttr, 4), "modification_percentage": round(paraphrased_mod_percentage, 2),
                 "word_count": paraphrased_word_count
            }}
        overall_end_time = time.time()
        # Log summary request time
        logging.info(f"[{req_id}] Request Processing Summary - Total Duration: {overall_end_time - overall_start_time:.2f} seconds")
        logging.info(f"[{req_id}] === END TEXT PROCESSING ===")
        return output_data

    except Exception as e:
        logging.exception(f"[{req_id}] An unexpected error occurred during core text processing:")
        overall_end_time = time.time()
        logging.error(f"[{req_id}] Processing failed after {overall_end_time - overall_start_time:.2f} seconds.")
        logging.info(f"[{req_id}] === END TEXT PROCESSING (ERROR) ===")
        return {'status': 'error', 'message': f"An unexpected processing error occurred: {str(e)}"}


# ==============================================================================
# 10. FLASK APP INITIALIZATION
# ==============================================================================
# (This section remains unchanged)
app = Flask(__name__)
CORS(app) # Keep CORS simple for local

# ==============================================================================
# 11. GLOBAL MODEL LOADING (Executed once when Flask app/worker starts)
# ==============================================================================
# (This section remains unchanged)
logging.info("="*15 + " INITIALIZING FLASK APP & LOADING MODELS " + "="*15)
loaded_models_global = load_models(device_preference=CONFIG.get("default_device", "auto"))
if not loaded_models_global:
    logging.critical("! FATAL: Models failed to load. !"); sys.exit("Model Loading Failed")
else:
    logging.info("*** Global models loaded successfully. Flask app ready. ***")
    logging.info(f"Using device: {loaded_models_global.get('device', 'UNKNOWN')}")
    # Log default spacy length on startup
    if 'nlp' in loaded_models_global and hasattr(loaded_models_global['nlp'], 'max_length'):
        logging.info(f"Initial spaCy max_length: {loaded_models_global['nlp'].max_length}")

# ==============================================================================
# 12. FLASK API ENDPOINTS (Reverted to simpler flow)
# ==============================================================================
# (This section remains unchanged)
@app.route('/', methods=['GET', 'POST'])
def index():
    method = request.method; request_id = f"{time.time():.0f}-{os.getpid()}"
    logging.info(f"REQ ID {request_id}: Root '/' received {method} from {request.remote_addr}")
    return jsonify({"status": "ok", "message": f"TextPulse backend alive. Received {method}."}), 200

@app.route('/humanize', methods=['POST'])
def humanize_endpoint():
    """API endpoint to process text (No citation handling)."""
    request_id = f"{time.time():.0f}-{os.getpid()}"
    start_request_time = time.time()
    logging.info(f"REQ ID {request_id}: START /humanize request from {request.remote_addr}")

    if not loaded_models_global:
        logging.error(f"REQ ID {request_id}: Models not loaded.")
        return jsonify({"status": "error", "message": "Service Unavailable: Backend models failed to load."}), 503

    if not request.is_json:
        logging.warning(f"REQ ID {request_id}: Request Content-Type is not application/json.")
        return jsonify({"status": "error", "message": "Invalid request format: Content-Type must be application/json"}), 415

    data = request.get_json()
    if not data:
        logging.warning(f"REQ ID {request_id}: Received empty JSON payload.")
        return jsonify({"status": "error", "message": "No input data received"}), 400

    input_text = data.get('text')

    if not input_text:
        logging.warning(f"REQ ID {request_id}: Request missing 'text' field.")
        return jsonify({"status": "error", "message": "Input 'text' is required"}), 400

    logging.info(f"REQ ID {request_id}: Calling process_text function...")
    try:
        # Call reverted process_text (which now uses the modified splitter)
        result_dict = process_text(
            input_text=input_text,
            models=loaded_models_global,
            config=CONFIG
        )

        if result_dict.get("status") == "success":
            response_data = {"status": "success", "final_text": result_dict.get("final_text", ""), "final_metrics": result_dict.get("final_metrics", {})}
            status_code = 200
            # Minimal success log
            # logging.info(f"REQ ID {request_id}: Processing successful.")
        else:
            response_data = {"status": "error", "message": result_dict.get('message', 'Unknown processing error')}
            msg_lower = response_data.get('message', '').lower()
            if "word limit" in msg_lower or "token limit" in msg_lower or "no valid sentences" in msg_lower or "tokenization failed" in msg_lower: status_code = 400 # Bad Request
            elif "service unavailable" in msg_lower: status_code = 503 # Service Unavailable
            else: status_code = 500 # Internal Server Error
            logging.error(f"REQ ID {request_id}: Processing failed. Status: {status_code}, Message: {response_data['message']}")

        end_request_time = time.time()
        # Log summary time and status code only
        logging.info(f"REQ ID {request_id}: END /humanize request. Total time: {end_request_time - start_request_time:.2f} seconds. Status Code: {status_code}")
        return jsonify(response_data), status_code

    except Exception as e:
        logging.exception(f"REQ ID {request_id}: UNEXPECTED error in /humanize endpoint:")
        end_request_time = time.time()
        logging.info(f"REQ ID {request_id}: END /humanize request (Exception). Total time: {end_request_time - start_request_time:.2f} seconds.")
        return jsonify({"status": "error", "message": f"An internal server error occurred."}), 500


# ==============================================================================
# 13. MAIN EXECUTION BLOCK (For Running the Server Directly)
# ==============================================================================
# (This section remains unchanged)
if __name__ == '__main__':
    # Keep settings suitable for local testing
    print("\n" + "="*60); print("--- TextPulse Flask Server (v17.1 - Improved Splitter) ---"); print("="*60) # Updated title
    print("\n >> Running in direct execution mode (for development/testing) <<\n")
    print(" --- Deployment Instructions ---")
    print(" For production (e.g., on DigitalOcean with Gunicorn):")
    print(" 1. Set production T5 model in CONFIG, logging level to INFO, DEBUG_SPLITTING=False.")
    print(" 2. Configure CORS origins properly.")
    print(" 3. Run using Gunicorn/Nginx/Systemd.")
    print("    Example: gunicorn --workers 2 --threads 4 --bind 0.0.0.0:5001 --timeout 180 main:app") # Adjust main:app if filename changes, bind to 0.0.0.0
    print(" -----------------------------")

    flask_debug_mode = True # Keep True for local testing
    flask_port = 5001

    print(f"\nStarting Flask development server on http://127.0.0.1:{flask_port}/")
    print(f" --> DEBUG MODE: {'ON' if flask_debug_mode else 'OFF'} <--")
    print(f" --> LOGGING LEVEL: {logging.getLevelName(logging.getLogger().getEffectiveLevel())} <--") # Show logging level
    print(" (Use Ctrl+C to stop)"); print("="*60 + "\n")

    app.run(debug=flask_debug_mode, host='127.0.0.1', port=flask_port, threaded=True)


# ==============================================================================
# 14. FINAL SCRIPT DOCSTRING
# ==============================================================================
# (This section remains unchanged)
"""
End of TextPulse.ai Flask Backend Script.
Provides the /humanize API endpoint (No citation handling).
Processing Flow: Input -> Preprocess -> T5 -> Splitter -> Filter -> OpenAI.
Minimal INFO logging for key intermediate outputs.
Configured for local testing.
**MODIFIED: Sentence splitter logic improved for safety.**
"""

# --- END OF PART 3 ---