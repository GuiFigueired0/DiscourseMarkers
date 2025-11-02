import os
import json
import sys
import spacy
from datasets import load_dataset

# --- Configuration Constants ---

# 1. Define configurations for each supported language
SUPPORTED_LANGUAGES = {
    "en": {
        "wiki_config": "20231101.en",
        "spacy_model": "en_core_web_sm",
        "dm_file": os.path.join("dm", "en.txt")
    },
    "it": {
        "wiki_config": "20231101.it",
        "spacy_model": "it_core_news_sm",
        "dm_file": os.path.join("dm", "it.txt")
    },
    "pt": {
        "wiki_config": "20231101.pt",
        "spacy_model": "pt_core_news_sm",
        "dm_file": os.path.join("dm", "pt.txt")
    }
}

# 2. Define base output directory
BASE_OUTPUT_DIR = "results"
PROGRESS_FILE_NAME = "progress.json"
OUTPUT_FILE_NAME = "mined_data.jsonl"


# --- Helper Functions (No changes needed) ---

def load_discourse_markers(filepath):
    """
    Loads DMs from a text file.
    Returns a list of DMs, sorted from longest to shortest.
    This is crucial to avoid partial matches (e.g., matching "as" instead of "as a result").
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Remove blank lines and extra whitespace
            markers = [line.strip() for line in f if line.strip()]
        
        # Sort by length, descending
        markers.sort(key=len, reverse=True)
        print(f"Loaded {len(markers)} discourse markers from '{filepath}'.")
        return markers
    except FileNotFoundError:
        print(f"Error: Marker file '{filepath}' not found.")
        print("Please make sure the file exists inside the 'dm/' folder.")
        sys.exit(1) # Exit the script if DMs can't be loaded

def load_progress(filepath):
    """Loads the index of the last processed article."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return data.get("articles_processed", 0)
        except json.JSONDecodeError:
            print(f"Warning: Progress file '{filepath}' is corrupted. Starting from 0.")
            return 0
    return 0

def save_progress(filepath, index):
    """Saves the index of the last processed article."""
    with open(filepath, 'w') as f:
        json.dump({"articles_processed": index}, f, indent=2)

def save_data(filepath, data_entry):
    """Appends a new (s1, s2, dm) pair to the JSONL output file."""
    # Add 'ensure_ascii=False' to write UTF-8 characters directly
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data_entry, ensure_ascii=False) + "\n")

def is_valid_sentence(text, min_words=3, max_words=32):
    """
    Simple validation function for a sentence.
    """
    word_count = len(text.split())
    return min_words <= word_count <= max_words

# --- Main Mining Function (Adapted) ---

def main():
    
    # --- 1. Get Language Input ---
    print("Please select a language to process:")
    print(f"Supported languages: {', '.join(SUPPORTED_LANGUAGES.keys())}")
    lang_code = input("Enter language code (e.g., 'en'): ").lower().strip()

    if lang_code not in SUPPORTED_LANGUAGES:
        print(f"Error: Unsupported language '{lang_code}'. Exiting.")
        sys.exit(1)
    
    print(f"\n--- Starting mining process for: {lang_code.upper()} ---")

    # --- 2. Get Language-Specific Configuration ---
    config = SUPPORTED_LANGUAGES[lang_code]
    spacy_model = config["spacy_model"]
    dm_file_path = config["dm_file"]
    wiki_config = config["wiki_config"]
    
    # --- 3. Create Dynamic Output Paths ---
    lang_output_dir = os.path.join(BASE_OUTPUT_DIR, lang_code)
    os.makedirs(lang_output_dir, exist_ok=True) # Create 'results/en/', 'results/it/', etc.
    
    progress_file_path = os.path.join(lang_output_dir, PROGRESS_FILE_NAME)
    output_file_path = os.path.join(lang_output_dir, OUTPUT_FILE_NAME)

    print(f"Results will be saved in: {lang_output_dir}")

    # --- 4. Load Resources (Dynamic) ---
    discourse_markers = load_discourse_markers(dm_file_path)
    
    print(f"Loading spaCy model '{spacy_model}'...")
    try:
        nlp = spacy.load(spacy_model, disable=["parser", "tagger", "ner", "lemmatizer"])
    except IOError:
        print(f"\n--- ERROR: spaCy model '{spacy_model}' not found. ---")
        print(f"Please install it by running: python -m spacy download {spacy_model}")
        sys.exit(1)
        
    nlp.add_pipe("sentencizer")
    nlp.max_length = 2000000 
    print("spaCy model loaded successfully.")

    # --- 5. Load Progress and Dataset (Dynamic) ---
    start_point = load_progress(progress_file_path)
    print(f"Resuming from article index: {start_point}")
    
    # Load the dataset in streaming mode using the correct lang config
    ds = load_dataset("wikimedia/wikipedia", wiki_config, streaming=True)
    stream = iter(ds['train'])

    # --- 6. Skip Already Processed Articles ---
    if start_point > 0:
        print(f"Skipping {start_point} already-processed articles...")
        try:
            for _ in range(start_point):
                next(stream)
        except StopIteration:
            print("Error: Start point is beyond the dataset length.")
            return
        print("Skipping complete.")

    print("Starting mining of new articles...")
    current_index = start_point
    pairs_found_session = 0

    try:
        # --- 7. Main Processing Loop (Unchanged logic) ---
        while True:
            article = next(stream)
            text = article['text']
            
            doc = nlp(text)
            sentences = list(doc.sents)
            
            if len(sentences) < 2:
                current_index += 1
                continue

            for i in range(1, len(sentences)):
                s1_span = sentences[i-1]
                s2_span = sentences[i]
                
                s2_text_lower_stripped = s2_span.text.lower().strip()

                for dm in discourse_markers:
                    if s2_text_lower_stripped.startswith(dm):
                        
                        rest_of_s2 = s2_text_lower_stripped[len(dm):].strip()
                        if rest_of_s2.startswith(','):
                            
                            s1_text = s1_span.text.strip()
                            s2_text = s2_span.text.strip()

                            if s1_text.endswith(('.', ';', '!', '?')) and \
                               is_valid_sentence(s1_text) and \
                               is_valid_sentence(s2_text,len(dm.split()) + 3):
                                
                                data_entry = {
                                    "s1": s1_text,
                                    "s2": s2_text,
                                    "dm_label": dm,
                                    "article_id": article['id']
                                }
                                # Save to the dynamic output path
                                save_data(output_file_path, data_entry)
                                pairs_found_session += 1
                            
                            break 
            
            current_index += 1
            
            if current_index % 100 == 0:
                # Save to the dynamic progress path
                save_progress(progress_file_path, current_index)
                print(f"Processed {current_index} articles. {pairs_found_session} pairs found this session.")

    except StopIteration:
        print("End of dataset! Processing complete.")
    except KeyboardInterrupt:
        print("\nInterrupt detected! Saving progress...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # --- 8. Save Final Progress (Dynamic) ---
        save_progress(progress_file_path, current_index)
        print(f"Process stopped. Progress saved.")
        print(f"Total articles processed: {current_index}")
        print(f"Total pairs found this session: {pairs_found_session}")

if __name__ == "__main__":
    main()