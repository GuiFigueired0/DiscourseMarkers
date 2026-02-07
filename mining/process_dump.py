import os
import json
import csv

DM_MAPPING = {
    # ----- ENGLISH -----
    # == Contrastive Discourse Markers (CDMs) ==
    # Show opposition, contrast, concession, or correction
    'although': 'CDM',
    'but': 'CDM',
    'by comparison': 'CDM',
    'by contrast': 'CDM',
    'conversely': 'CDM',
    'however': 'CDM',
    'in contrast': 'CDM',
    'instead': 'CDM',
    'nevertheless': 'CDM',
    'nonetheless': 'CDM',
    'on the contrary': 'CDM',
    'on the other hand': 'CDM',
    'otherwise': 'CDM',
    'rather': 'CDM',
    'regardless': 'CDM',
    'still': 'CDM',
    'though': 'CDM',
    'yet': 'CDM',

    # == Elaborative Discourse Markers (EDMs) ==
    # Add info, specify, rephrase, give examples, or add speaker stance
    'absolutely': 'EDM',
    'actually': 'EDM',
    'additionally': 'EDM',
    'admittedly': 'EDM',
    'again': 'EDM',
    'also': 'EDM',
    'alternately': 'EDM',
    'alternatively': 'EDM',
    'altogether': 'EDM',
    'amazingly': 'EDM',
    'and': 'EDM',
    'anyway': 'EDM',
    'apparently': 'EDM',
    'arguably': 'EDM',
    'basically': 'EDM',
    'besides': 'EDM',
    'certainly': 'EDM',
    'clearly': 'EDM',
    'coincidentally': 'EDM',
    'collectively': 'EDM',
    'curiously': 'EDM',
    'elsewhere': 'EDM',
    'especially': 'EDM',
    'essentially': 'EDM',
    'evidently': 'EDM',
    'for example': 'EDM',
    'for instance': 'EDM',
    'fortunately': 'EDM',
    'frankly': 'EDM',
    'further': 'EDM',
    'furthermore': 'EDM',
    'generally': 'EDM',
    'happily': 'EDM',
    'here': 'EDM',
    'honestly': 'EDM',
    'hopefully': 'EDM',
    'ideally': 'EDM',
    'importantly': 'EDM',
    'in fact': 'EDM',
    'in other words': 'EDM',
    'in particular': 'EDM',
    'in short': 'EDM',
    'in sum': 'EDM',
    'incidentally': 'EDM',
    'indeed': 'EDM',
    'interestingly': 'EDM',
    'ironically': 'EDM',
    'likewise': 'EDM',
    'locally': 'EDM',
    'luckily': 'EDM',
    'maybe': 'EDM',
    'meaning': 'EDM',
    'moreover': 'EDM',
    'mostly': 'EDM',
    'namely': 'EDM',
    'nationally': 'EDM',
    'naturally': 'EDM',
    'notably': 'EDM',
    'obviously': 'EDM',
    'oddly': 'EDM',
    'only': 'EDM',
    'optionally': 'EDM',
    'or': 'EDM',
    'overall': 'EDM',
    'particularly': 'EDM',
    'perhaps': 'EDM',
    'personally': 'EDM',
    'plus': 'EDM',
    'preferably': 'EDM',
    'presumably': 'EDM',
    'probably': 'EDM',
    'realistically': 'EDM',
    'really': 'EDM',
    'remarkably': 'EDM',
    'sadly': 'EDM',
    'separately': 'EDM',
    'seriously': 'EDM',
    'significantly': 'EDM',
    'similarly': 'EDM',
    'specifically': 'EDM',
    'strangely': 'EDM',
    'supposedly': 'EDM',
    'surely': 'EDM',
    'surprisingly': 'EDM',
    'technically': 'EDM',
    'thankfully': 'EDM',
    'theoretically': 'EDM',
    'together': 'EDM',
    'truly': 'EDM',
    'truthfully': 'EDM',
    'undoubtedly': 'EDM',
    'unfortunately': 'EDM',
    'unsurprisingly': 'EDM',
    'well': 'EDM',

    # == Implicative Discourse Markers (IDMs) ==
    # Show result, consequence, or inference
    'accordingly': 'IDM',
    'as a result': 'IDM',
    'because of that': 'IDM',
    'because of this': 'IDM',
    'by doing this': 'IDM',
    'consequently': 'IDM',
    'hence': 'IDM',
    'in turn': 'IDM',
    'inevitably': 'IDM',
    'so': 'IDM',
    'thereby': 'IDM',
    'therefore': 'IDM',
    'thus': 'IDM',

    # == Temporal Discourse Markers (TDMs) ==
    # Show time or sequence
    'afterward': 'TDM',
    'already': 'TDM',
    'by then': 'TDM',
    'currently': 'TDM',
    'eventually': 'TDM',
    'finally': 'TDM',
    'first': 'TDM',
    'firstly': 'TDM',
    'frequently': 'TDM',
    'gradually': 'TDM',
    'historically': 'TDM',
    'immediately': 'TDM',
    'in the end': 'TDM',
    'in the meantime': 'TDM',
    'increasingly': 'TDM',
    'initially': 'TDM',
    'lastly': 'TDM',
    'lately': 'TDM',
    'later': 'TDM',
    'meantime': 'TDM',
    'meanwhile': 'TDM',
    'next': 'TDM',
    'normally': 'TDM',
    'now': 'TDM',
    'occasionally': 'TDM',
    'often': 'TDM',
    'once': 'TDM',
    'originally': 'TDM',
    'presently': 'TDM',
    'previously': 'TDM',
    'recently': 'TDM',
    'second': 'TDM',
    'secondly': 'TDM',
    'simultaneously': 'TDM',
    'slowly': 'TDM',
    'sometimes': 'TDM',
    'soon': 'TDM',
    'subsequently': 'TDM',
    'suddenly': 'TDM',
    'then': 'TDM',
    'thereafter': 'TDM',
    'third': 'TDM',
    'thirdly': 'TDM',
    'traditionally': 'TDM',
    'typically': 'TDM',
    'ultimately': 'TDM',
    'usually': 'TDM',


    # ----- ITALIAN -----
    # CDM: Contrasto massiccio
    'tuttavia': 'CDM', 'comunque': 'CDM', 'ma': 'CDM', 'in realtà': 'CDM',
    'in ogni caso': 'CDM', 'invece': 'CDM', 'però': 'CDM', 'anzi': 'CDM',
    'eppure': 'CDM', 'a ogni modo': 'CDM', 'del resto': 'CDM', 'a dire il vero': 'CDM',

    # EDM: Spiegazione ed Esempio
    'infatti': 'EDM', 'ad esempio': 'EDM', 'per esempio': 'EDM', 'in effetti': 'EDM',
    'in altre parole': 'EDM', 'naturalmente': 'EDM', 'ovviamente': 'EDM',
    'in sostanza': 'EDM', 'a questo proposito': 'EDM', 'in sintesi': 'EDM',
    'insomma': 'EDM', 'cioè': 'EDM', 'addirittura': 'EDM', 'sì': 'EDM',  # 'sì' qui è conferma
    'forse': 'EDM', 'effettivamente': 'EDM', 'certo': 'EDM', 'ovvero': 'EDM',
    'voglio dire': 'EDM', 'vale a dire': 'EDM', 'praticamente': 'EDM', 'ossia': 'EDM',
    'appunto': 'EDM', 'precisamente': 'EDM',

    # SEQ (IDM+TDM): Flusso Logico
    'quindi': 'SEQ', 'poi': 'SEQ', 'ora': 'SEQ', 'dunque': 'SEQ', 'allora': 'SEQ',
    'in definitiva': 'SEQ', 'in conclusione': 'SEQ', 'dopo di che': 'SEQ',
    'ebbene': 'SEQ', 'detto questo': 'SEQ', 'ecco': 'SEQ', 'perché': 'SEQ',
    'tutto qui': 'SEQ', 'bene': 'SEQ',
                                
    
    # ----- PORTUGUESE -----
    # CDM: Contraste / Oposição / Concessão
    'no entanto': 'CDM', 'entretanto': 'CDM', 'contudo': 'CDM', 'por outro lado': 'CDM',
    'mas': 'CDM', 'apesar disso': 'CDM', 'todavia': 'CDM', 'ainda assim': 'CDM',
    'embora': 'CDM', 'ou': 'CDM', 'enquanto': 'CDM', 'apesar de': 'CDM',
    'ou então': 'CDM', 'ainda que': 'CDM', 'apenas': 'CDM', 'somente': 'CDM',

    # EDM: Adição / Exemplificação / Reformulação / Foco (O "além disso" vive aqui)
    'além disso': 'EDM', 'por exemplo': 'EDM', 'e': 'EDM', 'da mesma forma': 'EDM',
    'ou seja': 'EDM', 'em particular': 'EDM', 'em outras palavras': 'EDM',
    'também': 'EDM', 'isto é': 'EDM', 'ainda': 'EDM', 'especificamente': 'EDM',
    'inclusive': 'EDM', 'essencialmente': 'EDM', 'analogamente': 'EDM',
    'principalmente': 'EDM', 'assim como': 'EDM', 'bem como': 'EDM',
    'mas também': 'EDM', 'além de': 'EDM', 'até mesmo': 'EDM', 'tais como': 'EDM',

    # IDM: Conclusão / Consequência / Causa / Contexto Lógico
    'assim': 'IDM', 'portanto': 'IDM', 'então': 'IDM', 'dessa forma': 'IDM',
    'desta forma': 'IDM', 'logo': 'IDM', 'consequentemente': 'IDM',
    'neste caso': 'IDM', 'desse modo': 'IDM', 'nesse caso': 'IDM',
    'deste modo': 'IDM', 'nesse contexto': 'IDM', 'dessa maneira': 'IDM',
    'neste contexto': 'IDM', 'desta maneira': 'IDM', 'pois': 'IDM',
    'que': 'IDM', 'uma vez que': 'IDM', 'já que': 'IDM', 'visto que': 'IDM'
}

# Folders
RESULTS_DIR = "dump"
OUTPUT_DIR = "../data"
LANGS = ["en", "pt", "it"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for lang in LANGS:
    input_path = os.path.join(RESULTS_DIR, lang, "mined_data.jsonl")
    output_path = os.path.join(OUTPUT_DIR, f"{lang}.csv")

    if not os.path.exists(input_path):
        print(f"⚠️ File not found: {input_path}")
        continue

    rows = []
    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            data = json.loads(line)
            s1 = data.get("s1", "").strip()
            s2 = data.get("s2", "").strip()
            dm = data.get("dm_label", "").strip()
            article_id = data.get("article_id", "")

            # Remove "dm," from the beginning of s2
            if s2.lower().startswith(dm.lower() + ","):
                s2 = s2[len(dm) + 1:].strip()
            elif s2.lower().startswith(dm.lower()):
                s2 = s2[len(dm):].strip()

            if s2:
                s2 = s2[0].upper() + s2[1:]

            rows.append({
                "s1": s1,
                "s2": s2,
                "dm": dm,
                "label": DM_MAPPING.get(dm, dm),
                "article_id": article_id
            })

    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["s1", "s2", "dm", "article_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ {output_path} created with {len(rows)} lines.")