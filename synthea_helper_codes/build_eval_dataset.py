import os, json, re
from tqdm import tqdm

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FOLDER = os.path.join(BASE_DIR, "data", "evaluation_set_2000")
OUTPUT_TEXT = os.path.join(BASE_DIR, "data", "synthetic_eval_text_2000.txt")
OUTPUT_ANNOT = os.path.join(BASE_DIR, "data", "synthetic_eval_annotations_2000.jsonl")

# Label mapping (HIPAA 7-group)
LABEL_MAP = {
    "name": "NAME",
    "mothersmaidenname": "NAME",
    "fathersname": "NAME",
    "address": "LOCATION",
    "city": "LOCATION",
    "state": "LOCATION",
    "postalcode": "LOCATION",
    "birth": "DATE",
    "date": "DATE",
    "telecom": "CONTACT",
    "phone": "CONTACT",
    "fax": "CONTACT",
    "email": "CONTACT",
    "identifier": "ID",
    "id": "ID",
    "organization": "ORGANIZATION",
    "practitioner": "ORGANIZATION",
    "provider": "ORGANIZATION",
    "fhir/structuredefinition": "WEB",       
    "standardhealthrecord.org": "WEB",       
    "system": "WEB",                         
    "contenttype": "WEB",                    
    "photo": "WEB",                          
    "image": "WEB",                          
    "biometric": "WEB",                      
    "fingerprint": "WEB",
    "voice": "WEB",
    "ipaddress": "WEB"
}

# Helper functions
def clean_str(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def extract_label_from_keypath(keypath: str, value: str = "") -> str:
    keypath_lower = keypath.lower()
    for k, v in LABEL_MAP.items():
        if k in keypath_lower:
            # For 'system', only tag as WEB if the value looks like a URL
            if k == "system" and not re.search(r"https?://", str(value)):
                continue
            return v
    return "OTHER"

def get_fhir_fragment(url: str) -> str:
    """Extract final path component from FHIR extension URL"""
    if not isinstance(url, str):
        return ""
    return url.rstrip("/").split("/")[-1]

# Recursive walker
def walk_json(obj, context_path=None):
    """Flatten nested FHIR dicts/lists into (context_path, value) tuples."""
    context_path = context_path or []
    results = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            # For extensions: include last URL fragment
            if k == "url" and isinstance(v, str) and v.startswith("http"):
                frag = get_fhir_fragment(v)
                results.extend(walk_json(v, context_path + [frag]))
                continue
            results.extend(walk_json(v, context_path + [k]))

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            results.extend(walk_json(item, context_path))

    else:
        # Primitive value (str, int, bool, etc.)
        if isinstance(obj, (str, int, float)) and str(obj).strip():
            path = ".".join(context_path)
            results.append((path, str(obj)))

    return results

# Main builder
def build_eval_dataset():
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    all_lines = []
    all_annots = []

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".json")]
    print(f"🩺 Building evaluation dataset from {len(files)} files...")

    for fname in tqdm(files, desc="Processing Synthea JSONs"):
        fpath = os.path.join(INPUT_FOLDER, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] {fname}: {e}")
            continue

        for entry in data.get("entry", []):
            resource = entry.get("resource", {})
            if not resource:
                continue

            rtype = resource.get("resourceType", "Unknown")
            if rtype.lower() in ["observation", "diagnosticreport", "medicationrequest",
                        "medicationadministration", "medicationstatement",
                        "procedure", "condition", "immunization"]:
                continue
            flattened = walk_json(resource, context_path=[rtype])

            for keypath, value in flattened:
                # Skip non-text PHI-irrelevant types
                if len(value) < 2 or re.fullmatch(r"true|false", value, re.I):
                    continue

                label = extract_label_from_keypath(keypath)
                if label == "OTHER":
                    continue

                # Construct contextual line
                text_line = f"{keypath}: {clean_str(value)}"
                start = text_line.find(value)
                end = start + len(value)
                span = {"start": start, "end": end, "label": label, "text": value}

                all_lines.append(text_line)
                all_annots.append({
                    "file": fname,
                    "text": text_line,
                    "spans": [span]
                })

    # Write outputs
    with open(OUTPUT_TEXT, "w", encoding="utf-8") as ftxt:
        ftxt.write("\n".join(all_lines))
    with open(OUTPUT_ANNOT, "w", encoding="utf-8") as fjsonl:
        for entry in all_annots:
            fjsonl.write(json.dumps(entry) + "\n")

    print(f"✅ Created {len(all_lines)} evaluation lines.")
    print(f"📝 Text file: {OUTPUT_TEXT}")
    print(f"🧾 Annotation file: {OUTPUT_ANNOT}")

if __name__ == "__main__":
    build_eval_dataset()