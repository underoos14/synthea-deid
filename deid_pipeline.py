import re, torch, json, copy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from synthea_helper_codes.build_eval_dataset import LABEL_MAP, walk_json, clean_str

MODEL_NAME = "models/deid_finetuned"
CONF_THRESHOLD = 0.8

REGEX_PATTERNS = {
    "CONTACT": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", re.IGNORECASE),
    "DATE": re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", re.IGNORECASE),
    "WEB": re.compile(r"https?://[^\s]+|www\.[^\s]+|\b\d{1,3}(?:\.\d{1,3}){3}\b", re.IGNORECASE),
    "ID": re.compile(r"\b[a-fA-F0-9]{8,}\b|\b\d{3}-\d{2}-\d{4}\b", re.IGNORECASE),
    "LOCATION": re.compile(r"\b\d{5}(-\d{4})?\b"),
    "NAME": re.compile(r"\b[A-Z][a-z]+[0-9]{2,4}\b"),

}

# LOAD MODEL
def load_ner_model():
    print(f"📦 Loading fine-tuned model from {MODEL_NAME}")
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, ignore_mismatched_sizes=True)
    ner = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device)
    print(f"✅ Model ready on {'GPU' if device == 0 else 'CPU'}")
    return ner


# HELPERS
def standardize_label(label: str) -> str:
    label = label.upper()
    LABELS = {"NAME", "LOCATION", "DATE", "CONTACT", "ID", "WEB"}
    for key in LABELS:
        if key in label:
            return key
    return "OTHER"


NON_PHI_KEYS = ["resourceType", "status", "code", "coding", "display", "system", "text", "id", "div", 
                "status", "valueBoolean", "gender", "multipleBirthBoolean", "use", "clinicalStatus", "verificationStatus"]

def extract_label_from_keypath(keypath: str, value: str = "") -> str:
    keypath_lower = keypath.lower()

    # skip known non-PHI structural keys
    if any(k in keypath_lower.split(".")[-1] for k in NON_PHI_KEYS):
        return "OTHER"

    for k, v in LABEL_MAP.items():
        if k in keypath_lower:
            if k == "system" and not re.search(r"https?://", str(value)):
                continue
            return v
    return "OTHER"

def merge_entities(preds):
    """Merge contiguous tokens of same label (handles subwords, numbers, multi-word PHI)."""
    if not preds:
        return []

    merged = [preds[0].copy()]
    for e in preds[1:]:
        prev = merged[-1]
        if (
            e["entity_group"] == prev["entity_group"]
            and e["start"] <= prev["end"] + 1
        ):
            # Numeric or subword continuation merge 
            if e["word"].startswith("##") or (prev["word"].replace(" ", "").isdigit() and e["word"].replace(" ", "").isdigit()):
                prev["word"] = prev["word"] + e["word"].replace("##", "")
            else:
                # Regular word continuation (space-separated)
                prev["word"] = (prev["word"] + " " + e["word"].replace("##", "")).strip()

            # Update offsets and confidence
            prev["end"] = e["end"]
            prev["score"] = max(prev["score"], e["score"])
        else:
            merged.append({
                "word": e["word"].replace("##", ""),
                "entity_group": e["entity_group"],
                "start": e["start"],
                "end": e["end"],
                "score": e["score"],
            })

    return merged

# DEIDENTIFY LINE
def deidentify_line(keypath: str, value: str, ner):
    """Hybrid model + regex + keypath hints + multi-value handling."""
    if not isinstance(value, str) or not value.strip():
        return value, []

    preds = ner(value)
    preds = merge_entities(preds)

    redacted = value
    redacted_flag = False

    # Model-based redaction 
    for e in preds:
        label = standardize_label(e.get("entity_group", ""))
        score = e.get("score", 0)
        if score >= CONF_THRESHOLD:
            pattern = re.compile(re.escape(e["word"]), re.IGNORECASE)
            redacted = pattern.sub(f"[{label}]", redacted)
            redacted_flag = True
    
    regex_detections = []
    for label, pattern in REGEX_PATTERNS.items():
        if pattern.search(redacted):
            redacted = pattern.sub(f"[{label}]", redacted)
            regex_detections.append({
                "word": label,
                "entity_group": label,
                "score": 1.0,
                "source": "regex"
            })
    if regex_detections:
        preds.extend(regex_detections)
        redacted_flag = True

    # Multi-label confidence-based collapse (ignore regex) 
    valid_preds = [e for e in preds if e.get("source", "model") == "model"]
    if valid_preds:
        label_scores = {}
        for e in valid_preds:
            l = standardize_label(e["entity_group"])
            label_scores.setdefault(l, []).append(e.get("score", 0))
        best_label = max(label_scores, key=lambda l: sum(label_scores[l]) / len(label_scores[l]))
        redacted = f"[{best_label}]"
        redacted_flag = True

    # Keypath fallback 
    if not redacted_flag and redacted == value:
        hint = extract_label_from_keypath(keypath)
        if hint != "OTHER":
            redacted = f"[{hint}]"
    
    # Keypath hint override 
    if redacted == value or not redacted_flag:
        if re.search(r"name|maiden|given|family|prefix|suffix", keypath, re.I):
            redacted = "[NAME]"
        elif re.search(r"address|city|state|postal|geo", keypath, re.I):
            redacted = "[LOCATION]"
        elif re.search(r"birth|date", keypath, re.I):
            redacted = "[DATE]"
        elif re.search(r"phone|fax|email|telecom|contact", keypath, re.I):
            redacted = "[CONTACT]"
        elif re.search(r"id|identifier|license|account|passport", keypath, re.I):
            redacted = "[ID]"
        elif re.search(r"url|uri|web|internet|ip|image|photo", keypath, re.I):
            redacted = "[WEB]"

    return redacted, preds

# PARSE + DEIDENTIFY JSON
_INDEXED_SEG_RE = re.compile(r"^([^\[\]]+)(?:\[(\d+)\])?$")

def set_value_at_path(obj, path, new_value):
    """
    Safely traverse nested dict/list structure and set the given value.
    Supports both implicit (all list elements) and explicit indexed paths,
    e.g.:
        entry.resource.identifier.value        # all identifiers
        entry[0].resource.address.city         # only the first entry
        entry.resource.address[0].line         # specific list index
    """
    if not path:
        return

    segments = path.split(".")

    def parse_segment(seg):
        m = _INDEXED_SEG_RE.match(seg)
        if not m:
            return seg, None
        key, idx = m.group(1), m.group(2)
        return key, (int(idx) if idx is not None else None)

    parsed = [parse_segment(s) for s in segments]

    def _set(current, segs):
        if not segs:
            return
        key, idx = segs[0]

        # If we're at a list level
        if isinstance(current, list):
            if idx is not None:
                if 0 <= idx < len(current):
                    _set(current[idx], segs)
                return
            for item in current:
                _set(item, segs)
            return

        # If we're at a dict level
        if isinstance(current, dict) and key in current:
            next_obj = current[key]

            #If this is the last segment — perform the replacement
            if len(segs) == 1:
                if isinstance(next_obj, list):
                    # Replace lists of strings individually
                    if all(isinstance(x, str) for x in next_obj):
                        for i in range(len(next_obj)):
                            next_obj[i] = new_value
                    elif idx is not None and 0 <= idx < len(next_obj):
                        next_obj[idx] = new_value
                    else:
                        current[key] = new_value
                else:
                    current[key] = new_value
                return

            # Continue deeper
            if idx is not None:
                if isinstance(next_obj, list) and 0 <= idx < len(next_obj):
                    _set(next_obj[idx], segs[1:])
            else:
                _set(next_obj, segs[1:])

    _set(obj, parsed)

def deidentify_json(data, ner):
    flattened = []  # will hold tuples (orig_keypath, value)

    # build flattened list per resource (orig keypath has leading resourceType)
    for entry in data.get("entry", []):
        resource = entry.get("resource", {})
        if not resource:
            continue
        rtype = resource.get("resourceType", "Unknown")
        if rtype.lower() in [
            "observation", "diagnosticreport", "medicationrequest",
            "medicationadministration", "medicationstatement",
            "procedure", "condition", "immunization"
        ]:
            continue

        # walk_json returns keypaths that start with the resource type
        for kp, val in walk_json(resource, context_path=[rtype]):
            flattened.append((kp, val))

    # make a deep copy for safe in-place replacement
    redacted_json = copy.deepcopy(data)
    all_entities = []

    for orig_keypath, value in flattened:
        value_clean = str(value).strip()
        if not value_clean:
            continue
        # strict match only for last path component
        if any(orig_keypath.lower().split(".")[-1] == k.lower() for k in NON_PHI_KEYS):
            continue

        # Deidentify the value (model + regex + fallbacks)
        redacted_value, ents = deidentify_line(orig_keypath, value_clean, ner)

        # Convert resource-rooted keypath -> bundle-rooted path expected by setter 
        parts = orig_keypath.split(".", 1)  # split into [resourceType, rest...]
        if len(parts) == 2:
            _, rest = parts
            target_path = f"entry.resource.{rest}"
        else:
            # just target entry.resource
            target_path = "entry.resource"

        # Apply redaction into the deep-copied JSON
        set_value_at_path(redacted_json, target_path, redacted_value)

        # collect entities for table display 
        for e in ents:
            all_entities.append({
                "keypath": orig_keypath,
                "original": e.get("word", ""),
                "label": e.get("entity_group", ""),
                "confidence": round(e.get("score", e.get("confidence", 1.0)), 3)
            })

    return redacted_json, all_entities