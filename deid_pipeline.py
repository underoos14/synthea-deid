import re, torch, json, copy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from synthea_helper_codes.build_eval_dataset import LABEL_MAP, walk_json

MODEL_NAME = "models/deid_finetuned"
CONF_THRESHOLD = 0.8


# Regex safeguard patterns tuned for Synthea dataset
REGEX_PATTERNS = [
    {
        "label": "NAME",  
        "pattern": re.compile(r"\b[A-Z][a-z]+[0-9]{2,}(?:\s[A-Z][a-z]+[0-9]{2,})?\b"),
        "score": 1.0
    },
    {
        "label": "LOCATION",
        "pattern": re.compile(r'\b[A-Z][a-z]+,\s[A-Z]{2}\b'),
        "score": 1.0
    },
    {
        "label": "CONTACT",
        "pattern": re.compile(
            r'((?:\(\d{3}\)\s\d{3}-\d{4}|\b\d{1,3}-\d{3}-\d{3}-\d{4}|\b\d{3}\.\d{3}\.\d{4}|\b\d{3}-\d{3}-\d{4})(\s?x\d{3,5})?)'
        ),
        "score": 1.0
    },
    {
        "label": "ID",
        "pattern": re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
        "score": 1.0
    },
    {
        "label": "ID",
        "pattern": re.compile(r'\b[A-Z]\d{8}\b'),
        "score": 1.0
    },
    {
        "label": "ID",
        "pattern": re.compile(r'\bX\d{8}X\b'),
        "score": 1.0
    },
    {
        "label": "LOCATION",
        "pattern": re.compile(r'\b\d{5}\b'),
        "score": 1.0
    },
    {
        "label": "ID",
        "pattern": re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', re.IGNORECASE),
        "score": 1.0
    }
]

# Label priority (higher wins)
LABEL_PRIORITY = {
    "NAME": 6,
    "CONTACT": 5,
    "LOCATION": 4,
    "DATE": 3,
    "WEB": 2,
    "ID": 1,
    "OTHER": 0
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


NON_PHI_KEYS = ["resourceType", "reference", "status", "code", "coding", "display", "system", "text", "id", "div", 
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
    """Hybrid deidentification: Model → Keypath context → Regex safeguard (priority-based)."""
    if not isinstance(value, str) or not value.strip():
        return value, []

    preds = ner(value)
    preds = merge_entities(preds)
    redacted = value
    redacted_flag = False
    chosen_label = None
    redaction_source = None
    confidence_score = 0.0

    # --- 1️⃣ MODEL-BASED REDACTION ---
    for e in preds:
        label = standardize_label(e.get("entity_group", ""))
        score = e.get("score", 0)
        if score >= CONF_THRESHOLD:
            pattern = re.compile(re.escape(e["word"]), re.IGNORECASE)
            redacted = pattern.sub(f"[{label}]", redacted)
            chosen_label = label
            confidence_score = max(confidence_score, score)
            redaction_source = "model"
            redacted_flag = True

    # --- 2️⃣ KEYPATH CONTEXT HINTS ---
    # Apply always, but only override if higher priority
    kp = keypath.lower()
    hint = None
    if re.search(r"name|maiden|given|family|prefix|suffix", kp):
        hint = "NAME"
    elif re.search(r"address|city|state|postal|geo", kp):
        hint = "LOCATION"
    elif re.search(r"birth|date", kp):
        hint = "DATE"
    elif re.search(r"phone|fax|email|telecom|contact", kp):
        hint = "CONTACT"
    elif re.search(r"id|identifier|license|account|passport", kp):
        hint = "ID"
    elif re.search(r"url|uri|web|internet|ip|image|photo", kp):
        hint = "WEB"

    # Override label if keypath has higher priority
    if hint and (not chosen_label or LABEL_PRIORITY[hint] > LABEL_PRIORITY.get(chosen_label, 0)):
        redacted = f"[{hint}]"
        chosen_label = hint
        redaction_source = "keypath"
        redacted_flag = True

    # --- 3️⃣ REGEX SAFEGUARD (only if unredacted or keypath hint ambiguous) ---
    if not redacted_flag or "[" not in redacted:
        for regex_entry in REGEX_PATTERNS:
            label = regex_entry["label"]
            pattern = regex_entry["pattern"]
            matches = pattern.findall(value)
            if matches:
                for match in matches:
                    match_text = match if isinstance(match, str) else match[0]
                    if not chosen_label or LABEL_PRIORITY[label] >= LABEL_PRIORITY.get(chosen_label, 0):
                        redacted = re.sub(re.escape(match_text), f"[{label}]", redacted)
                        chosen_label = label
                        confidence_score = regex_entry.get("score", 1.0)
                        redaction_source = "regex"
                        redacted_flag = True
                        break

    # --- 4️⃣ CONTEXTUAL FIXES (post-label cleanup) ---
    val_stripped = value.strip()
    if re.fullmatch(r"[A-Z]{2}", val_stripped):
        redacted, chosen_label = "[LOCATION]", "LOCATION"
    elif re.fullmatch(r"\d{5}", val_stripped):
        redacted, chosen_label = "[LOCATION]", "LOCATION"
    elif re.fullmatch(r"[A-Z][a-z]+[0-9]{2,}", val_stripped):
        redacted, chosen_label = "[NAME]", "NAME"

    if chosen_label:
        summary = [{
            "label": chosen_label,
            "original": value,
            "confidence": round(confidence_score or 1.0, 3),
            "source": redaction_source or "model"
        }]
    else:
        summary = []

    return redacted, summary

# PARSE + DEIDENTIFY JSON
_INDEXED_SEG_RE = re.compile(r"^([^\[\]]+)(?:\[(\d+)\])?$")

def set_value_at_path(obj, path, new_value):
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
                "original": e["original"],
                "label": e["label"],
                "confidence": e.get("confidence", 1.0),
                "source": e.get("source", "model")
            })

    return redacted_json, all_entities