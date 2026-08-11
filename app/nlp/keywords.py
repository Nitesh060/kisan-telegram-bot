"""
Keyword mappings for intent detection
"""

# Keyword to intent mapping
# Each intent has a list of keywords that trigger it
INTENT_KEYWORDS = {
    "DIAGNOSIS": [
        # English
        "what disease", "which disease", "disease hai", "bimari",
        "problem", "issue", "affected", "infected", "what's wrong",
        "kya problem", "kya bimari", "isme kya", "ye kya",
        # Hindi
        "रोग", "बीमारी", "समस्या", "क्या समस्या", "क्या रोग",
        "कौन सा रोग", "इसमें क्या", "यह क्या है",
        # Hinglish
        "iska rog", "kaunsa disease", "ye plant sick", "crop mein problem",
    ],
    "SYMPTOMS": [
        # English
        "symptoms", "what are symptoms", "signs", "signs of",
        "looks like", "appearance", "appears",
        # Hindi
        "लक्षण", "क्या लक्षण", "निशानी", "दिख", "दिखता",
        "कैसा", "कैसी", "दिखाई",
        # Hinglish
        "lakshan", "lakshan kya", "dikh raha", "dikhta", "kaisa",
        "patta kaisa", "leaf kaisa",
    ],
    "CAUSE": [
        # English
        "why", "reason", "cause", "caused by", "because of",
        "what causes", "how did", "how this happen",
        # Hindi
        "क्यों", "कारण", "क्या कारण", "किस वजह से", "कैसे हुआ",
        "वजह", "कारक",
        # Hinglish
        "kyu", "kyun", "karan", "wajah", "matlab iska",
        "iska karan kya", "ye kaise hua",
    ],
    "MANAGEMENT": [
        # English
        "what to do", "what should i do", "management", "solution",
        "treatment", "remedy", "cure", "how to treat", "how to manage",
        "what treatment", "spray", "fertilizer", "pesticide",
        # Hindi
        "क्या करें", "क्या करना चाहिए", "इलाज", "समाधान", "दवा",
        "क्या दवा", "स्प्रे", "कीटनाशक", "उपचार",
        # Hinglish
        "kya kare", "kya karun", "ilaj", "ilaj kya", "dava",
        "kya dava", "upachar", "solution", "spray kya",
    ],
    "TREATMENT": [
        # English
        "treatment", "medicines", "drugs", "chemicals", "pesticide",
        "fungicide", "bactericide", "herbicide", "recommended",
        # Hindi
        "दवा", "दवाई", "औषधि", "रसायन", "कीटनाशक", "सुझाव",
        # Hinglish
        "medicine", "chemical spray", "ki dava", "istemal kare",
    ],
    "PREVENTION": [
        # English
        "prevent", "prevention", "avoid", "how to avoid", "how to prevent",
        "precaution", "precautions", "protect", "protection", "safe",
        # Hindi
        "रोकथाम", "बचाव", "कैसे बचें", "सावधानी", "सुरक्षा",
        "रोकें", "बचाएं", "सुरक्षित",
        # Hinglish
        "rok sakte", "kaise bacha", "bachav", "precaution",
        "safe kaise rakhe", "protect kaun", "avoid kaise",
    ],
    "SEVERITY": [
        # English
        "severe", "serious", "dangerous", "critical", "how bad",
        "how serious", "fatal", "risk", "danger", "urgent",
        # Hindi
        "गंभीर", "खतरनाक", "जोखिम", "संकट", "कितना गंभीर",
        # Hinglish
        "kitna serious", "dangerous", "risk", "bad", "urgent",
    ],
    "GENERAL_INFO": [
        # English
        "tell me", "explain", "information", "about", "info",
        "what is", "how", "where", "when", "help",
        # Hindi
        "बताएं", "समझाएं", "जानकारी", "के बारे में", "क्या है",
        "कैसे", "कब", "कहां",
        # Hinglish
        "batao", "samjhao", "bata", "info", "details",
        "kya hai", "malum do", "samajhte ho",
    ],
}


# Common crop names
CROP_KEYWORDS = {
    "Rice": ["rice", "chawal", "धान", "चावल"],
    "Wheat": ["wheat", "gehu", "गेहूं"],
    "Maize": ["maize", "corn", "makka", "मक्का"],
    "Cotton": ["cotton", "kapas", "कपास"],
    "Sugarcane": ["sugarcane", "ganna", "गन्ना"],
    "Tomato": ["tomato", "tamatar", "टमाटर"],
    "Potato": ["potato", "alu", "आलू"],
    "Onion": ["onion", "pyaz", "प्याज"],
    "Chilli": ["chilli", "mirch", "मिर्च"],
    "Mango": ["mango", "aam", "आम"],
    "Apple": ["apple", "seb", "सेब"],
    "Citrus": ["citrus", "orange", "lemon", "santra", "नारंगी"],
    "Banana": ["banana", "kela", "केला"],
    "Grape": ["grape", "angur", "अंगूर"],
    "Cabbage": ["cabbage", "bandh gobi", "बंद गोभी"],
    "Cucumber": ["cucumber", "kheera", "खीरा"],
    "Pumpkin": ["pumpkin", "kaddu", "कद्दू"],
    "Pea": ["pea", "matar", "मटर"],
}


# Disease keywords for search
DISEASE_KEYWORDS = {
    "Early Blight": ["early blight", "shuruwati khaad"],
    "Late Blight": ["late blight"],
    "Fusarium Wilt": ["wilt", "fusarium", "wilting"],
    "Septoria Leaf Spot": ["spot", "septoria", "daag"],
    "Bacterial Wilt": ["bacterial wilt"],
    "Powdery Mildew": ["mildew", "powder", "white powder", "safed chatri"],
    "Downy Mildew": ["downy mildew"],
    "Rust": ["rust", "khaad"],
    "Leaf Curl": ["curl", "leaf curl"],
    "Mosaic Virus": ["mosaic", "virus", "yellowing"],
}


def get_keywords_for_intent(intent: str) -> list:
    """Get keywords for a specific intent."""
    return INTENT_KEYWORDS.get(intent, [])


def get_crop_keywords(crop: str) -> list:
    """Get keywords for a crop."""
    return CROP_KEYWORDS.get(crop, [])


def find_crop_from_text(text: str) -> str:
    """Find crop name from text."""
    text_lower = text.lower()
    
    for crop, keywords in CROP_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return crop
    
    return None


def find_diseases_from_text(text: str) -> list:
    """Find disease keywords from text."""
    text_lower = text.lower()
    found_diseases = []
    
    for disease, keywords in DISEASE_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found_diseases.append(disease)
                break
    
    return found_diseases
