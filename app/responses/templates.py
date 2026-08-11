"""
Response templates for Telegram bot
Deterministic response generation - no LLM
"""

# Language support
LANGUAGES = {
    "en": "English",
    "hi": "हिंदी",
    "hinglish": "Hinglish",
}

# Greeting messages
GREETING_MESSAGES = {
    "en": (
        "🌱 *Welcome to Kisan Crop Assistant!*\n\n"
        "I can help you identify crop diseases and provide management guidance.\n\n"
        "*What you can do:*\n"
        "📷 Send a crop/leaf photo\n"
        "💬 Ask about diseases\n"
        "🩺 Get disease information\n"
        "🛠 Get management guidance\n\n"
        "Send me a crop photo to begin!"
    ),
    "hi": (
        "🌱 *किसान फसल सहायक में आपका स्वागत है!*\n\n"
        "मैं आपको फसल रोगों की पहचान और प्रबंधन में मदद कर सकता हूँ।\n\n"
        "*आप क्या कर सकते हैं:*\n"
        "📷 फसल/पत्ती की तस्वीर भेजें\n"
        "💬 रोगों के बारे में पूछें\n"
        "🩺 रोग जानकारी प्राप्त करें\n"
        "🛠 प्रबंधन सलाह लें\n\n"
        "शुरू करने के लिए मुझे एक फसल की तस्वीर भेजें!"
    ),
    "hinglish": (
        "🌱 *Kisan Crop Assistant mein aapka swagat hai!*\n\n"
        "Main aapko fasal ke rog pehchanne aur unka upchar batane mein madad kar sakta hoon.\n\n"
        "*Aap kya kar sakte hain:*\n"
        "📷 Crop/patti ki photo bheje\n"
        "💬 Diseases ke baare mein poocho\n"
        "🩺 Disease info paao\n"
        "🛠 Management guidance lo\n\n"
        "Shuru karne ke liye mujhe ek fasal ki photo bhejo!"
    ),
}

# Help messages
HELP_MESSAGES = {
    "en": (
        "🌾 *How to use Kisan Crop Assistant*\n\n"
        "*1. Send a Photo*\n"
        "Take a clear photo of the affected leaf, stem, or fruit and send it to the bot.\n\n"
        "*2. Ask a Question*\n"
        "You can ask questions like:\n"
        "- 'What disease is this?'\n"
        "- 'How to treat this?'\n"
        "- 'How to prevent this?'\n\n"
        "*3. Get Information*\n"
        "The bot will identify the disease and provide management guidance from verified sources.\n\n"
        "*Commands:*\n"
        "/start - Start the bot\n"
        "/help - Show this help\n"
        "/language - Change language\n"
        "/about - About this bot\n"
    ),
    "hi": (
        "🌾 *किसान फसल सहायक का उपयोग कैसे करें*\n\n"
        "*1. फोटो भेजें*\n"
        "प्रभावित पत्ती, तना या फल की स्पष्ट तस्वीर लें और बॉट को भेजें।\n\n"
        "*2. प्रश्न पूछें*\n"
        "आप इस तरह प्रश्न पूछ सकते हैं:\n"
        "- 'यह कौन सा रोग है?'\n"
        "- 'इसका इलाज कैसे करें?'\n"
        "- 'इससे कैसे बचें?'\n\n"
        "*3. जानकारी प्राप्त करें*\n"
        "बॉट रोग की पहचान करेगा और सत्यापित स्रोतों से प्रबंधन सलाह देगा।\n\n"
        "*कमांड:*\n"
        "/start - बॉट शुरू करें\n"
        "/help - यह सहायता दिखाएं\n"
        "/language - भाषा बदलें\n"
        "/about - इस बॉट के बारे में\n"
    ),
}

# About messages
ABOUT_MESSAGES = {
    "en": (
        "🌿 *About Kisan Crop Assistant*\n\n"
        "This is an AI-powered agricultural bot designed to help farmers identify crop diseases and get management guidance.\n\n"
        "*Features:*\n"
        "✅ Image-based crop disease detection\n"
        "✅ Verified disease information from agricultural database\n"
        "✅ Hindi, English, and Hinglish support\n"
        "✅ Rule-based intelligent responses (no LLM)\n"
        "✅ Lightweight and cost-effective\n\n"
        "*Important:*\n"
        "This is a preliminary identification tool. For severe conditions, consult with local agricultural experts.\n\n"
        "Made by Annapurna Finance (AFPL) for Kisan Community"
    ),
    "hi": (
        "🌿 *किसान फसल सहायक के बारे में*\n\n"
        "यह एक AI-संचालित कृषि बॉट है जो किसानों को फसल रोगों की पहचान करने और प्रबंधन सलाह पाने में मदद करता है।\n\n"
        "*विशेषताएं:*\n"
        "✅ छवि-आधारित फसल रोग पहचान\n"
        "✅ कृषि डेटाबेस से सत्यापित रोग जानकारी\n"
        "✅ हिंदी, अंग्रेजी और हिंग्लिश समर्थन\n"
        "✅ नियम-आधारित बुद्धिमान प्रतिक्रियाएं\n"
        "✅ हल्का और लागत प्रभावी\n\n"
        "*महत्वपूर्ण:*\n"
        "यह एक प्रारंभिक पहचान उपकरण है। गंभीर परिस्थितियों के लिए स्थानीय कृषि विशेषज्ञों से परामर्श लें।\n\n"
        "अन्नपूर्णा फाइनेंस (AFPL) द्वारा किसान समुदाय के लिए बनाया गया"
    ),
}

# Error messages
ERROR_MESSAGES = {
    "en": {
        "invalid_image": "❌ Invalid image. Please send a clear, high-quality photo of the affected crop/leaf.",
        "image_too_large": "❌ Image too large. Please send an image smaller than 10 MB.",
        "image_too_small": "❌ Image too small. Please send a clear photo (minimum 64x64 pixels).",
        "model_not_ready": "⚠️ ML model is not configured. Cannot process images at this time.",
        "database_error": "⚠️ Database error. Please try again later.",
        "no_disease_found": "🤔 I couldn't find information about this disease in my database. Please try sending a clearer image.",
        "low_confidence": "🔍 I'm not confident about this prediction. Please send a clearer photo showing the affected leaves/stem/fruit.",
        "processing_error": "⚠️ Error processing image. Please try again with another photo.",
        "telegram_error": "⚠️ Telegram error. Please try again.",
    },
    "hi": {
        "invalid_image": "❌ अमान्य छवि। कृपया प्रभावित फसल/पत्ती की स्पष्ट, उच्च-गुणवत्ता वाली तस्वीर भेजें।",
        "image_too_large": "❌ छवि बहुत बड़ी है। कृपया 10 MB से छोटी छवि भेजें।",
        "image_too_small": "❌ छवि बहुत छोटी है। कृपया स्पष्ट तस्वीर भेजें (न्यूनतम 64x64 पिक्सेल)।",
        "model_not_ready": "⚠️ ML मॉडल कॉन्फ़िगर नहीं है। इस समय छवियों को संसाधित नहीं कर सकते।",
        "database_error": "⚠️ डेटाबेस त्रुटि। कृपया बाद में पुन: प्रयास करें।",
        "no_disease_found": "🤔 मुझे अपने डेटाबेस में इस रोग की जानकारी नहीं मिली। कृपया स्पष्ट तस्वीर भेजने का प्रयास करें।",
        "low_confidence": "🔍 मुझे इस भविष्यवाणी में आत्मविश्वास नहीं है। कृपया प्रभावित पत्तियों/तने/फल की स्पष्ट तस्वीर भेजें।",
        "processing_error": "⚠️ छवि संसाधित करने में त्रुटि। कृपया दूसरी तस्वीर के साथ पुन: प्रयास करें।",
        "telegram_error": "⚠️ Telegram त्रुटि। कृपया पुन: प्रयास करें।",
    },
}

# Disease prediction response template
DISEASE_PREDICTION_TEMPLATE = {
    "en": (
        "🌱 *Disease Identification Result*\n\n"
        "*Crop:* {crop}\n"
        "*Possible Disease:* {disease}\n"
        "*Confidence:* {confidence_percent}%\n\n"
        "🔍 *Symptoms:*\n{symptoms}\n\n"
        "🛠 *Management:*\n{management}\n\n"
        "{prevention_section}"
        "{treatment_section}"
        "⚠️ *Important Note:*\n"
        "This is an AI-based preliminary identification. For severe or persistent conditions, "
        "please consult with a local agricultural expert or extension officer.\n"
    ),
    "hi": (
        "🌱 *रोग पहचान परिणाम*\n\n"
        "*फसल:* {crop}\n"
        "*संभावित रोग:* {disease}\n"
        "*विश्वास:* {confidence_percent}%\n\n"
        "🔍 *लक्षण:*\n{symptoms}\n\n"
        "🛠 *प्रबंधन:*\n{management}\n\n"
        "{prevention_section}"
        "{treatment_section}"
        "⚠️ *महत्वपूर्ण नोट:*\n"
        "यह एक AI-आधारित प्रारंभिक पहचान है। गंभीर या स्थायी स्थितियों के लिए, "
        "कृपया स्थानीय कृषि विशेषज्ञ या प्रसार अधिकारी से परामर्श लें।\n"
    ),
}

# Confidence level templates
CONFIDENCE_TEMPLATES = {
    "en": {
        "high": "✅ *High Confidence:* I'm quite sure about this identification.",
        "medium": "⚠️ *Medium Confidence:* This is a possible identification, but please verify with the symptoms.",
        "low": "🔍 *Low Confidence:* I'm not very sure. Please send a clearer photo for better identification.",
    },
    "hi": {
        "high": "✅ *उच्च विश्वास:* मुझे इस पहचान के बारे में काफी निश्चितता है।",
        "medium": "⚠️ *मध्यम विश्वास:* यह एक संभावित पहचान है, लेकिन कृपया लक्षणों के साथ सत्यापित करें।",
        "low": "🔍 *कम विश्वास:* मुझे पूरा यकीन नहीं है। बेहतर पहचान के लिए कृपया स्पष्ट तस्वीर भेजें।",
    },
}

# Language selection message
LANGUAGE_MESSAGE = {
    "en": "Please select your preferred language:",
    "hi": "कृपया अपनी पसंदीदा भाषा चुनें:",
}
