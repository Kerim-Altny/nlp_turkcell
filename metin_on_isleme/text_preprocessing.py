"""
Metin Ön İşleme (Text Preprocessing)
====================================
İçerik:
  1. Veri temizleme (HTML, URL, e-posta, emoji, noktalama, sayı, boşluk)
  2. Tokenizasyon (kelime ve cümle)
  3. Stop words kaldırma
  4. Stemming (Porter, Snowball, Lancaster; Türkçe için snowballstemmer)
  5. Lemmatization (İngilizce: WordNet + POS; Türkçe: zeyrek)
  6. Hepsini birleştiren preprocess() pipeline'ı

Kurulum:
  pip install nltk
  pip install snowballstemmer zeyrek     # sadece Türkçe için (isteğe bağlı)

Kullanım:
  python text_preprocessing.py
  veya
  from text_preprocessing import preprocess
  preprocess("The cats are running!", lang="english", lemmatize=True)
"""

import contextlib
import html
import io
import logging
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# Türkçe için isteğe bağlı kütüphaneler
try:
    import snowballstemmer  # Türkçe stemmer
except ImportError:
    snowballstemmer = None

try:
    import zeyrek  # Türkçe morfolojik analiz / lemmatization
except ImportError:
    zeyrek = None


# ---------------------------------------------------------------------------
# 0. NLTK kaynaklarını indir (sadece eksik olanları)
# ---------------------------------------------------------------------------
NLTK_RESOURCES = {
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}


def download_nltk_resources():
    for name, path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


# ---------------------------------------------------------------------------
# 1. VERİ TEMİZLEME
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NUMBER_RE = re.compile(r"\d+")
MULTI_SPACE_RE = re.compile(r"\s+")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # semboller, emoji, vb.
    "\U00002600-\U000027BF"
    "\U0001F000-\U0001F2FF"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)


def turkish_lower(text: str) -> str:
    """Türkçe'ye uygun küçük harf (I -> ı, İ -> i). str.lower() bunu yanlış yapar."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def clean_text(
    text: str,
    lang: str = "english",
    lowercase: bool = True,
    remove_html: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_mentions: bool = True,
    keep_hashtag_word: bool = True,
    remove_emojis: bool = True,
    remove_numbers: bool = False,
    remove_punct: bool = True,
) -> str:
    """Ham metni temizler. Sıra önemlidir: önce URL/e-posta, en son noktalama."""
    if not isinstance(text, str):
        return ""

    if remove_html:
        text = html.unescape(text)  # &amp; -> &
        text = HTML_TAG_RE.sub(" ", text)
    if remove_urls:
        text = URL_RE.sub(" ", text)
    if remove_emails:
        text = EMAIL_RE.sub(" ", text)
    if remove_mentions:
        text = MENTION_RE.sub(" ", text)

    # #nlp -> nlp (kelimeyi tut) ya da tamamen sil
    text = HASHTAG_RE.sub(r"\1" if keep_hashtag_word else " ", text)

    if remove_emojis:
        text = EMOJI_RE.sub(" ", text)
    if remove_numbers:
        text = NUMBER_RE.sub(" ", text)

    # Türkçe: İstanbul'a -> İstanbul (kesme işaretinden sonraki ek atılır)
    if lang == "turkish":
        text = re.sub(r"['’]\w+", "", text)

    if lowercase:
        text = turkish_lower(text) if lang == "turkish" else text.lower()

    if remove_punct:
        # Noktalama yerine boşluk koy ("don't" -> "don t" olur; istersen
        # önce tokenize edip sonra noktalamayı at, bkz. remove_punct_tokens)
        text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        text = re.sub(r"[“”‘’«»…–—]", " ", text)  # tipografik işaretler

    return MULTI_SPACE_RE.sub(" ", text).strip()


def remove_punct_tokens(tokens):
    """Tokenizasyondan SONRA noktalama token'larını atmak için."""
    return [t for t in tokens if t not in string.punctuation and t not in {"``", "''", "..."}]


# ---------------------------------------------------------------------------
# 2. TOKENİZASYON
# ---------------------------------------------------------------------------
def tokenize_words(text: str, lang: str = "english"):
    return word_tokenize(text, language=lang)


def tokenize_sentences(text: str, lang: str = "english"):
    return sent_tokenize(text, language=lang)


# ---------------------------------------------------------------------------
# 3. STOP WORDS
# ---------------------------------------------------------------------------
def get_stopwords(lang: str = "english", extra=None, keep=None):
    """
    extra: listeye eklenecek özel kelimeler
    keep : listeden çıkarılacak kelimeler (örn. duygu analizi için "not", "no")
    """
    sw = set(stopwords.words(lang))
    if extra:
        sw |= set(extra)
    if keep:
        sw -= set(keep)
    return sw


def remove_stopwords(tokens, lang: str = "english", extra=None, keep=None):
    sw = get_stopwords(lang, extra, keep)
    return [t for t in tokens if t.lower() not in sw]


# ---------------------------------------------------------------------------
# 4. STEMMING
# ---------------------------------------------------------------------------
_porter = PorterStemmer()
_lancaster = LancasterStemmer()  # daha agresif


def stem_tokens(tokens, method: str = "snowball", lang: str = "english"):
    """
    method: "porter" | "snowball" | "lancaster"
    Türkçe için NLTK'de Snowball yok; snowballstemmer paketi kullanılır.
    """
    if lang == "turkish":
        if snowballstemmer is None:
            raise ImportError("Türkçe stemming için: pip install snowballstemmer")
        stemmer = snowballstemmer.stemmer("turkish")
        return stemmer.stemWords(tokens)

    if method == "porter":
        return [_porter.stem(t) for t in tokens]
    if method == "lancaster":
        return [_lancaster.stem(t) for t in tokens]
    if method == "snowball":
        s = SnowballStemmer(lang)
        return [s.stem(t) for t in tokens]
    raise ValueError(f"Bilinmeyen stemming yöntemi: {method}")


# ---------------------------------------------------------------------------
# 5. LEMMATIZATION
# ---------------------------------------------------------------------------
_wnl = WordNetLemmatizer()
_turkish_analyzer = None


def _penn_to_wordnet(tag: str) -> str:
    """Penn Treebank POS etiketini WordNet etiketine çevirir."""
    if tag.startswith("J"):
        return "a"  # sıfat
    if tag.startswith("V"):
        return "v"  # fiil
    if tag.startswith("R"):
        return "r"  # zarf
    return "n"      # isim (varsayılan)


def lemmatize_tokens(tokens, lang: str = "english", use_pos: bool = True):
    """
    İngilizce: WordNetLemmatizer. use_pos=True ise POS etiketi kullanılır;
               böylece "running" -> "run", "better" -> "good" gibi doğru sonuç çıkar.
    Türkçe   : zeyrek (morfolojik analiz), ilk lemma alınır.
    """
    global _turkish_analyzer

    if lang == "turkish":
        if zeyrek is None:
            raise ImportError("Türkçe lemmatization için: pip install zeyrek")
        if _turkish_analyzer is None:
            _turkish_analyzer = zeyrek.MorphAnalyzer()
            # zeyrek her analiz için logger.warning ile gereksiz mesaj basıyor
            logging.getLogger("zeyrek").setLevel(logging.ERROR)
            logging.getLogger("zeyrek.rulebasedanalyzer").setLevel(logging.ERROR)
        lemmas = []
        for tok in tokens:
            with contextlib.redirect_stdout(io.StringIO()):
                result = _turkish_analyzer.lemmatize(tok)  # [(kelime, [lemma, ...])]
            lemma = result[0][1][0] if result and result[0][1] else tok
            lemmas.append(turkish_lower(lemma))
        return lemmas

    if use_pos:
        tagged = nltk.pos_tag(tokens)
        return [_wnl.lemmatize(w, _penn_to_wordnet(t)) for w, t in tagged]
    return [_wnl.lemmatize(w) for w in tokens]


# ---------------------------------------------------------------------------
# 6. PIPELINE
# ---------------------------------------------------------------------------
def preprocess(
    text: str,
    lang: str = "english",
    remove_stop: bool = True,
    stem: str | None = None,      # None | "porter" | "snowball" | "lancaster"
    lemmatize: bool = False,
    remove_numbers: bool = False,
    min_token_len: int = 2,
    extra_stopwords=None,
    keep_stopwords=None,
):
    """
    Ham metin -> temiz token listesi.
    Not: stemming ve lemmatization genelde birlikte kullanılmaz, birini seç.
    """
    text = clean_text(text, lang=lang, remove_numbers=remove_numbers)
    tokens = tokenize_words(text, lang=lang)
    tokens = remove_punct_tokens(tokens)

    if remove_stop:
        tokens = remove_stopwords(tokens, lang, extra_stopwords, keep_stopwords)

    # Lemmatization POS etiketine ihtiyaç duyar; bu yüzden stop words'ten sonra,
    # ama stemming'den önce yapılır.
    if lemmatize:
        tokens = lemmatize_tokens(tokens, lang=lang)
    elif stem:
        tokens = stem_tokens(tokens, method=stem, lang=lang)

    return [t for t in tokens if len(t) >= min_token_len]


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------
def demo():
    download_nltk_resources()

    raw = (
        "<p>Check out https://example.com! The cats are running faster than "
        "the dogs 🐱🐶. Email me: test@mail.com #NLP @kerim "
        "I've been studying 3 hours &amp; it's GREAT!</p>"
    )

    print("=== HAM METİN ===")
    print(raw)

    print("\n=== 1) TEMİZLEME ===")
    cleaned = clean_text(raw)
    print(cleaned)

    print("\n=== 2) TOKENİZASYON ===")
    print("Kelime  :", tokenize_words(cleaned))
    print("Cümle   :", tokenize_sentences("Hello world. NLP is fun! Do you agree?"))

    print("\n=== 3) STOP WORDS ===")
    tokens = tokenize_words(cleaned)
    print(remove_stopwords(tokens))

    print("\n=== 4) STEMMING ===")
    words = ["running", "studies", "faster", "cats", "better", "generously"]
    print("Porter   :", stem_tokens(words, "porter"))
    print("Snowball :", stem_tokens(words, "snowball"))
    print("Lancaster:", stem_tokens(words, "lancaster"))

    print("\n=== 5) LEMMATIZATION ===")
    print("POS'suz  :", lemmatize_tokens(words, use_pos=False))
    print("POS'lu   :", lemmatize_tokens(words, use_pos=True))

    print("\n=== 6) PIPELINE ===")
    print("Lemma    :", preprocess(raw, lemmatize=True))
    print("Stem     :", preprocess(raw, stem="porter"))

    # Türkçe (zeyrek ve snowballstemmer kuruluysa)
    if zeyrek and snowballstemmer:
        print("\n=== TÜRKÇE ===")
        tr = "Çocuklar bahçede koşuyorlardı ve kitapları okuyacaklar! İSTANBUL'a gidiyoruz."
        print("Temiz    :", clean_text(tr, lang="turkish"))
        print("Stem     :", preprocess(tr, lang="turkish", stem="snowball"))
        print("Lemma    :", preprocess(tr, lang="turkish", lemmatize=True))
    else:
        print("\n(Türkçe demo için: pip install snowballstemmer zeyrek)")


if __name__ == "__main__":
    demo()
