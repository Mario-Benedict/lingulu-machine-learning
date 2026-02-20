"""
Phoneme conversion utilities.
Converts between ARPABET and IPA phoneme representations.
"""
from typing import List, Dict
import nltk
import os
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Set NLTK data path if not already set
if 'NLTK_DATA' in os.environ:
    nltk_data_path = os.environ['NLTK_DATA']
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_path)
        logger.debug(f"Added NLTK data path: {nltk_data_path}")

# Download required NLTK data for g2p_en (only if not found)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    logger.debug("NLTK averaged_perceptron_tagger_eng found")
except LookupError:
    try:
        logger.info("Downloading NLTK averaged_perceptron_tagger_eng...")
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    except Exception as e:
        logger.warning(f"Failed to download NLTK data: {e}")

try:
    nltk.data.find('corpora/cmudict')
    logger.debug("NLTK cmudict found")
except LookupError:
    try:
        logger.info("Downloading NLTK cmudict...")
        nltk.download('cmudict', quiet=True)
    except Exception as e:
        logger.warning(f"Failed to download NLTK cmudict: {e}")


# ARPABET to IPA mapping with stress markers
ARPABET_TO_IPA: Dict[str, str] = {
    # Vowels (Monophthongs)
    'AA': 'ɑ', 'AA0': 'ɑ', 'AA1': 'ɑ', 'AA2': 'ɑ',       # bot
    'AE': 'æ', 'AE0': 'æ', 'AE1': 'æ', 'AE2': 'æ',       # bat
    'AH': 'ʌ', 'AH0': 'ə', 'AH1': 'ʌ', 'AH2': 'ʌ',       # butt (stressed vs. unstressed)
    'AO': 'ɔ', 'AO0': 'ɔ', 'AO1': 'ɔ', 'AO2': 'ɔ',       # bought
    'EH': 'ɛ', 'EH0': 'ɛ', 'EH1': 'ɛ', 'EH2': 'ɛ',       # bet
    'ER': 'ɝ', 'ER0': 'ɚ', 'ER1': 'ɝ', 'ER2': 'ɝ',       # bird (stressed vs. unstressed)
    'IH': 'ɪ', 'IH0': 'ɪ', 'IH1': 'ɪ', 'IH2': 'ɪ',       # bit
    'IY': 'i', 'IY0': 'i', 'IY1': 'i', 'IY2': 'i',       # beat
    'UH': 'ʊ', 'UH0': 'ʊ', 'UH1': 'ʊ', 'UH2': 'ʊ',       # book
    'UW': 'u', 'UW0': 'u', 'UW1': 'u', 'UW2': 'u',       # boot

    # Vowels (Diphthongs)
    'AW': 'aʊ', 'AW0': 'aʊ', 'AW1': 'aʊ', 'AW2': 'aʊ',   # bout
    'AY': 'aɪ', 'AY0': 'aɪ', 'AY1': 'aɪ', 'AY2': 'aɪ',   # bite
    'EY': 'eɪ', 'EY0': 'eɪ', 'EY1': 'eɪ', 'EY2': 'eɪ',   # bait
    'OW': 'oʊ', 'OW0': 'oʊ', 'OW1': 'oʊ', 'OW2': 'oʊ',   # boat
    'OY': 'ɔɪ', 'OY0': 'ɔɪ', 'OY1': 'ɔɪ', 'OY2': 'ɔɪ',   # boy

    # Consonants
    'P': 'p', 'B': 'b', 'T': 't', 'D': 'd', 'K': 'k', 'G': 'g',
    'CH': 'tʃ', 'JH': 'dʒ', 'F': 'f', 'V': 'v', 'TH': 'θ', 'DH': 'ð',
    'S': 's', 'Z': 'z', 'SH': 'ʃ', 'ZH': 'ʒ', 'HH': 'h', 'M': 'm',
    'N': 'n', 'NG': 'ŋ', 'L': 'l', 'R': 'ɹ', 'W': 'w', 'Y': 'j',

    # Syllabic Consonants
    'EM': 'm̩', 'EN': 'n̩', 'EL': 'l̩', 'NX': 'ɾ̃',
}


# IPA diphthongs that should be split for CTC alignment
IPA_SPLIT_MAP: Dict[str, List[str]] = {
    "aɪ": ["a", "ɪ"],
    "aʊ": ["a", "ʊ"],
    "eɪ": ["e", "ɪ"],
    "oʊ": ["o", "ʊ"],
    "ɔɪ": ["ɔ", "ɪ"],
}


class PhonemeConverter:
    """Converts phonemes between different representations."""
    
    def __init__(self):
        """Initialize the phoneme converter."""
        # Store mappings with different names to avoid overwriting methods
        self._arpabet_to_ipa_map = ARPABET_TO_IPA
        self._ipa_split_map = IPA_SPLIT_MAP
        
        # Cache for G2P conversions to avoid repeated calls
        self._g2p_cache: Dict[str, List[str]] = {}
        self._word_ipa_cache: Dict[tuple, List[str]] = {}
        
        logger.debug("PhonemeConverter initialized with caching")
    
    def arpabet_to_ipa(self, arpabet_list: List[str]) -> List[str]:
        """
        Convert ARPABET phonemes to IPA.
        
        Args:
            arpabet_list: List of ARPABET phonemes
            
        Returns:
            List of IPA phonemes
        """
        ipa = []
        for phoneme in arpabet_list:
            ipa_symbol = self._arpabet_to_ipa_map.get(phoneme)
            if ipa_symbol:
                ipa.append(ipa_symbol)
            else:
                logger.warning(f"Unknown ARPABET phoneme: {phoneme}")
        
        return ipa
    
    def split_ipa_diphthongs(self, ipa_phonemes: List[str]) -> List[str]:
        """
        Split IPA diphthongs into individual phonemes for better alignment.
        
        Args:
            ipa_phonemes: List of IPA phonemes
            
        Returns:
            List of split IPA phonemes
        """
        result = []
        for phoneme in ipa_phonemes:
            split = self._ipa_split_map.get(phoneme, [phoneme])
            result.extend(split)
        
        return result
    
    def text_to_ipa(
        self,
        text: str,
        g2p_model,
        split_diphthongs: bool = True
    ) -> List[str]:
        """
        Convert text to IPA phonemes with caching.
        
        Args:
            text: Input text
            g2p_model: G2P model instance
            split_diphthongs: Whether to split diphthongs
            
        Returns:
            List of IPA phonemes
        """
        # Create cache key
        cache_key = (text.lower().strip(), split_diphthongs)
        
        # Check cache first
        if cache_key in self._word_ipa_cache:
            return self._word_ipa_cache[cache_key]
        
        # Get ARPABET phonemes from g2p
        arpabet = [
            p for p in g2p_model(text) 
            if p.strip() and p not in [' ', ',', '.', '!', '?', '-', "'"]
        ]
        
        # Convert to IPA
        ipa = self.arpabet_to_ipa(arpabet)
        
        # Split diphthongs if requested
        if split_diphthongs:
            ipa = self.split_ipa_diphthongs(ipa)
        
        # Cache result
        self._word_ipa_cache[cache_key] = ipa
        
        logger.debug(f"Text '{text}' -> ARPABET: {arpabet} -> IPA: {ipa}")
        
        return ipa
    
    def word_to_ipa(
        self,
        word: str,
        g2p_model,
        split_diphthongs: bool = True
    ) -> List[str]:
        """
        Convert a single word to IPA phonemes with caching.
        
        Args:
            word: Input word
            g2p_model: G2P model instance
            split_diphthongs: Whether to split diphthongs
            
        Returns:
            List of IPA phonemes for the word
        """
        # Clean word
        word = word.strip().lower()
        
        # Create cache key
        cache_key = (word, split_diphthongs)
        
        # Check cache first
        if cache_key in self._word_ipa_cache:
            return self._word_ipa_cache[cache_key]
        
        # Get ARPABET
        arpabet = [p for p in g2p_model(word) if p.strip()]
        
        # Convert to IPA
        ipa = self.arpabet_to_ipa(arpabet)
        
        # Split diphthongs if requested
        if split_diphthongs:
            ipa = self.split_ipa_diphthongs(ipa)
        
        # Cache result
        self._word_ipa_cache[cache_key] = ipa
        
        return ipa
