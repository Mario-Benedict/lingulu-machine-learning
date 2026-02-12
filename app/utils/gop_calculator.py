"""
Goodness of Pronunciation (GOP) calculation utilities.
Implements CTC forced alignment and GOP scoring.
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict

from app.utils.logger import get_logger
from app.utils.phoneme_converter import PhonemeConverter

logger = get_logger(__name__)


@dataclass
class PhonemeScore:
    """Score for a single phoneme."""
    phoneme: str
    score: float  # Percentage 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'phoneme': self.phoneme,
            'score': round(self.score, 1)
        }


@dataclass
class WordScore:
    """Score for a single word."""
    word: str
    score: float  # Average percentage 0-100
    phonemes: List[PhonemeScore]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'word': self.word,
            'score': round(self.score, 1),
            'phonemes': [p.to_dict() for p in self.phonemes]
        }


@dataclass
class SentenceScore:
    """Score for entire sentence."""
    text: str
    average_score: float  # Overall average 0-100
    words: List[WordScore]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'average_score': round(self.average_score, 1),
            'words': [w.to_dict() for w in self.words]
        }


class GOPCalculator:
    """Calculate Goodness of Pronunciation scores."""
    
    def __init__(self, phoneme_converter: Optional[PhonemeConverter] = None):
        """
        Initialize GOP calculator.
        
        Args:
            phoneme_converter: PhonemeConverter instance
        """
        self.phoneme_converter = phoneme_converter or PhonemeConverter()
        logger.debug("GOPCalculator initialized")
    
    def ctc_forced_alignment(
        self,
        log_probs: torch.Tensor,
        target_ids: List[int],
        blank_id: int
    ) -> List[List[int]]:
        """
        Perform CTC forced alignment.
        
        Args:
            log_probs: Log probabilities tensor (T, V)
            target_ids: Target phoneme IDs
            blank_id: Blank token ID
            
        Returns:
            List of frame indices for each target phoneme
        """
        T, V = log_probs.shape
        N = len(target_ids)
        
        # Initialize trellis
        trellis = torch.full((T + 1, N + 1), -float("inf"))
        trellis[0, 0] = 0.0
        
        # Forward pass
        for t in range(T):
            # Blank transition
            trellis[t + 1, 0] = trellis[t, 0] + log_probs[t, blank_id]
            
            # Phoneme transitions
            for n in range(N):
                # Stay in current state (emit blank)
                stay = trellis[t, n + 1] + log_probs[t, blank_id]
                # Move to next phoneme
                move = trellis[t, n] + log_probs[t, target_ids[n]]
                trellis[t + 1, n + 1] = torch.maximum(stay, move)
        
        # Backtrack to find alignment
        t, n = T, N
        alignment = [[] for _ in range(N)]
        
        while t > 0 and n >= 0:
            if n > 0:
                # Check if we emitted phoneme at this step
                move_score = trellis[t - 1, n - 1] + log_probs[t - 1, target_ids[n - 1]]
                if torch.abs(move_score - trellis[t, n]) < 1e-6:
                    alignment[n - 1].append(t - 1)
                    t -= 1
                    n -= 1
                    continue
            
            # Otherwise, we emitted blank
            t -= 1
        
        # Reverse alignments (they were built backwards)
        return [list(reversed(a)) for a in alignment]
    
    def calculate_gop_scores(
        self,
        log_probs: torch.Tensor,
        ipa_phonemes: List[str],
        tokenizer
    ) -> List[float]:
        """
        Calculate GOP scores for phonemes.
        
        Args:
            log_probs: Log probabilities from model (T, V)
            ipa_phonemes: List of IPA phonemes
            tokenizer: Model tokenizer
            
        Returns:
            List of GOP scores (raw, not normalized)
        """
        vocab = tokenizer.get_vocab()
        
        # Get phoneme IDs, skip unknown phonemes
        phoneme_ids = []
        valid_phonemes = []
        for p in ipa_phonemes:
            if p in vocab:
                phoneme_ids.append(vocab[p])
                valid_phonemes.append(p)
            else:
                logger.warning(f"Phoneme '{p}' not in vocabulary, skipping")
        
        if not phoneme_ids:
            logger.warning("No valid phonemes found in vocabulary")
            return []
        
        blank_id = tokenizer.pad_token_id
        
        # Perform forced alignment
        try:
            alignment = self.ctc_forced_alignment(log_probs, phoneme_ids, blank_id)
        except Exception as e:
            logger.error(f"CTC alignment failed: {e}", exc_info=True)
            return [-5.0] * len(phoneme_ids)
        
        gop_scores = []
        
        for pid, frames in zip(phoneme_ids, alignment):
            if not frames:
                # No frames aligned to this phoneme
                gop_scores.append(-5.0)
                logger.debug(f"No alignment for phoneme ID {pid}")
                continue
            
            diffs = []
            for t in frames:
                frame = log_probs[t]
                target_score = frame[pid]
                
                # Get top 2 scores
                top2 = torch.topk(frame, min(2, len(frame)))
                
                # Find competitor (highest non-target score)
                if pid == torch.argmax(frame).item():
                    # Target is highest, use second highest
                    competitor_score = top2.values[1] if len(top2.values) > 1 else top2.values[0]
                else:
                    # Use highest
                    competitor_score = top2.values[0]
                
                # GOP is difference between target and competitor
                diff = target_score - competitor_score
                diffs.append(diff)
            
            # Average GOP for this phoneme
            avg_gop = torch.mean(torch.stack(diffs)).item()
            gop_scores.append(avg_gop)
        
        return gop_scores
    
    def normalize_gop_to_percentage(self, scores: List[float]) -> List[float]:
        """
        Normalize GOP scores to 0-100% range.
        
        Args:
            scores: Raw GOP scores
            
        Returns:
            Normalized scores (0-100)
        """
        if not scores:
            return []
        
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        
        # Use percentiles for robust normalization
        p5 = torch.quantile(scores_tensor, 0.05)
        p95 = torch.quantile(scores_tensor, 0.95)
        
        # Normalize to 0-1
        normalized = (scores_tensor - p5) / (p95 - p5 + 1e-6)
        normalized = torch.clamp(normalized, 0.0, 1.0)
        
        # Convert to percentage
        percentage = (normalized * 100).tolist()
        
        return percentage
    
    def map_phonemes_to_words(
        self,
        text: str,
        gop_scores: List[float],
        g2p_model
    ) -> Tuple[List[WordScore], List[str]]:
        """
        Map phoneme scores back to words.
        
        Args:
            text: Original text
            gop_scores: GOP scores for all phonemes
            g2p_model: G2P model for text-to-phoneme conversion
            
        Returns:
            Tuple of (list of WordScore objects, list of all phonemes)
        """
        words = text.split()
        idx = 0
        results = []
        all_phonemes = []
        
        for word in words:
            # Get IPA phonemes for this word
            ipa = self.phoneme_converter.word_to_ipa(
                word,
                g2p_model,
                split_diphthongs=True
            )
            
            all_phonemes.extend(ipa)
            n = len(ipa)
            
            # Get scores for this word's phonemes
            word_scores = gop_scores[idx:idx + n]
            
            # Handle case where we don't have enough scores
            if len(word_scores) < n:
                logger.warning(
                    f"Not enough scores for word '{word}': "
                    f"expected {n}, got {len(word_scores)}"
                )
                # Pad with zeros
                word_scores.extend([0.0] * (n - len(word_scores)))
            
            # Create PhonemeScore objects
            phoneme_scores = [
                PhonemeScore(phoneme=p, score=round(s, 1))
                for p, s in zip(ipa, word_scores)
            ]
            
            # Calculate word average
            if phoneme_scores:
                word_avg = sum(p.score for p in phoneme_scores) / len(phoneme_scores)
            else:
                word_avg = 0.0
            
            results.append(WordScore(
                word=word,
                score=word_avg,
                phonemes=phoneme_scores
            ))
            
            idx += n
        
        return results, all_phonemes
    
    def calculate_sentence_score(
        self,
        text: str,
        log_probs: torch.Tensor,
        tokenizer,
        g2p_model
    ) -> SentenceScore:
        """
        Calculate GOP scores for entire sentence.
        
        Args:
            text: Input text
            log_probs: Model log probabilities (T, V)
            tokenizer: Model tokenizer
            g2p_model: G2P model
            
        Returns:
            SentenceScore object with all scores
        """
        logger.debug(f"Calculating GOP for text: '{text}'")
        
        # Convert text to IPA phonemes
        ipa_phonemes = self.phoneme_converter.text_to_ipa(
            text,
            g2p_model,
            split_diphthongs=True
        )
        
        if not ipa_phonemes:
            logger.warning(f"No phonemes extracted from text: '{text}'")
            return SentenceScore(
                text=text,
                average_score=0.0,
                words=[]
            )
        
        # Calculate raw GOP scores
        gop_raw = self.calculate_gop_scores(log_probs, ipa_phonemes, tokenizer)
        
        if not gop_raw:
            logger.warning("No GOP scores calculated")
            return SentenceScore(
                text=text,
                average_score=0.0,
                words=[]
            )
        
        # Normalize to percentage
        gop_normalized = self.normalize_gop_to_percentage(gop_raw)
        
        # Map to words
        word_scores, phonemes_used = self.map_phonemes_to_words(
            text,
            gop_normalized,
            g2p_model
        )
        
        # Calculate sentence average
        if word_scores:
            sentence_avg = sum(w.score for w in word_scores) / len(word_scores)
        else:
            sentence_avg = 0.0
        
        logger.info(
            f"GOP calculated: {len(word_scores)} words, "
            f"avg score: {sentence_avg:.1f}%"
        )
        
        return SentenceScore(
            text=text,
            average_score=sentence_avg,
            words=word_scores
        )
