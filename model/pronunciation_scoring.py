"""
Pronunciation Scoring Utilities

Helper functions untuk menilai kualitas pronunciation
berdasarkan phoneme predictions dari model.
"""

import torch
import numpy as np
from jiwer import cer
from typing import List, Dict, Tuple


def calculate_phoneme_accuracy(predicted: str, reference: str) -> float:
    """
    Calculate phoneme-level accuracy between predicted and reference.
    
    Args:
        predicted: Predicted phoneme sequence
        reference: Reference/expected phoneme sequence
    
    Returns:
        Accuracy score (0-1)
    """
    error_rate = cer(reference, predicted)
    accuracy = max(0.0, 1.0 - error_rate)
    return accuracy


def phoneme_level_alignment(predicted: str, reference: str) -> List[Dict]:
    """
    Align predicted and reference phonemes to identify errors.
    
    Returns:
        List of dicts with phoneme-level comparison
    """
    pred_phonemes = list(predicted.replace("|", " ").replace(".", ""))
    ref_phonemes = list(reference.replace("|", " ").replace(".", ""))
    
    # Simple alignment (can be improved with DTW)
    max_len = max(len(pred_phonemes), len(ref_phonemes))
    alignment = []
    
    for i in range(max_len):
        pred_ph = pred_phonemes[i] if i < len(pred_phonemes) else None
        ref_ph = ref_phonemes[i] if i < len(ref_phonemes) else None
        
        alignment.append({
            'position': i,
            'predicted': pred_ph,
            'reference': ref_ph,
            'correct': pred_ph == ref_ph
        })
    
    return alignment


def calculate_pronunciation_score(
    predicted: str,
    reference: str,
    weights: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive pronunciation score.
    
    Args:
        predicted: Predicted phoneme sequence
        reference: Reference phoneme sequence
        weights: Custom weights for scoring components
    
    Returns:
        Dictionary with various scoring metrics
    """
    if weights is None:
        weights = {
            'accuracy': 0.5,
            'completeness': 0.3,
            'fluency': 0.2
        }
    
    # 1. Phoneme accuracy
    accuracy = calculate_phoneme_accuracy(predicted, reference)
    
    # 2. Completeness (length similarity)
    len_ratio = min(len(predicted), len(reference)) / max(len(predicted), len(reference))
    completeness = len_ratio
    
    # 3. Fluency (word boundary preservation)
    pred_words = predicted.split("|")
    ref_words = reference.split("|")
    word_score = min(len(pred_words), len(ref_words)) / max(len(pred_words), len(ref_words))
    fluency = word_score
    
    # Overall score
    overall_score = (
        weights['accuracy'] * accuracy +
        weights['completeness'] * completeness +
        weights['fluency'] * fluency
    )
    
    return {
        'overall': overall_score,
        'accuracy': accuracy,
        'completeness': completeness,
        'fluency': fluency,
        'components': {
            'predicted_length': len(predicted),
            'reference_length': len(reference),
            'predicted_words': len(pred_words),
            'reference_words': len(ref_words)
        }
    }


def get_pronunciation_feedback(
    predicted: str,
    reference: str,
    threshold: float = 0.7
) -> Dict[str, any]:
    """
    Generate detailed feedback for pronunciation improvement.
    
    Args:
        predicted: Predicted phoneme sequence
        reference: Reference phoneme sequence
        threshold: Accuracy threshold for "good" pronunciation
    
    Returns:
        Feedback dictionary with suggestions
    """
    score = calculate_pronunciation_score(predicted, reference)
    alignment = phoneme_level_alignment(predicted, reference)
    
    # Find errors
    errors = [a for a in alignment if not a['correct']]
    error_positions = [e['position'] for e in errors]
    
    # Generate feedback
    feedback = {
        'score': score,
        'overall_rating': 'Excellent' if score['overall'] >= 0.9 
                         else 'Good' if score['overall'] >= threshold
                         else 'Needs Improvement',
        'errors': {
            'count': len(errors),
            'positions': error_positions,
            'details': errors[:5]  # Top 5 errors
        },
        'suggestions': []
    }
    
    # Add specific suggestions
    if score['accuracy'] < threshold:
        feedback['suggestions'].append(
            f"Practice individual phonemes. Accuracy: {score['accuracy']:.1%}"
        )
    
    if score['completeness'] < 0.8:
        if len(predicted) < len(reference):
            feedback['suggestions'].append(
                "You're missing some sounds. Speak all phonemes clearly."
            )
        else:
            feedback['suggestions'].append(
                "You're adding extra sounds. Be more concise."
            )
    
    if score['fluency'] < 0.8:
        feedback['suggestions'].append(
            "Pay attention to word boundaries and rhythm."
        )
    
    # Identify problem phonemes
    problem_phonemes = {}
    for error in errors:
        ref_ph = error['reference']
        if ref_ph and ref_ph not in problem_phonemes:
            problem_phonemes[ref_ph] = 0
        if ref_ph:
            problem_phonemes[ref_ph] += 1
    
    if problem_phonemes:
        top_problem = max(problem_phonemes.items(), key=lambda x: x[1])
        feedback['suggestions'].append(
            f"Focus on phoneme '{top_problem[0]}' - appears in {top_problem[1]} errors"
        )
    
    return feedback


def stress_pattern_analysis(predicted: str, reference: str) -> Dict:
    """
    Analyze stress pattern correctness.
    
    Args:
        predicted: Predicted phoneme sequence with stress markers
        reference: Reference phoneme sequence with stress markers
    
    Returns:
        Stress pattern analysis
    """
    # Count stress markers
    pred_primary = predicted.count('ˈ')
    ref_primary = reference.count('ˈ')
    pred_secondary = predicted.count('ˌ')
    ref_secondary = reference.count('ˌ')
    
    # Find stress positions
    pred_stress_pos = [i for i, c in enumerate(predicted) if c in 'ˈˌ']
    ref_stress_pos = [i for i, c in enumerate(reference) if c in 'ˈˌ']
    
    # Calculate stress accuracy
    correct_stress = len(set(pred_stress_pos) & set(ref_stress_pos))
    total_stress = len(ref_stress_pos)
    stress_accuracy = correct_stress / total_stress if total_stress > 0 else 1.0
    
    return {
        'stress_accuracy': stress_accuracy,
        'predicted_stress_count': pred_primary + pred_secondary,
        'reference_stress_count': ref_primary + ref_secondary,
        'primary_stress_correct': pred_primary == ref_primary,
        'secondary_stress_correct': pred_secondary == ref_secondary,
        'details': {
            'predicted_positions': pred_stress_pos,
            'reference_positions': ref_stress_pos
        }
    }


def batch_scoring(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Calculate aggregate scores for a batch of predictions.
    
    Args:
        predictions: List of predicted phoneme sequences
        references: List of reference phoneme sequences
    
    Returns:
        Aggregate statistics
    """
    scores = [
        calculate_pronunciation_score(pred, ref)
        for pred, ref in zip(predictions, references)
    ]
    
    return {
        'mean_overall': np.mean([s['overall'] for s in scores]),
        'std_overall': np.std([s['overall'] for s in scores]),
        'mean_accuracy': np.mean([s['accuracy'] for s in scores]),
        'mean_completeness': np.mean([s['completeness'] for s in scores]),
        'mean_fluency': np.mean([s['fluency'] for s in scores]),
        'min_score': min(s['overall'] for s in scores),
        'max_score': max(s['overall'] for s in scores),
        'count': len(scores)
    }


# Example usage
if __name__ == "__main__":
    # Example pronunciation comparison
    reference = "həˈloʊ|wɜrld"
    predicted = "həˈloʊ|wərld"  # Small error in "world"
    
    print("=== Pronunciation Analysis ===\n")
    print(f"Reference: {reference}")
    print(f"Predicted: {predicted}\n")
    
    # Get comprehensive score
    score = calculate_pronunciation_score(predicted, reference)
    print(f"Overall Score: {score['overall']:.2%}")
    print(f"  - Accuracy:     {score['accuracy']:.2%}")
    print(f"  - Completeness: {score['completeness']:.2%}")
    print(f"  - Fluency:      {score['fluency']:.2%}\n")
    
    # Get feedback
    feedback = get_pronunciation_feedback(predicted, reference)
    print(f"Rating: {feedback['overall_rating']}")
    print(f"Errors: {feedback['errors']['count']}")
    if feedback['suggestions']:
        print("\nSuggestions:")
        for suggestion in feedback['suggestions']:
            print(f"  - {suggestion}")
    
    # Stress analysis
    stress = stress_pattern_analysis(predicted, reference)
    print(f"\nStress Accuracy: {stress['stress_accuracy']:.2%}")
