import numpy as np
from typing import List, Tuple


class TermNameGenerator:
    """Generates linguistic term names based on center positions within data range."""

    # Constants for position thresholds
    POSITION_THRESHOLDS = {
        "very_low": 0.15,
        "low": 0.35,
        "medium_low": 0.45,
        "medium": 0.55,
        "medium_high": 0.65,
        "high": 0.85,
    }

    # Term name mappings
    STANDARD_TERMS = {
        1: ["Medium"],
        2: {"balanced": ["Low", "High"], "unbalanced": ["Lower", "Higher"]},
        3: {
            "balanced": ["Low", "Medium", "High"],
            "unbalanced": ["Lower", "Middle", "Higher"],
        },
    }

    def generate_term_names_from_centers(
        self, centers: np.ndarray, data_min: float, data_max: float
    ) -> List[str]:
        """Generate linguistic term names based on actual center positions within data range."""
        if len(centers) == 0:
            return []

        normalized_positions, sorted_indices = self._prepare_centers(
            centers, data_min, data_max
        )
        num_terms = len(centers)

        if num_terms <= 3:
            term_names = self._generate_simple_terms(normalized_positions, num_terms)
        else:
            term_names = self._generate_complex_terms(normalized_positions)

        return self._restore_original_order(term_names, sorted_indices, num_terms)

    def _prepare_centers(
        self, centers: np.ndarray, data_min: float, data_max: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and normalize center positions with better distribution."""
        centers_array = np.array(centers)
        sorted_indices = np.argsort(centers_array)
        sorted_centers = centers_array[sorted_indices]

        # For better distribution of term names, use equally spaced positions
        num_centers = len(centers)
        if num_centers > 1:
            # Create evenly spaced positions between 0 and 1
            normalized_positions = np.linspace(0, 1, num_centers)
        else:
            normalized_positions = np.array([0.5])

        return normalized_positions, sorted_indices

    def _generate_simple_terms(
        self, normalized_positions: np.ndarray, num_terms: int
    ) -> List[str]:
        """Generate term names for 1-3 terms."""
        if num_terms == 1:
            return self.STANDARD_TERMS[1]

        is_balanced = self._is_balanced_distribution(normalized_positions, num_terms)
        term_key = "balanced" if is_balanced else "unbalanced"

        return self.STANDARD_TERMS[num_terms][term_key]

    def _is_balanced_distribution(self, positions: np.ndarray, num_terms: int) -> bool:
        """Check if positions are balanced within expected ranges."""
        if num_terms == 2:
            return positions[0] <= 0.5 and positions[1] >= 0.5
        elif num_terms == 3:
            return (
                positions[0] <= 0.33
                and 0.33 <= positions[1] <= 0.67
                and positions[2] >= 0.67
            )
        return False

    def _generate_complex_terms(self, normalized_positions: np.ndarray) -> List[str]:
        """Generate term names for 4+ terms using position-based mapping."""
        term_names = []

        for pos in normalized_positions:
            base_name = self._get_term_for_position(pos)
            unique_name = self._ensure_unique_name(base_name, term_names)
            term_names.append(unique_name)

        return term_names

    def _get_term_for_position(self, position: float) -> str:
        """Map a normalized position to a base term name using more distributed thresholds."""
        if position <= 0.2:
            return "Very_Low"
        elif position <= 0.35:
            return "Low"
        elif position <= 0.45:
            return "Medium_Low"
        elif position <= 0.55:
            return "Medium"
        elif position <= 0.65:
            return "Medium_High"
        elif position <= 0.8:
            return "High"
        else:
            return "Very_High"

    def _ensure_unique_name(self, base_name: str, existing_names: List[str]) -> str:
        """Ensure the name is unique by appending a counter if necessary."""
        if base_name not in existing_names:
            return base_name

        counter = 2
        while f"{base_name}_{counter}" in existing_names:
            counter += 1

        return f"{base_name}_{counter}"

    def _restore_original_order(
        self, term_names: List[str], sorted_indices: np.ndarray, num_terms: int
    ) -> List[str]:
        """Restore term names to their original center order."""
        final_names = [""] * num_terms
        for i, sorted_idx in enumerate(sorted_indices):
            final_names[sorted_idx] = term_names[i]

        return final_names


# Example usage and integration with existing class
def generate_term_names_from_centers(
    self, centers: np.ndarray, data_min: float, data_max: float
) -> List[str]:
    """Refactored method that can be integrated into your existing class."""
    generator = TermNameGenerator()
    return generator.generate_term_names_from_centers(centers, data_min, data_max)
