from pyit2fls import (
            min_t_norm, product_t_norm, lukasiewicz_t_norm, 
            drastic_t_norm, nilpotent_minimum_t_norm, hamacher_product_t_norm,
            max_s_norm, probabilistic_sum_s_norm, bounded_sum_s_norm,
            drastic_s_norm, nilpotent_maximum_s_norm, einstein_sum_s_norm
        )

T_NORM_MAP = {
            "min_t_norm": min_t_norm,
            "product_t_norm": product_t_norm,
            "lukasiewicz_t_norm": lukasiewicz_t_norm,
            "drastic_t_norm": drastic_t_norm,
            "nilpotent_minimum_t_norm": nilpotent_minimum_t_norm,
            "hamacher_product_t_norm": hamacher_product_t_norm,
        }

S_NORM_MAP = {
            "max_s_norm": max_s_norm,
            "probabilistic_sum_s_norm": probabilistic_sum_s_norm,
            "bounded_sum_s_norm": bounded_sum_s_norm,
            "drastic_s_norm": drastic_s_norm,
            "nilpotent_maximum_s_norm": nilpotent_maximum_s_norm,
            "einstein_sum_s_norm": einstein_sum_s_norm,
        }

def s_norm_fn(s_norm_name: str):
    """
    Get the s-norm function based on name.
    
    Args:
        s_norm_name: Name of the s-norm function
        
    Returns:
        S-norm function from PyIT2FLS
    """

    return S_NORM_MAP.get(s_norm_name, max_s_norm)

def t_norm_fn(t_norm_name: str):
    """
    Get the t-norm function based on name.

    Args:
        t_norm_name: Name of the t-norm function

    Returns:
        T-norm function from PyIT2FLS
    """
    return T_NORM_MAP.get(t_norm_name, min_t_norm)
