"""
Styled metric card component.
"""

import streamlit as st
from typing import Optional, Union


def metrics_card(
    label: str,
    value: Union[str, int, float],
    delta: Optional[Union[str, int, float]] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None
) -> None:
    """
    Display a styled metric card.
    
    Args:
        label: Metric label
        value: Metric value (formatted string or number)
        delta: Optional delta value
        delta_color: 'normal', 'inverse', or 'off'
        help_text: Optional help tooltip
    """
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text
    )


def format_currency(value: float, prefix: str = "$") -> str:
    """
    Format a number as currency.
    
    Args:
        value: Number to format
        prefix: Currency prefix
        
    Returns:
        Formatted string
    """
    if value >= 1_000_000:
        return f"{prefix}{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{prefix}{value/1_000:.1f}K"
    else:
        return f"{prefix}{value:.0f}"


def format_percentage(value: float) -> str:
    """
    Format a number as percentage.
    
    Args:
        value: Number (0-1) to format
        
    Returns:
        Formatted string
    """
    return f"{value * 100:.0f}%"


