"""
Risk badge component with color coding.
"""

import streamlit as st
from typing import Literal


RiskLevel = Literal["Low", "Medium", "High", "Critical"]


# Color mapping for risk levels
RISK_COLORS = {
    "Low": {"bg": "#10B981", "text": "#ECFDF5", "icon": "circle-check"},
    "Medium": {"bg": "#F59E0B", "text": "#FFFBEB", "icon": "circle-alert"},
    "High": {"bg": "#EF4444", "text": "#FEF2F2", "icon": "circle-x"},
    "Critical": {"bg": "#7C2D12", "text": "#FEF2F2", "icon": "flame"},
}


def risk_badge(
    risk_category: RiskLevel,
    risk_score: int,
    show_score: bool = True
) -> str:
    """
    Generate a risk badge with color coding.
    
    Returns a string representation for the badge.
    
    Args:
        risk_category: Risk level category
        risk_score: Numeric risk score (0-100)
        show_score: Whether to show the numeric score
        
    Returns:
        Badge string
    """
    icons = {
        "Low": "[OK]",
        "Medium": "[!]",
        "High": "[!!]",
        "Critical": "[!!!]",
    }
    
    icon = icons.get(risk_category, "[?]")
    
    if show_score:
        return f"{icon} {risk_category} ({risk_score})"
    else:
        return f"{icon} {risk_category}"


def risk_badge_html(
    risk_category: RiskLevel,
    risk_score: int,
    show_score: bool = True
) -> str:
    """
    Generate an HTML risk badge with styling.
    
    Args:
        risk_category: Risk level category
        risk_score: Numeric risk score (0-100)
        show_score: Whether to show the numeric score
        
    Returns:
        HTML string for the badge
    """
    colors = RISK_COLORS.get(risk_category, RISK_COLORS["Medium"])
    
    score_display = f" ({risk_score})" if show_score else ""
    
    return f'''
    <span style="
        background-color: {colors['bg']};
        color: {colors['text']};
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-block;
    ">
        {risk_category}{score_display}
    </span>
    '''


def get_risk_emoji(risk_category: RiskLevel) -> str:
    """
    Get an emoji representation for risk level.
    
    Note: Uses ASCII alternatives for Windows compatibility.
    
    Args:
        risk_category: Risk level
        
    Returns:
        ASCII representation
    """
    emojis = {
        "Low": "[OK]",
        "Medium": "[!]",
        "High": "[X]",
        "Critical": "[!!]",
    }
    return emojis.get(risk_category, "[?]")


def risk_color(risk_category: RiskLevel) -> str:
    """
    Get the color for a risk level.
    
    Args:
        risk_category: Risk level
        
    Returns:
        CSS color string
    """
    colors = {
        "Low": "#10B981",
        "Medium": "#F59E0B",
        "High": "#EF4444",
        "Critical": "#7C2D12",
    }
    return colors.get(risk_category, "#6B7280")


