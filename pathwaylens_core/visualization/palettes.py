"""
Colorblind-safe and publication-quality color palettes.

This module provides scientifically validated color schemes that are:
- Colorblind-accessible (tested for deuteranopia, protanopia, tritanopia)
- Print-safe (CMYK compatible)
- Publication-ready (meets journal standards)
"""

from typing import List, Dict
from enum import Enum


class PaletteType(str, Enum):
    """Types of color palettes."""
    SEQUENTIAL = "sequential"
    DIVERGING = "diverging"
    QUALITATIVE = "qualitative"
    COLORBLIND_SAFE = "colorblind_safe"


# Colorblind-safe qualitative palettes
COLORBLIND_SAFE_8 = [
    "#000000",  # Black
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7"   # Reddish Purple
]

# Wong's palette (widely used in scientific publications)
WONG_PALETTE = [
    "#000000",  # Black
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7"   # Reddish Purple
]

# Tol's bright palette (optimized for colorblindness)
TOL_BRIGHT = [
    "#4477AA",  # Blue
    "#EE6677",  # Red
    "#228833",  # Green
    "#CCBB44",  # Yellow
    "#66CCEE",  # Cyan
    "#AA3377",  # Purple
    "#BBBBBB"   # Grey
]

# Tol's muted palette (subtle, professional)
TOL_MUTED = [
    "#332288",  # Indigo
    "#88CCEE",  # Cyan
    "#44AA99",  # Teal
    "#117733",  # Green
    "#999933",  # Olive
    "#DDCC77",  # Sand
    "#CC6677",  # Rose
    "#882255",  # Wine
    "#AA4499"   # Purple
]

# Sequential palettes (for continuous data)
VIRIDIS = [
    "#440154", "#482777", "#3E4A89", "#31688E",
    "#26828E", "#1F9E89", "#35B779", "#6DCD59",
    "#B4DE2C", "#FDE724"
]

PLASMA = [
    "#0D0887", "#46039F", "#7201A8", "#9C179E",
    "#BD3786", "#D8576B", "#ED7953", "#FB9F3A",
    "#FDCA26", "#F0F921"
]

# Diverging palettes (for data with meaningful midpoint)
BLUE_RED_DIVERGING = [
    "#053061", "#2166AC", "#4393C3", "#92C5DE",
    "#D1E5F0", "#F7F7F7", "#FDDBC7", "#F4A582",
    "#D6604D", "#B2182B", "#67001F"
]

PURPLE_GREEN_DIVERGING = [
    "#40004B", "#762A83", "#9970AB", "#C2A5CF",
    "#E7D4E8", "#F7F7F7", "#D9F0D3", "#A6DBA0",
    "#5AAE61", "#1B7837", "#00441B"
]

# Publication-specific palettes
NATURE_PALETTE = [
    "#E64B35",  # Red
    "#4DBBD5",  # Blue
    "#00A087",  # Green
    "#3C5488",  # Dark Blue
    "#F39B7F",  # Orange
    "#8491B4",  # Purple
    "#91D1C2",  # Teal
    "#DC0000"   # Dark Red
]

SCIENCE_PALETTE = [
    "#3B4992",  # Blue
    "#EE0000",  # Red
    "#008B45",  # Green
    "#631879",  # Purple
    "#008280",  # Teal
    "#BB0021",  # Crimson
    "#5F559B",  # Violet
    "#A20056"   # Magenta
]


class ColorPalette:
    """Color palette manager for publication-quality visualizations."""
    
    # Registry of all available palettes
    PALETTES = {
        # Colorblind-safe
        "colorblind_safe": COLORBLIND_SAFE_8,
        "wong": WONG_PALETTE,
        "tol_bright": TOL_BRIGHT,
        "tol_muted": TOL_MUTED,
        
        # Sequential
        "viridis": VIRIDIS,
        "plasma": PLASMA,
        
        # Diverging
        "blue_red": BLUE_RED_DIVERGING,
        "purple_green": PURPLE_GREEN_DIVERGING,
        
        # Journal-specific
        "nature": NATURE_PALETTE,
        "science": SCIENCE_PALETTE
    }
    
    @classmethod
    def get_palette(cls, name: str, n_colors: int = None) -> List[str]:
        """
        Get color palette by name.
        
        Args:
            name: Palette name
            n_colors: Number of colors to return (None = all)
            
        Returns:
            List of hex color codes
        """
        if name not in cls.PALETTES:
            raise ValueError(f"Unknown palette: {name}. Available: {list(cls.PALETTES.keys())}")
        
        palette = cls.PALETTES[name]
        
        if n_colors is None:
            return palette
        elif n_colors <= len(palette):
            return palette[:n_colors]
        else:
            # Cycle through palette if more colors needed
            return [palette[i % len(palette)] for i in range(n_colors)]
    
    @classmethod
    def get_colorblind_safe_palette(cls, n_colors: int = 8) -> List[str]:
        """Get colorblind-safe palette."""
        return cls.get_palette("wong", n_colors)
    
    @classmethod
    def get_sequential_palette(cls, name: str = "viridis", n_colors: int = 10) -> List[str]:
        """Get sequential palette for continuous data."""
        return cls.get_palette(name, n_colors)
    
    @classmethod
    def get_diverging_palette(cls, name: str = "blue_red", n_colors: int = 11) -> List[str]:
        """Get diverging palette for data with midpoint."""
        return cls.get_palette(name, n_colors)
    
    @classmethod
    def get_publication_palette(cls, journal: str = "nature") -> List[str]:
        """Get journal-specific palette."""
        journal_map = {
            "nature": "nature",
            "science": "science",
            "cell": "nature",  # Use Nature palette for Cell
            "default": "wong"
        }
        palette_name = journal_map.get(journal.lower(), "wong")
        return cls.get_palette(palette_name)


# Plotly-compatible theme configurations
PLOTLY_PUBLICATION_THEME = {
    "layout": {
        "font": {
            "family": "Arial, sans-serif",
            "size": 11,
            "color": "#000000"
        },
        "plot_bgcolor": "#FFFFFF",
        "paper_bgcolor": "#FFFFFF",
        "colorway": WONG_PALETTE,
        "xaxis": {
            "showgrid": True,
            "gridcolor": "#E5E5E5",
            "gridwidth": 1,
            "zeroline": False,
            "showline": True,
            "linewidth": 1,
            "linecolor": "#000000",
            "mirror": True
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": "#E5E5E5",
            "gridwidth": 1,
            "zeroline": False,
            "showline": True,
            "linewidth": 1,
            "linecolor": "#000000",
            "mirror": True
        }
    }
}


def get_plotly_theme(style: str = "publication") -> Dict:
    """
    Get Plotly theme configuration.
    
    Args:
        style: Theme style ('publication', 'presentation', 'web')
        
    Returns:
        Plotly theme dictionary
    """
    if style == "publication":
        return PLOTLY_PUBLICATION_THEME
    elif style == "presentation":
        # Larger fonts for presentations
        theme = PLOTLY_PUBLICATION_THEME.copy()
        theme["layout"]["font"]["size"] = 14
        return theme
    else:  # web
        return {}  # Use Plotly defaults
