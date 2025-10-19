"""
Theme manager for visualization styling.
"""

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
from loguru import logger


class ThemeManager:
    """Manager for visualization themes and styling."""
    
    def __init__(self):
        """Initialize the theme manager."""
        self.logger = logger.bind(module="theme_manager")
        self._plotly_themes = self._initialize_plotly_themes()
        self._matplotlib_themes = self._initialize_matplotlib_themes()
    
    def _initialize_plotly_themes(self) -> Dict[str, Dict[str, Any]]:
        """Initialize Plotly theme configurations."""
        return {
            "default": {
                "layout": {
                    "font": {"family": "Arial", "size": 12},
                    "plot_bgcolor": "white",
                    "paper_bgcolor": "white",
                    "grid": {"color": "lightgray", "width": 0.5},
                    "colorway": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
                },
                "colorscale": "Viridis"
            },
            "dark": {
                "layout": {
                    "font": {"family": "Arial", "size": 12, "color": "white"},
                    "plot_bgcolor": "#1e1e1e",
                    "paper_bgcolor": "#1e1e1e",
                    "grid": {"color": "darkgray", "width": 0.5},
                    "colorway": ["#00d4ff", "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57", "#ff9ff3", "#54a0ff", "#5f27cd", "#00d2d3"]
                },
                "colorscale": "Viridis"
            },
            "scientific": {
                "layout": {
                    "font": {"family": "Times New Roman", "size": 14},
                    "plot_bgcolor": "white",
                    "paper_bgcolor": "white",
                    "grid": {"color": "lightgray", "width": 0.5},
                    "colorway": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E", "#8E44AD", "#E67E22", "#34495E", "#1ABC9C", "#E74C3C"]
                },
                "colorscale": "Viridis"
            },
            "colorblind": {
                "layout": {
                    "font": {"family": "Arial", "size": 12},
                    "plot_bgcolor": "white",
                    "paper_bgcolor": "white",
                    "grid": {"color": "lightgray", "width": 0.5},
                    "colorway": ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999", "#FFFFFF"]
                },
                "colorscale": "Viridis"
            }
        }
    
    def _initialize_matplotlib_themes(self) -> Dict[str, Dict[str, Any]]:
        """Initialize matplotlib theme configurations."""
        return {
            "default": {
                "style": "whitegrid",
                "palette": "Set2",
                "rcParams": {
                    "figure.dpi": 300,
                    "savefig.dpi": 300,
                    "font.size": 12,
                    "axes.titlesize": 14,
                    "axes.labelsize": 12,
                    "xtick.labelsize": 10,
                    "ytick.labelsize": 10,
                    "legend.fontsize": 10,
                    "figure.titlesize": 16
                }
            },
            "dark": {
                "style": "darkgrid",
                "palette": "Set2",
                "rcParams": {
                    "figure.dpi": 300,
                    "savefig.dpi": 300,
                    "font.size": 12,
                    "axes.titlesize": 14,
                    "axes.labelsize": 12,
                    "xtick.labelsize": 10,
                    "ytick.labelsize": 10,
                    "legend.fontsize": 10,
                    "figure.titlesize": 16,
                    "figure.facecolor": "#1e1e1e",
                    "axes.facecolor": "#1e1e1e",
                    "text.color": "white",
                    "axes.labelcolor": "white",
                    "xtick.color": "white",
                    "ytick.color": "white"
                }
            },
            "scientific": {
                "style": "whitegrid",
                "palette": "Set1",
                "rcParams": {
                    "figure.dpi": 300,
                    "savefig.dpi": 300,
                    "font.size": 14,
                    "font.family": "serif",
                    "axes.titlesize": 16,
                    "axes.labelsize": 14,
                    "xtick.labelsize": 12,
                    "ytick.labelsize": 12,
                    "legend.fontsize": 12,
                    "figure.titlesize": 18
                }
            },
            "minimal": {
                "style": "white",
                "palette": "Set2",
                "rcParams": {
                    "figure.dpi": 300,
                    "savefig.dpi": 300,
                    "font.size": 12,
                    "axes.titlesize": 14,
                    "axes.labelsize": 12,
                    "xtick.labelsize": 10,
                    "ytick.labelsize": 10,
                    "legend.fontsize": 10,
                    "figure.titlesize": 16,
                    "axes.spines.top": False,
                    "axes.spines.right": False
                }
            }
        }
    
    def apply_plotly_theme(
        self,
        fig: go.Figure,
        theme_name: str = "default"
    ) -> go.Figure:
        """
        Apply a theme to a Plotly figure.
        
        Args:
            fig: Plotly figure object
            theme_name: Name of the theme to apply
            
        Returns:
            Updated Plotly figure
        """
        try:
            if theme_name not in self._plotly_themes:
                self.logger.warning(f"Unknown theme: {theme_name}, using default")
                theme_name = "default"
            
            theme = self._plotly_themes[theme_name]
            
            # Apply layout theme
            fig.update_layout(**theme["layout"])
            
            # Apply colorscale to traces if applicable
            for trace in fig.data:
                if hasattr(trace, 'colorscale'):
                    trace.colorscale = theme["colorscale"]
            
            self.logger.info(f"Applied Plotly theme: {theme_name}")
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to apply Plotly theme: {e}")
            return fig
    
    def apply_matplotlib_theme(
        self,
        theme_name: str = "default"
    ) -> None:
        """
        Apply a theme to matplotlib.
        
        Args:
            theme_name: Name of the theme to apply
        """
        try:
            if theme_name not in self._matplotlib_themes:
                self.logger.warning(f"Unknown theme: {theme_name}, using default")
                theme_name = "default"
            
            theme = self._matplotlib_themes[theme_name]
            
            # Apply style
            sns.set_style(theme["style"])
            
            # Apply palette
            sns.set_palette(theme["palette"])
            
            # Apply rcParams
            plt.rcParams.update(theme["rcParams"])
            
            self.logger.info(f"Applied matplotlib theme: {theme_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply matplotlib theme: {e}")
    
    def get_plotly_theme(self, theme_name: str) -> Dict[str, Any]:
        """
        Get a Plotly theme configuration.
        
        Args:
            theme_name: Name of the theme
            
        Returns:
            Theme configuration dictionary
        """
        return self._plotly_themes.get(theme_name, self._plotly_themes["default"])
    
    def get_matplotlib_theme(self, theme_name: str) -> Dict[str, Any]:
        """
        Get a matplotlib theme configuration.
        
        Args:
            theme_name: Name of the theme
            
        Returns:
            Theme configuration dictionary
        """
        return self._matplotlib_themes.get(theme_name, self._matplotlib_themes["default"])
    
    def list_plotly_themes(self) -> List[str]:
        """
        List available Plotly themes.
        
        Returns:
            List of theme names
        """
        return list(self._plotly_themes.keys())
    
    def list_matplotlib_themes(self) -> List[str]:
        """
        List available matplotlib themes.
        
        Returns:
            List of theme names
        """
        return list(self._matplotlib_themes.keys())
    
    def create_custom_plotly_theme(
        self,
        theme_name: str,
        layout: Dict[str, Any],
        colorscale: str = "Viridis"
    ) -> bool:
        """
        Create a custom Plotly theme.
        
        Args:
            theme_name: Name for the custom theme
            layout: Layout configuration
            colorscale: Colorscale name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._plotly_themes[theme_name] = {
                "layout": layout,
                "colorscale": colorscale
            }
            
            self.logger.info(f"Created custom Plotly theme: {theme_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create custom Plotly theme: {e}")
            return False
    
    def create_custom_matplotlib_theme(
        self,
        theme_name: str,
        style: str,
        palette: str,
        rcParams: Dict[str, Any]
    ) -> bool:
        """
        Create a custom matplotlib theme.
        
        Args:
            theme_name: Name for the custom theme
            style: Seaborn style
            palette: Seaborn palette
            rcParams: Matplotlib rcParams
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._matplotlib_themes[theme_name] = {
                "style": style,
                "palette": palette,
                "rcParams": rcParams
            }
            
            self.logger.info(f"Created custom matplotlib theme: {theme_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create custom matplotlib theme: {e}")
            return False
    
    def get_color_palette(self, theme_name: str, n_colors: int = 10) -> List[str]:
        """
        Get a color palette for a theme.
        
        Args:
            theme_name: Name of the theme
            n_colors: Number of colors to return
            
        Returns:
            List of color codes
        """
        try:
            if theme_name in self._plotly_themes:
                colorway = self._plotly_themes[theme_name]["layout"].get("colorway", [])
                return colorway[:n_colors]
            elif theme_name in self._matplotlib_themes:
                palette = self._matplotlib_themes[theme_name]["palette"]
                return sns.color_palette(palette, n_colors).as_hex()
            else:
                self.logger.warning(f"Unknown theme: {theme_name}")
                return sns.color_palette("Set2", n_colors).as_hex()
                
        except Exception as e:
            self.logger.error(f"Failed to get color palette: {e}")
            return sns.color_palette("Set2", n_colors).as_hex()
    
    def reset_matplotlib_theme(self):
        """Reset matplotlib to default theme."""
        try:
            plt.rcdefaults()
            sns.reset_orig()
            self.logger.info("Reset matplotlib theme to default")
            
        except Exception as e:
            self.logger.error(f"Failed to reset matplotlib theme: {e}")
    
    def export_theme_config(
        self,
        theme_name: str,
        output_file: str
    ) -> bool:
        """
        Export theme configuration to file.
        
        Args:
            theme_name: Name of the theme
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            
            theme_config = {
                "plotly": self._plotly_themes.get(theme_name, {}),
                "matplotlib": self._matplotlib_themes.get(theme_name, {})
            }
            
            with open(output_file, 'w') as f:
                json.dump(theme_config, f, indent=2)
            
            self.logger.info(f"Exported theme configuration: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export theme configuration: {e}")
            return False
    
    def import_theme_config(
        self,
        input_file: str
    ) -> bool:
        """
        Import theme configuration from file.
        
        Args:
            input_file: Input file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            
            with open(input_file, 'r') as f:
                theme_config = json.load(f)
            
            # Import Plotly themes
            if "plotly" in theme_config:
                for theme_name, theme_data in theme_config["plotly"].items():
                    self._plotly_themes[theme_name] = theme_data
            
            # Import matplotlib themes
            if "matplotlib" in theme_config:
                for theme_name, theme_data in theme_config["matplotlib"].items():
                    self._matplotlib_themes[theme_name] = theme_data
            
            self.logger.info(f"Imported theme configuration: {input_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import theme configuration: {e}")
            return False
