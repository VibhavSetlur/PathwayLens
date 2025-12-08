from dataclasses import dataclass

@dataclass
class AnalysisConfig:
    MIN_GENES: int = 10
    MAX_GENES: int = 500
    MIN_OVERLAP: int = 3
    FDR_THRESHOLD: float = 0.05
    PVALUE_THRESHOLD: float = 0.05
    LFC_THRESHOLD: float = 0.0
    EFFECT_THRESHOLD: float = 0.0
    FILTER_LOW_EXPRESSION: bool = True
    NORMALIZE_AUTO: bool = True
    BATCH_CORRECT: bool = False
    TOP_PATHWAYS: int = 20
    COLORBLIND_SAFE: bool = True
    THEME: str = "publication"
    DPI: int = 300

DEFAULT_CONFIG = AnalysisConfig()
