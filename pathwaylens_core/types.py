from enum import Enum

class OmicType(str, Enum):
    TRANSCRIPTOMICS = "transcriptomics"
    PROTEOMICS = "proteomics"
    EPIGENOMICS = "epigenomics"
    METABOLOMICS = "metabolomics"

class DataType(str, Enum):
    # Transcriptomics
    BULK = "bulk"
    SINGLECELL = "singlecell"
    SPATIAL = "spatial"
    TIMESERIES = "timeseries"
    
    # Proteomics
    SHOTGUN = "shotgun"
    TARGETED = "targeted"
    DIA = "dia"
    
    # Epigenomics
    ATACSEQ = "atacseq"
    CHIPSEQ = "chipseq"
    METHYL = "methyl"
    HICHIP = "hichip"
    
    # Metabolomics
    UNTARGETED = "untargeted"
    LIPIDOMICS = "lipidomics"
    FLUX = "flux"
