from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class SpeciesInfo:
    name: str
    common_name: str
    ensembl_name: str
    ncbi_id: int
    aliases: List[str]

class Species:
    # Built-in species
    HUMAN = SpeciesInfo("human", "Human", "homo_sapiens", 9606, ["hsa", "homo_sapiens", "9606"])
    MOUSE = SpeciesInfo("mouse", "Mouse", "mus_musculus", 10090, ["mmu", "mus_musculus", "10090"])
    RAT = SpeciesInfo("rat", "Rat", "rattus_norvegicus", 10116, ["rno", "rattus_norvegicus", "10116"])
    ZEBRAFISH = SpeciesInfo("zebrafish", "Zebrafish", "danio_rerio", 7955, ["dre", "danio_rerio", "7955"])
    DROSOPHILA = SpeciesInfo("drosophila", "Fruit Fly", "drosophila_melanogaster", 7227, ["dme", "drosophila_melanogaster", "7227"])
    CELEGANS = SpeciesInfo("celegans", "Roundworm", "caenorhabditis_elegans", 6239, ["cel", "caenorhabditis_elegans", "6239"])
    YEAST = SpeciesInfo("yeast", "Yeast", "saccharomyces_cerevisiae", 4932, ["sce", "saccharomyces_cerevisiae", "4932"])
    ARABIDOPSIS = SpeciesInfo("arabidopsis", "Thale Cress", "arabidopsis_thaliana", 3702, ["ath", "arabidopsis_thaliana", "3702"])
    ECOLI = SpeciesInfo("ecoli", "E. coli", "escherichia_coli", 511145, ["eco", "escherichia_coli", "511145"])
    PIG = SpeciesInfo("pig", "Pig", "sus_scrofa", 9823, ["ssc", "sus_scrofa", "9823"])
    COW = SpeciesInfo("cow", "Cow", "bos_taurus", 9913, ["bta", "bos_taurus", "9913"])
    CHICKEN = SpeciesInfo("chicken", "Chicken", "gallus_gallus", 9031, ["gga", "gallus_gallus", "9031"])
    DOG = SpeciesInfo("dog", "Dog", "canis_familiaris", 9615, ["cfa", "canis_familiaris", "9615"])
    MACAQUE = SpeciesInfo("macaque", "Macaque", "macaca_mulatta", 9544, ["mcc", "macaca_mulatta", "9544"])
    CHIMPANZEE = SpeciesInfo("chimpanzee", "Chimpanzee", "pan_troglodytes", 9598, ["ptr", "pan_troglodytes", "9598"])

    _registry: Dict[str, SpeciesInfo] = {}

    @classmethod
    def _initialize_registry(cls):
        if not cls._registry:
            for attr_name in dir(cls):
                attr = getattr(cls, attr_name)
                if isinstance(attr, SpeciesInfo):
                    cls._registry[attr.name] = attr
                    for alias in attr.aliases:
                        cls._registry[alias] = attr

    @classmethod
    def get(cls, query: str) -> Optional[SpeciesInfo]:
        cls._initialize_registry()
        query = str(query).lower()
        return cls._registry.get(query)

    @classmethod
    def all_species(cls) -> List[SpeciesInfo]:
        cls._initialize_registry()
        return list({s.name: s for s in cls._registry.values()}.values())
