from .factorized_engine import FactorizedEngine
from .hyperprior_engine import HyperpriorEngine
from .meanscalehyperprior_engine import MeanScaleHyperpriorEngine
from .tcm_engine import TCMEngine
from .dcae_engine import DCAEEngine

ENGINE_TABLE = {
    "bmshj2018_factorized": FactorizedEngine,
    "bmshj2018_hyperprior": HyperpriorEngine,
    "mbt2018_mean":MeanScaleHyperpriorEngine,
    "tcm": TCMEngine,
    "dcae": DCAEEngine,
}

def create_engine(model_name: str, **kwargs):
    if model_name not in ENGINE_TABLE:
        raise ValueError(f"Unsupported model: {model_name}")
    return ENGINE_TABLE[model_name](**kwargs)
