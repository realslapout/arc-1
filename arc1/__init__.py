"""ARC-1: a 1.7B typed-decision model (choice / score / yes-no probability) for fast local inference."""
__version__ = "0.1.0"


def __getattr__(name):
    if name == "ARC1Predictor":
        from arc1.inference.predictor import ARC1Predictor
        return ARC1Predictor
    raise AttributeError(name)
