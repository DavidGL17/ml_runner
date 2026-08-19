from abc import ABC, abstractmethod


class LayerParser(ABC):
    def __init__(self, layer_type: str):
        self.layer_type = layer_type

    @abstractmethod
    def to_dict(self) -> dict:
        """Return this layer's JSON-serializable representation, matching the `Layer` enum the Rust runtime deserializes from."""
        pass
