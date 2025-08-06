
import os
from typing import TypeVar, Generic

T = TypeVar("T")

class ConfigVar(Generic[T]):
    """
    Config flag that reads from environment variables at creation time

    Supports comparison operators
    """
    def __init__(self, name: str, default: T):
        self.name = name
        self.default = default
        self.value: T = self._get_from_env()

    def _get_from_env(self) -> T:
        env_value = os.environ.get(self.name)
        if env_value is None:
            return self.default
        # cast to the type of the default
        try:
            return type(self.default)(env_value)
        except (ValueError, TypeError):
            return self.default

    def __eq__(self, other) -> bool:
        return self.value == other

    def __ne__(self, other) -> bool:
        return self.value != other

    def __lt__(self, other) -> bool:
        return self.value < other

    def __le__(self, other) -> bool:
        return self.value <= other

    def __gt__(self, other) -> bool:
        return self.value > other

    def __ge__(self, other) -> bool:
        return self.value >= other

    def __repr__(self):
        return f"<ConfigVar {self.name}={self.value!r}>"

# add all environment variables below
DEBUG = ConfigVar('DEBUG', 0)
