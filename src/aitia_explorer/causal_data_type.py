import enum


class CausalDataType(enum.Enum):
    """
    Used to set the type of data an algorithm can take.
    """
    MIXED_DATA = 'mixed'
    DISCRETE_DATA = 'discrete'
    CONTINUOUS_DATA = 'continuous'
