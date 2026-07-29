class ConfigError(ValueError):
    """Raised when the configuration file is invalid."""


class LaunchListError(ValueError):
    """Raised when the launch list CSV is invalid."""


class PropellantUseProfileError(ValueError):
    """Raised when propellant use profile input data is invalid."""


class EngineDataError(ValueError):
    """Raised when engine CSV input data is invalid."""


class PrimaryEmissionError(ValueError):
    """Raised when primary emission calculation fails."""


class ExportError(ValueError):
    """Raised when exporting output files fails."""


class PostCombustionError(ValueError):
    """Raised when post-combustion calculation fails."""


class InventoryTransformError(ValueError):
    """Raised when timestep inventory transformation fails."""

class VerificationError(ValueError):
    """Raised when inventory verification fails."""
