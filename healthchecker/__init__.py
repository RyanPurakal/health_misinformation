"""Health claim checker: training-data classifier + semantic explanation matching."""

__all__ = ["ClaimCheckerService"]


def __getattr__(name: str):
    if name == "ClaimCheckerService":
        from healthchecker.service import ClaimCheckerService

        return ClaimCheckerService
    raise AttributeError(name)
