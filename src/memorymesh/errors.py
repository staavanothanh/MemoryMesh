class MemoryMeshError(Exception):
    pass

class RouterError(MemoryMeshError):
    def __init__(self, model: str, attempt: int, original_error: Exception):
        self.model = model
        self.attempt = attempt
        self.original_error = original_error
        super().__init__(f"RouterError: model={model}, attempts={attempt}, error={original_error}")

class EmbeddingError(MemoryMeshError):
    pass

class StorageError(MemoryMeshError):
    def __init__(self, operation: str, reason: str):
        self.operation = operation
        super().__init__(f"StorageError({operation}): {reason}")

class ValidationError(MemoryMeshError):
    pass

class LLMUnavailableError(MemoryMeshError):
    pass