class SemanticFilter:
    _NOISE = frozenset({
        "hello", "hi", "thanks", "thank you", "ok", "okay", "sure", "got it",
        "cảm ơn", "được", "vâng", "ừ", "hiểu rồi", "dạ", "vâng ạ",
    })

    @staticmethod
    def is_valuable(content: str) -> bool:
        if not content or len(content.strip()) < 20:
            return False
        return content.strip().lower() not in SemanticFilter._NOISE
