class ContextHistory:
    def __init__(self):
        self.entries = []

    def log(self, step: str, trace: list[dict]):
        self.entries.append({"step": step, "trace": trace})

    def recent(self, n=3):
        return self.entries[-n:]
    