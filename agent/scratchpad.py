import pandas as pd 
from agent.utils import summarize_value

class Scratchpad:
    def __init__(self):
        self.memory = {}

    def __contains__(self, key):
        return key in self.memory

    def __getitem__(self, key):
        return self.memory[key]

    def items(self):
        return self.memory.items()

    def set(self, key, value):
        self.memory[key] = value

    def get(self, key):
        return self.memory.get(key)
    
    def describe(self):
        return {
            k: summarize_value(v)
            for k, v in self.memory.items()
        }