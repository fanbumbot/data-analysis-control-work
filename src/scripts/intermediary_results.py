class IntermediaryResults(dict):
    def __getattr__(self, key):
        return self.get(key)
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        del self[key]

    def __iadd__(self, other):
        if isinstance(other, dict) or isinstance(other, IntermediaryResults):
            for key, value in other.items():
                self[key] = value
            return self
            
    def __repr__(self):
        return f"IntermediaryResults({dict(self)})"