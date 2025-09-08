
class tqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
        
    def __iter__(self):
        if self.iterable:
            return iter(self.iterable)
        return self
        
    def __next__(self):
        raise StopIteration
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        pass
        
    def set_description(self, desc):
        pass
        
    def update(self, n=1):
        pass
