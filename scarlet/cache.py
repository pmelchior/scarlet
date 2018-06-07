class Cache:
    """Cache to hold all complex proximal operators, transformation etc.

    Convention to use is that the lookup `name` refers to the class or method
    that pushes content onto the cache, the `key` can be chosen at will.

    """
    _cache = {}

    @staticmethod
    def check(name, key):
        try:
            Cache._cache[name]
        except KeyError:
            Cache._cache[name] = {}
        return Cache._cache[name][key]

    @staticmethod
    def set(name, key, content):
        try:
            Cache._cache[name]
        except KeyError:
            Cache._cache[name] = {}
        Cache._cache[name][key] = content

    @staticmethod
    def __repr__(self):
        repr(Cache._cache)
