class Model(object):
    """Base class for classification models."""
    def __init__(self, path, **kwargs):
        self.path = path
        self.data = kwargs.get("data", [])
        self.labels = kwargs.get("labels", [])
        self.tokens = kwargs.get("tokens", [])
        self.predicted = kwargs.get("predicted", [])
        self.subtypes = kwargs.get("subtypes", [])
        self.scores = kwargs.get("scores", [])

    def reset(self):
        self.sids = []
        self.sentences = []
        self.data = []
        self.labels = []
        self.sids = []
        self.tokens = []
        self.results = []
        self.predicted = []
        self.scores = []
