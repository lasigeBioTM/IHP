import logging

class Token2(object):
    """
    Token that is part of a sentence
    The 2 is because there's already a token class in NLTK
    """
    def __init__(self, text, **kwargs):
        # TODO: require start and end and dstart and dend
        self.text = text
        self.sid = kwargs.get("sid")
        self.order = kwargs.get("order")
        # logging.debug("order: {}".format(self.order))
        self.features = {}
        self.tags = {}
        self.tid = kwargs.get("tid")