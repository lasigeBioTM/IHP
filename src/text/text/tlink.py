from text.pair import Pair


class TLink(Pair):
    def __init__(self, source, target, relation=False, *args, **kwargs):
        super(TLink, self).__init__([source, target], relation, *args, **kwargs)
        self.source_entity = source
        self.target_entity = target
        self.type = kwargs.get("rtype")
        self.original_id = kwargs.get("original_id")
        self.pid = kwargs.get("pid")