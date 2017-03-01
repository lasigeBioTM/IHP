import logging
partial_overlap_before = 1
partial_overlap_after = -1
no_overlap = 0
contains = 2
contained_by = -2
perfect_overlap = -3


class Offset(object):
    """
    Offset relative to a fragment of text
    """
    def __init__(self, start, end, **kwargs):
        self.start = start
        self.end = end
        self.text = kwargs.get("text")
        self.sid = kwargs.get("sid")
        self.eid = kwargs.get("eid")
        self.tag = kwargs.get("tag")

    def overlap(self, other_offset):
        """
            basic principle: if self.start before or is bigger than other, return positive
        :param other_offset: entity to compare if it overlaps with this one
        :return: 0 if they do not overlap,
                 -1 if other_entity ends with the beginning of this one,
                 1 if other_entity starts with the ending of this one,
                 2 if this entity contains other_entity
                 -2 if other_entity contains this entity
                 -3 perfect overlap
        """

        if self.start == other_offset.start:
            if self.end == other_offset.end: # perfect overlap
                return perfect_overlap
            # same start
            elif self.end > other_offset.end: # self is longer
                return 2
            else: # other is longer
                return -2
        elif self.end == other_offset.end: #same end
            if self.start < other_offset.start: # self is longer
                return 2
            else:
                return -2
        elif self.start < other_offset.start:
            if self.end < other_offset.start: # self appears before other_entity
                return no_overlap
            elif self.end < other_offset.end and self.end >= other_offset.start: # partial overlap or perfect
                return partial_overlap_before
            elif self.end > other_offset.end: # complete overlap
                return contains
            else:
                return 5
        # self appears after other_entity
        else: #other_offset.start <= self.start:
            if other_offset.end < self.start:
                return no_overlap
            # partial overlap or perfect
            elif other_offset.end < self.end and other_offset.end > self.start:
                return partial_overlap_after
            # complete overlap
            elif other_offset.end > self.end:
                return contained_by
            else:
                return -5


class Offsets(object):
    """
    Set of offsets relative to a text.
    """
    def __init__(self):
        self.offsets = set()

    def __iter__(self):
        return self

    def add_offset(self, o, exclude_this_if, exclude_others_if):
        """
        Check if offset is not repeated or overlapped and add.
        :param o: Offset object
        :return:
        """
        overlapping = []
        to_exclude = []
        v = 0
        toadd = True
        for oi, oo in enumerate(self.offsets):
            over = o.overlap(oo)
            if over in exclude_this_if:
                toadd = False
                v = over
                overlapping.append(oo)
                break
            elif over in exclude_others_if:
                toadd = True
                v = over
                to_exclude.append(oo)
            #if over not in (no_overlap,perfect_overlap):
            #    logging.info("Overlap of %s:%s:%s:%s and %s:%s:%s:%s = %s" % (o.text, o.start, o.end, o.sid,
            #                                                            oo.text, oo.start, oo.end, o.sid, over))
        if toadd:
            self.offsets.add(o)
            for oo in to_exclude:
                self.offsets.remove(oo)
        #logging.info(str(len(self.offsets)))
        return toadd, v, overlapping, to_exclude
