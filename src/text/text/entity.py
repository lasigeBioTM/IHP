from __future__ import division, absolute_import, unicode_literals

import xml.etree.ElementTree as ET
import logging
from text.offset import Offset, Offsets, perfect_overlap, contained_by
  
class Entity(object):
    """Base entity class"""

    def __init__(self, tokens, *args, **kwargs):
        self.type = kwargs.get('e_type', None)
        self.text = kwargs.get("text", None)
        self.did = kwargs.get("did", None)
        self.sid = kwargs.get("sid", None)
        self.eid = kwargs.get("eid")
        self.tokens = tokens
        if len(tokens) > 0:
            self.start = tokens[0].start
            self.end = tokens[-1].end
            self.dstart = tokens[0].dstart
            self.dend = tokens[-1].dend
        self.exclude = None
        self.dexclude = None
        self.recognized_by = []
        self.subentities = []
        self.targets = [] # targets should be (eid, relationtype)
        self.score = kwargs.get("score", 0)
        self.original_id = kwargs.get("original_id")
        self.normalized = self.text
        self.normalized_score = 0
        self.normalized_ref = "text"
        # logging.info("created entity {} with score {}".format(self.text, self.score))
        # print "entity", args, kwargs

    def __str__(self):
        output = "{}, s-offset: {}:{}, d-offset: {}:{}, tokens: {}, type: {}".format(self.text, self.start, self.end,
                                                                                     self.dstart, self.dend,
                                                                                     ' '.join([t.text for t in self.tokens]),
                                                                                     self.type)
        return output

    def write_chemdner_line(self, outfile, rank=1):
        if self.sid.endswith(".s0"):
            ttype = "T"
        else:
            ttype = "A"
        start = str(self.tokens[0].dstart)
        end = str(self.tokens[-1].dend)
        loc = ttype + ":" + start + ":" + end
        if isinstance(self.score ,dict):
            conf = sum(self.score.values())/len(self.score)
        else:
            conf = self.score
        #outfile.write('\t'.join([self.did, loc, str(rank)]) + '\n')
        outfile.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(self.did, loc, str(rank), str(conf), self.text))
        return (self.did, loc, str(rank), str(conf), self.text)

    def write_bioc_annotation(self, parent):
        bioc_annotation = ET.SubElement(parent, "annotation")
        bioc_annotation_text = ET.SubElement(bioc_annotation, "text")
        bioc_annotation_text.text = self.text
        bioc_annotation_info = ET.SubElement(bioc_annotation, "infon", {"key":"type"})
        bioc_annotation_info.text = self.type
        bioc_annotation_id = ET.SubElement(bioc_annotation, "id")
        bioc_annotation_id.text = self.eid
        bioc_annotation_offset = ET.SubElement(bioc_annotation, "offset")
        bioc_annotation_offset.text = str(self.dstart)
        bioc_annotation_length = ET.SubElement(bioc_annotation, "length")
        bioc_annotation_length.text = str(self.dend - self.dstart)
        return bioc_annotation

    def get_dic(self):
        dic = {}
        dic["text"] = self.text
        dic["type"] = self.type
        dic["eid"] = self.eid
        dic["offset"] = self.dstart
        dic["size"] = self.dend - self.dstart
        dic["sentence_offset"] = self.start
        return dic

    def validate(self, ths, rules, *args, **kwargs):
        return True

    def normalize(self):
        pass

class Entities(object):
    """Group of entities related to a text"""

    def __init__(self, **kwargs):
        self.elist = {"goldstandard":[]}
        self.sid = kwargs.get("sid")
        self.did = kwargs.get("did")

    def add_entity(self, entity, esource):
        """
        Add an entity to this entity group, indexed to esource
        """
        #entity.normalize()
        if esource not in self.elist:
            self.elist[esource] = []
        if esource + "_" + entity.type not in self.elist:
            self.elist[esource + "_" + entity.type] = []
            # logging.debug("created new entry %s for %s" % (esource, self.sid))
        #if entity in self.elist[esource]:
        #    logging.info("Repeated entity! %s", entity.eid)
        self.elist[esource].append(entity)
        self.elist[esource + "_" + entity.type].append(entity)

    def get_unique_entities(self, source, ths, rules):
        entities = set()
        offsets = Offsets()
        for s in self.elist:
            if s.startswith(source):
                for e in self.elist[s]:
                    val = e.validate(ths, rules)
                    if not val:
                        continue
                    eid_offset = Offset(e.dstart, e.dend, text=e.text, sid=e.sid)
                    exclude = [perfect_overlap]
                    if "contained_by" in rules:
                        exclude.append(contained_by)
                    toadd, v, alt = offsets.add_offset(eid_offset, exclude_if=exclude)
                    if toadd:
                        entities.add((e.text,))
        return entities

        #             for new_e in val: # validate should return a list of entities
        #                 eid_offset = Offset(new_e.dstart, new_e.dend, text=new_e.text, sid=new_e.sid)
        #                 exclude = [perfect_overlap]
        #                 if "contained_by" in rules:
        #                     exclude.append(contained_by)
        #                 toadd, v, overlaping, to_exclude = offsets.add_offset(eid_offset, exclude_this_if=exclude, exclude_others_if=[])
        #                 # print toadd, v, overlaping, to_exclude, new_e.normalized
        #                 if toadd:
        #                     # entities[new_e.text] = []
        #                     entities[new_e.normalized] = []
        # # print entities
        # return entities

    def write_chemdner_results(self, source, outfile, ths={"ssm":0.0}, rules=[], totalentities=0):
        """
        Write results that can be evaluated with the BioCreative evaluation script
        :param source: Base model path
        :param outfile: Text Results path to be evaluated
        :param ths: Thresholds
        :param rules: Validation rules
        :param totalentities: Number of entities already validated on this document (for ranking)
        :return:
        """
        lines = []
        offsets = Offsets()
        rank = totalentities
        #    print self.elist.keys()
        for s in self.elist:
            #if s != "goldstandard":
            #    logging.info("%s - %s(%s)" % (self.sid, s, source))
            if s.startswith(source): #use everything
                #logging.info("%s - %s" % (self.sid, s))

                for e in self.elist[s]:
                    val = e.validate(ths, rules)
                    if not val:
                        continue

                    # Overlap rules
                    eid_offset = Offset(e.dstart, e.dend, text=e.text, sid=e.sid)
                    exclude = [perfect_overlap]
                    if "contained_by" in rules:
                        exclude.append(contained_by)
                    toadd, v, alt = offsets.add_offset(eid_offset, exclude_if=exclude)
                    if toadd:
                        #logging.info("added %s" % e)
                        line = e.write_chemdner_line(outfile, rank)
                        lines.append(line)
                        rank += 1
        return lines, rank

    def get_results(self, esource):
        return self.elist.get(esource)

    def find_entity(self, start, end):
        """Find entity in this sentence between start and end (relative to document)"""
        entity = None
        for eid in self.elist["combined_results"]:
            if eid.start == start and eid.end == end:
                entity = eid
        return entity

    def combine_entities(self, base_model, name):
        """
        Combine entities from multiple models starting with base_model into one module named name
        :param base_model: string corresponding to the prefix of the models
        :param name: new model path
        """
        combined = {}
        offsets = Offsets()
        for s in self.elist:
            #logging.info("%s - %s" % (self.sid, s))
            if s.startswith(base_model) and s != name: #use everything
                for e in self.elist[s]: # TODO: filter for classifier confidence
                    #if any([word in e.text for word in self.stopwords]):
                    #    logging.info("ignored stopword %s" % e.text)
                    #    continue
                    #eid_alt =  e.sid + ":" + str(e.dstart) + ':' + str(e.dend)
                    next_eid = "{0}.e{1}".format(e.sid, len(combined))
                    eid_offset = Offset(e.dstart, e.dend, text=e.text, sid=e.sid, eid=next_eid)
                    added = False
                    # check for perfect overlaps
                    for i, o in enumerate(offsets.offsets):
                        overlap = eid_offset.overlap(o)
                        if overlap == perfect_overlap:
                            combined[o.eid].recognized_by.append(s)
                            combined[o.eid].score[s] = e.score
                            combined[o.eid].ssm_score_all[s] = e.ssm_score
                            added = True
                            #logging.info(combined[o.eid].ssm_score_all)
                            #logging.info("added {0}-{1} to entity {2}".format(s.split("_")[-1], e.text, combined[o.eid].text))
                            break
                    if not added:
                        offsets.offsets.add(eid_offset)
                        e.recognized_by = [s]
                        e.score = {s: e.score}
                        e.ssm_score_all= {s: e.ssm_score}
                        combined[next_eid] = e
                        #logging.info("new entity: {0}-{1}".format(s.split("_")[-1], combined[next_eid].text))
        self.elist[name] = combined.values()

    def get_offsets(self, esource, ths, rules):
        spans = []
        offsets = Offsets()
        for s in self.elist:
            #print "******", s, esource
            # logging.info("{}".format(s))
            # logging.info("esource: {}".format(es))
            if s.startswith(esource):
                # logging.info("using {}".format(s))
                for e in self.elist[s]:
                    val = e.validate(ths, rules)
                    if not val:
                        logging.info("excluded {}".format(e.text))
                        continue
                    eid_offset = Offset(e.dstart, e.dend, text=e.text, sid=e.sid)
                    exclude = [perfect_overlap]
                    if "contained_by" in rules:
                        exclude.append(contained_by)
                    #print "********", eid_offset.start, eid_offset.end, eid_offset.text
                    toadd, v, overlapped, to_exclude = offsets.add_offset(eid_offset, exclude_this_if=exclude, exclude_others_if=[])
                    #print toadd, v
                    #print e.dstart, e.dend, e.text
                    if toadd:
                        #logging.debug("added {}".format(e.text))
                        spans.append((e.dstart, e.dend, e.text))
                        # logging.info("added {}".format(e.text))
                    #else:
                        #logging.debug("did not add {}".format(e.text))
        return spans

    def get_entity(self, eid, source="goldstandard"):
        for e in self.elist[source]:
            if e.eid == eid:
                return e
        print "entity not found:", eid, source


    def get_offsets2(self, esource, ths, rules):
        spans = []
        offsets = Offsets()
        new_entities = []
        for s in self.elist:
            if s.startswith(esource):
                for e in self.elist[s]:
                    new_entities.append(e)
                    validated_entity = self.validate(e, ths, rules) #possibly make it as a list
                    if validated_entity != None:           
                        eid = self.sid + ".e" + str(len(self.elist[esource]))
                        new_entity = Entity(validated_entity[1], e.sid, text=validated_entity[0], did=e.did, #score=score,
                                            type=e.type, eid=eid)
                        new_entity.type = e.type
                        new_entities.append(new_entity)

                    for new_e in new_entities:
                        eid_offset = Offset(new_e.dstart, new_e.dend, text=new_e.text, sid=new_e.sid)
                        exclude = [perfect_overlap]
                        if "contained_by" in rules:
                            exclude.append(contained_by)
                        toadd, v, overlapped, to_exclude = offsets.add_offset(eid_offset, exclude_this_if=exclude, exclude_others_if=[])
                        if toadd:
                            spans.append((new_e.dstart, new_e.dend, new_e.text))
        return spans


    def validate(self, entity, ths, rules):
        if "andor" in rules:
            terms = []
            words = entity.text.split(" ")
            if "and" in words:
                smaller_entity = " ".join(words[:words.index("and")])
                token_list = entity.tokens[:words.index("and")]
                if len(smaller_entity.split(" ")) > 1:
                    return (smaller_entity, token_list)
            if "or" in words:
                smaller_entity = " ".join(words[:words.index("or")])
                token_list = entity.tokens[:words.index("or")]
                if len(smaller_entity.split(" ")) > 1:
                    return (smaller_entity, token_list)
        else:
            return None
