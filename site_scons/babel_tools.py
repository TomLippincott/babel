from SCons.Builder import Builder
from SCons.Script import *
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
import cPickle as pickle
import numpy
import math
try:
    import lxml.etree as et
except:
    import xml.etree.ElementTree as et
import xml.sax
import sys
import gzip
from os.path import join as pjoin
from os import listdir
import tarfile
from random import randint, shuffle
from common_tools import DataSet, meta_open

def bad_word(w):
    return (w.startswith("*") and w.endswith("*")) or (w.startswith("<") and w.endswith(">")) or (w.startswith("(") and w.endswith(")"))

def load_analyses(fname):
    retval = {}
    with meta_open(fname) as ifd:
        for i, l in enumerate(ifd):
            word, analyses = l.split("\t")
            retval[word] = (i, set())
            for a in analyses.split(","):
                i, ss = retval[word]
                ss.add(tuple([x.split(":")[0].strip() for x in a.split() if not x.startswith("~")]))
                retval[word] = (i, ss)
    return retval

def dummy(sources):
    return []

def collate_results(target, source, env):
    data = {}
    for k, v in env["MORPHOLOGY_RESULTS"].iteritems():
        with meta_open(v[0].rstr()) as ifd:
            names, values = [x.strip().split("\t") for x in ifd][0:2]
            data[k] = {n : "%.3f" % float(v) for n, v in zip(names, values)}
    for k, v in env["TAGGING_RESULTS"].iteritems():
        with meta_open(v[0].rstr()) as ifd:
            names, values = [x.strip().split("\t") for x in ifd][0:2]
            if k in data:
                data[k].update({n : "%.3f" % float(v) for n, v in zip(names, values)})
            else:
                data[k] = {n : "%.3f" % float(v) for n, v in zip(names, values)}
    with meta_open(target[0].rstr(), "w") as ofd:
        properties = ["Lang", "Method", "Units"]
        names = sorted(set(sum([x.keys() for x in data.values()], [])))
        ofd.write("\t".join(properties + names) + "\n")
        for k, v in data.iteritems():
            k = {"METHOD" : k[1], "LANG" : k[0], "UNITS" : k[2].split("-")[0]}
            ofd.write("\t".join([k.get(p.upper(), "").title() for p in properties] + [v.get(n, "") for n in names]) + "\n")
    return None

def top_words_by_tag(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[-1]
    counts = numpy.zeros(shape=(len(data.indexToWord), len(data.indexToTag)))
    for sentence in data.sentences:
        for w, t, aa in sentence:
            counts[w, t] += 1
    tag_totals = counts.sum(0)
    word_totals = counts.sum(1)
    keep = 10
    with meta_open(target[0].rstr(), "w") as ofd:
        for tag_id, tag_total in enumerate(tag_totals):
            word_counts = counts[:, tag_id] #.argsort()
            indices = [(i, word_counts[i]) for i in reversed(word_counts.argsort())][0:keep]
            ofd.write(" ".join(["%s-%.2f-%.2f" % (data.indexToWord[i], float(c) / tag_total, float(c) / word_totals[i]) for i, c in indices]) + "\n")
    return None

def conllish_to_xml(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        sentences = [[(w, t, []) for w, t in [re.split(r"\s+", x) for x in s.split("\n") if not re.match(r"^\s*$", x)]] for s in re.split(r"\n\n", ifd.read(), flags=re.M)]
    data = DataSet.from_sentences(sentences)
    with meta_open(target[0].rstr(), "w") as ofd:
        data.write(ofd)
    return None

def rtm_to_data(target, source, env):
    sentences = []
    with meta_open(source[0].rstr()) as ifd:
        for sentence in ifd:
            words = [w for w in sentence.split()[5:] if w not in ["(())", "IGNORE_TIME_SEGMENT_IN_SCORING"]]
            if len(words) > 0:
                sentences.append(words)
    dataset = DataSet.from_sentences([[(w, None, []) for w in s] for s in sentences])
    with meta_open(target[0].rstr(), "w") as ofd:
        dataset.write(ofd)
    return None

def extract_transcripts(target, source, env):
    args = source[-1].read()
    data = {}    
    tb = et.TreeBuilder()
    tb.start("xml", {})
    with tarfile.open(source[0].rstr()) as tf:
        for name in [n for n in tf.getnames() if re.match(args.get("PATTERN", r".*transcription.*\.txt"), n)]:
            text = tf.extractfile(name).read()
            try:
                tb.start("file", {"name" : name})
                tb.data(text.decode("utf-8"))
                tb.end("file")        
            except:
                print name, text
                raise
    tb.end("xml")
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(et.tostring(tb.close()))
    return None

def penn_to_transcripts(target, source, env):
    args = source[-1].read()
    data = {}    
    tb = et.TreeBuilder()
    tb.start("xml", {})
    with tarfile.open(source[0].rstr()) as tf:
        for name in [n for n in tf.getnames() if re.match(args.get("PATTERN", r".*parsed/mrg/wsj/.*mrg"), n)]:
            text = tf.extractfile(name).read()
            try:
                tb.start("file", {"name" : name})
                for sentence in text.split("(S "):
                    words = [m.group(1) for m in re.finditer(r"\(\S+ (\S+)\)", sentence)]
                    if len(words) > 0:
                        tb.data(" ".join(words) + "\n")
                tb.end("file")        
            except:
                print name, text
                raise
    tb.end("xml")
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(et.tostring(tb.close()))
    return None

def transcripts_to_data(target, source, env):
    args = source[-1].read()
    sentences = []
    with meta_open(source[0].rstr()) as ifd:
        for f in et.parse(ifd).getiterator("file"):
            name = f.get("name")
            for line in [l for l in f.text.split("\n") if not l.startswith("[")]:
                words = [(w, None, []) for w in line.split()]
                if len(words) > 0:
                    sentences.append(words)
    with meta_open(target[0].rstr(), "w") as ofd:
        DataSet.from_sentences(sentences).write(ofd)
    return None

def stm_to_data(target, source, env):
    if source[0].rstr().endswith("tgz"):
        pattern = source[1].read()
        with tarfile.open(source[0].rstr()) as tf:
            names = [n for n in tf.getnames() if re.match(env.subst(pattern), n)]
            if len(names) == 0:
                return "No file in archive %s matched pattern %s" % (source[0].rstr(), pattern)
            elif len(names) > 1:
                return "More than one file in archive %s matched pattern %s" % (source[0].rstr(), pattern)
            else:
                text = tf.extractfile(names[0]).read()            
    else:
        return "Must provide an archive and file name pattern"
    sentences = [x.split()[3:] for x in text.split("\n")]
    data = DataSet.from_sentences([[(w, None, []) for w in s] for s in sentences])
    with meta_open(target[0].rstr(), "w") as ofd:
        data.write(ofd)
    return None

def generate_data_subset(target, source, env):
    args = source[-1].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[0]
    indices = range(len(data.sentences))
    if args.get("RANDOM", False):
        shuffle(indices)
    to_keep = [1]
    total_words = 0
    while total_words < args.get("WORDS", 0):
        total_words += len(data.sentences[indices[0]])
        to_keep.append(indices[0])
        indices = indices[1:]
    with meta_open(target[0].rstr(), "w") as ofd:
        data.get_subset(to_keep).write(ofd)
    return None

def TOOLS_ADD(env):
    env.Append(BUILDERS = {
        "PennToTranscripts" : Builder(action=penn_to_transcripts),
        "GenerateDataSubset" : Builder(action=generate_data_subset),
        "StmToData" : Builder(action=stm_to_data),
        "ExtractTranscripts" : Builder(action=extract_transcripts),
        "TranscriptsToData" : Builder(action=transcripts_to_data),
        "CONLLishToXML" : Builder(action=conllish_to_xml),
        #"CollateResults" : Builder(action=collate_results),
        "TopWordsByTag" : Builder(action=top_words_by_tag),
        "RtmToData" : Builder(action=rtm_to_data),
    })
