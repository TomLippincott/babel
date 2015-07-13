from SCons.Builder import Builder
from SCons.Script import *
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
import pickle
import math
import xml.etree.ElementTree as et
import xml.sax
import sys
import gzip
from os.path import join as pjoin
from os import listdir
import tarfile
from random import randint, shuffle
from common_tools import DataSet, meta_open, pairs, temp_file, temp_dir
import time
import codecs
from attila import Audio, Decimator


def collect_text(target, source, env):
    """Gathers all the transcripts from a language pack whose file name matches a regular expression.

    This is used to gather e.g. just the LLP ("subtraining") data, versus the full FLP.

    Sources: language pack file, regular expression string
    Targets: transcript file
    """
    pattern = re.compile(source[1].read())
    discard = re.compile(r"^(\[.*\]|\<.*\>|\(.*\))\s*$")
    keep = re.compile(r".*\w+.*", re.UNICODE)
    with tarfile.open(source[0].rstr()) as ifd, meta_open(target[0].rstr(), "w") as ofd:
        for name in ifd.getnames():
            if pattern.match(name):
                if name.endswith("gz"):
                    stream = gzip.GzipFile(fileobj=ifd.extractfile(name))
                else:
                    stream = ifd.extractfile(name)
                for line in codecs.decode(stream.read(), "utf-8").split("\n"):
                    if not discard.match(line):
                        words = [word.strip("*") for word in line.strip().split() if not discard.match(word) and keep.match(word)]
                        words = set([word for word in words if word != ""])
                        if len(words) > 0:
                            ofd.write("%s\n" % (" ".join(words)))
    return None


def stm_to_data(target, source, env):
    """Gathers all the transcripts from the STM files that constitute VLLP, which isn't part of the basic language packs.

    Sources: language pack definition tarball, regular expression string
    Targets: transcript file
    """
    if source[0].rstr().endswith("tgz"):
        pattern = source[1].read()
        with tarfile.open(source[0].rstr()) as tf:
            names = [n for n in tf.getnames() if re.match(env.subst(pattern), n)]
            if len(names) == 0:
                return "No file in archive %s matched pattern %s" % (source[0].rstr(), pattern)
            elif len(names) > 1:
                return "More than one file in archive %s matched pattern %s" % (source[0].rstr(), pattern)
            else:
                text = codecs.decode(tf.extractfile(names[0]).read(), "utf-8")
    else:
        return "Must provide an archive and file name pattern"
    sentences = [[w for w in x.split()[3:] if not (w.startswith("(") or w.startswith("<"))] for x in text.split("\n")]
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join([" ".join(s) for s in sentences if len(s) > 0]) + "\n")
    return None


def segment_transcripts(target, source, env):
    """Applies a segmentation (mapping from words to morphs) and converts a word-transcript to a morph-transcript.

    Sources: word transcript file, segmentation file
    Targets: morph transcript file
    """
    mapping = {}
    with meta_open(source[1].rstr()) as morph_fd:
        for l in morph_fd:
            toks = l.strip().split()
            mapping["".join([x.strip("+") for x in toks])] = toks
    sentences = []
    with meta_open(source[0].rstr()) as text_fd:
        for l in text_fd:
            sentences.append(sum([mapping.get(w, [w]) for w in l.split()], []))
    with meta_open(target[0].rstr(), "w", enc="utf-8") as ofd:
        ofd.write("\n".join([" ".join(s) for s in sentences]))
    return None


def prepare_segmentations_for_release(target, source, env):
    """Intended to produce files suitable for shipping directly to partners.

    This is not well-planned, and should probably not be used as-is, but I think
    it would be useful to incorporate something automatic into the build system
    to make our deliverables more consistent.

    Sources: segmentation file 1, word file 1, segmentation file 2, word file 2 ...
    Targets: deliverable file 1, deliverable file 2 ...
    """
    nag = env.get("NON_ACOUSTIC_GRAPHEMES")
    rx_str = "^(%s)+$" % ("|".join([unichr(int(x, base=16)) for x in env.get("NON_ACOUSTIC_GRAPHEMES")]))
    rx = re.compile(rx_str)
    for (seg_file, word_file), out in zip(pairs(source, 2), target):
        with meta_open(seg_file.rstr()) as ifd:
            data = [line.strip().split() for line in ifd]
            morphs = {"".join([x.strip("+") for x in ms]) : ms for ms in data}
        with meta_open(word_file.rstr()) as ifd:
            lines = [l.strip().split() for l in ifd if "_" not in l]
            
        for words in lines:
            for word in sum([x.split("-") for x in words], []):
                if word != "" and word not in morphs and "_" not in word and "<" not in word and not re.match(r"^\d+$", word):
                    return "%s, %s, %s" % (seg_file, word_file, word)
                    
        with meta_open(out.rstr(), "w") as ofd:
            for morph, seg in sorted(morphs.iteritems()):
                ofd.write("%s\t%s\n" % (morph, " ".join(seg)))

    return None


def word_list(target, source, env):
    """Turns a transcript file into a list of unique words and their associated frequencies.

    Sources: transcript file
    Targets: word list file
    """
    words = {}
    with meta_open(source[0].rstr()) as ifd:
        for line in ifd:
            for word in line.strip().split():
                words[word] = words.get(word, 0) + 1
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(["%s %d" % (k, v) for k, v in words.iteritems()]) + "\n")            
    return None


def morfessor_babel_experiment(env, target_base, non_acoustic, word_lists=[]):
    """Trains a Morfessor model on the first word list provided, applies it to the remainder.

    Inputs: output path, list of non-acoustic phonemes/graphemes, list of word list files
    Outputs: segmented version of each word list input
    """
    segmented = []
    word_lists = Flatten(word_lists)
    if env.get("RUN_SEGMENTATION", True):
        env.Replace(TRAINING_NAME=word_lists[0])
        training_segmentations, model = env.TrainMorfessor(["work/morfessor/${SOURCE.filebase}_segmented.txt",
                                                            "work/morfessor/${SOURCE.filebase}.model"], word_lists[0], NON_ACOUSTIC_GRAPHEMES=non_acoustic)
        segmented.append(training_segmentations)
        env.PrepareSegmentationsForRelease("work/segmentations/${SOURCES[1].filebase}+${SOURCES[1].filebase}+morfessor.txt",
                                           [training_segmentations, word_lists[0]],
                                           NON_ACOUSTIC_GRAPHEMES=non_acoustic)
        for word_list in word_lists[1:]:
            segmentations = env.ApplyMorfessor("work/morfessor/%s+${SOURCES[1].filebase}+morfessor.txt" % (os.path.splitext(os.path.basename(word_lists[0].rstr()))[0]),
                                               [model, word_list])
            env.PrepareSegmentationsForRelease("work/segmentations/%s+${SOURCES[1].filebase}+morfessor.txt" % (os.path.splitext(os.path.basename(word_lists[0].rstr()))[0]),
                                               [segmentations, word_list],
                                               NON_ACOUSTIC_GRAPHEMES=non_acoustic)
    return segmented


def adaptor_grammar_babel_experiment(env, target_path, model_name, non_acoustic, word_lists=[]):
    """Trains an Adaptor Grammar model on the first word list provided, applies it to the remainder.

    Inputs: output path, model name, list of non-acoustic phonemes/graphemes, list of word list files
    Outputs: segmented version of each word list input
    """
    segmented = []
    env.Replace(MODEL_NAME=model_name)
    env.Replace(TARGET_PATH=target_path)
    #target_path = env.subst(target_path)
    #target_base = os.path.splitext(target)[0]
    #target_path = os.path.dirname(target)
    #print target, target_base, target_path

    #env.Replace(TARGET_BASE=target_base)

    training_words = word_lists[0]
    characters = env.CharacterProductions("${TARGET_PATH}/characters.txt", word_lists,
                                          NON_ACOUSTIC_GRAPHEMES=non_acoustic)
    pycfg_data = env.MorphologyData("${TARGET_PATH}/data.txt", training_words, NON_ACOUSTIC_GRAPHEMES=non_acoustic)
    cfg = env.ComposeGrammars("${TARGET_PATH}/cfg.txt", ["data/grammar_templates/simple_${MODEL_NAME}.txt", characters])
    segmentations, grammar, trace = env.RunPYCFG(["${TARGET_PATH}/output.txt",
                                                  "${TARGET_PATH}/grammar.txt",
                                                  "${TARGET_PATH}/trace.txt",
                                              ],
                                                 [cfg, pycfg_data])
    training_segmentations = env.NormalizePYCFGOutput("${TARGET_PATH}/output_normalized.txt", segmentations)
    segmented = [training_segmentations]
    #for keyword_list in Flatten(word_lists[1:]):
    #    kw_data = env.MorphologyData("${SOURCE.base}_${TRAINING_NAME}_${MODEL_NAME}.txt", keyword_list, NON_ACOUSTIC_GRAPHEMES=non_acoustic)
    #    segmented.append(env.ApplyAdaptorGrammar("${SOURCES[1].base}_${TRAINING_NAME}_${MODEL_NAME}_segmented.txt", [grammar, kw_data[0]]))
            
    return segmented


def TOOLS_ADD(env):
    """Conventional way to add the four builders and two methods to an SCons environment."""
    env.Append(BUILDERS = {
        "CollectText" : Builder(action=collect_text),
        "PrepareSegmentationsForRelease" : Builder(action=Action(prepare_segmentations_for_release, varlist=["NON_ACOUSTIC_GRAPHEMES"])),
        "SegmentTranscripts" : Builder(action=segment_transcripts),
        "WordList" : Builder(action=word_list),
        "StmToData" : Builder(action=stm_to_data),
    })
    env.AddMethod(morfessor_babel_experiment, "MorfessorBabelExperiment")
    env.AddMethod(adaptor_grammar_babel_experiment, "AdaptorGrammarBabelExperiment")
