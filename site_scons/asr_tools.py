from SCons.Builder import Builder
from SCons.Action import Action
from SCons.Subst import scons_subst
from SCons.Util import is_List
from SCons.Node.FS import Dir
from scons_tools import ThreadedBuilder, threaded_run
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
import cPickle as pickle
import math
import xml.etree.ElementTree as et
import gzip
import subprocess
import shlex
import time
import shutil
import tempfile
import codecs
import locale
import bisect
from babel import ProbabilityList, Arpabo, Pronunciations, Vocabulary, FrequencyList
from common_tools import Probability, temp_file, temp_dir, meta_open
from torque_tools import run_command, torque_run
import torque
from os.path import join as pjoin
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot
import numpy
import sys
import user
import dsearch
import misc
import dbase
import frontend
import nnet
from attila import NNScorer, errorHandler


def ibm_train_language_model(target, source, env):
    text_file = source[0].rstr()
    n = source[1].read()
    with temp_dir() as prefix_dir, temp_file() as vocab_file, temp_file(suffix=".txt") as sentence_file, meta_open(text_file) as text_fd:
        # first create count files
        sentences = ["<s> %s </s>" % (l) for l in text_fd]
        words =  set(sum([s.split() for s in sentences], []) + ["<s>", "</s>", "<UNK>"])
        with meta_open(vocab_file, "w") as ofd:
            ofd.write("\n".join(words))
        with meta_open(sentence_file, "w") as ofd:
            ofd.write("\n".join(sentences))
        prefix = os.path.join(prefix_dir, "counts")
        cmd = "${ATTILA_PATH}/tools/lm_64/CountNGram -n %d %s %s %s" % (n, sentence_file, vocab_file, prefix)
        out, err, success = run_command(env.subst(cmd))
        if not success:
            return err
        
        # build LM
        lm = ".".join(target[0].rstr().split(".")[0:-2])
        cmd = "${ATTILA_PATH}/tools/lm_64/BuildNGram.sh -n %d -arpabo %s %s" % (n, prefix, lm)
        out, err, success = run_command(env.subst(cmd), env={"SFCLMTOOLS" : env.subst("${ATTILA_PATH}/tools/lm_64")})
        if not success:
            return err
        
    return None


def train_pronunciation_model(target, source, env):
    """
    g2p.py --train - --devel 5% --model test.model2 --ramp-up --write-model test.model3
    """
    train_fname = source[0].rstr()
    dev_percent = source[1].read()
    if len(source) == 3:
        previous = source[2].rstr()
        cmd = "${SEQUITUR_PATH}/bin/g2p.py --train - --devel %d%% --write-model %s --ramp-up --model %s" % (dev_percent, target[0].rstr(), previous)        
    else:
        cmd = "${SEQUITUR_PATH}/bin/g2p.py --train - --devel %d%% --write-model %s" % (dev_percent, target[0].rstr())
    with open(train_fname) as ifd:
        data = "\n".join([re.sub(r"^(\S+)\(\d+\) (\S+) \[ wb \] (.*) \[ wb \]$", r"\1 \2 \3", line.strip()) for line in ifd if "REJ" not in line and line[0] != "<" and "SIL" not in line])
        #print data
        out, err, success = run_command(env.subst(cmd), env={"PYTHONPATH" : env.subst("${SEQUITUR_PATH}/lib/python2.7/site-packages")}, data=data)
        if not success:
            return err
        else:
            return None


def transcript_vocabulary(target, source, env):
    """
    Input: list of transcript files
    Output: sorted vocabulary file
    """
    words = set()
    for f in source:
        with meta_open(f.rstr()) as ifd:
            words = words.union(set(sum([[word.strip().lower() for word in line.split() if not word[0] == "<"] for line in ifd if not re.match(r"^\[\d+\.\d+\]$", line)], [])))
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(sorted(words)))
    return None


def missing_vocabulary(target, source, env):
    """
    
    """
    with meta_open(source[0].rstr()) as lm_fd, meta_open(source[1].rstr()) as dict_fd, meta_open(target[0].rstr(), "w") as new_dict:
        dict_words = {}
        for l in dict_fd:
            if "REJ" not in l:
                m = re.match(r"^(.*)\(\d+\) (.*)$", l)
                word, pron = m.groups()
                dict_words[word] = dict_words.get(pron, []) + [pron.replace("[ wb ]", "")]
        lm_words = set([m.group(1) for m in re.finditer(r"^\-\d+\.\d+ (\S+) \-\d+\.\d+$", lm_fd.read(), re.M)])
        for word, prons in dict_words.iteritems():
            if word not in lm_words:
                for pron in prons:
                    new_dict.write("%s %s\n" % (word, pron))
    return None


def augment_language_model(target, source, env):
    """
    Input: old language model, old pronunciations, new pronunciations|
    ** old language model, old pronunciations, new pronunciations
    Output: new language model, new vocab, new pronunciations
    """
    #from arpabo import Arpabo, Pronunciations

    weighted = len(source) == 5
        

    old_prons = Pronunciations(meta_open(source[0].rstr()))
    old_lm = Arpabo(meta_open(source[1].rstr()))
    new_prons = Pronunciations(meta_open(source[2].rstr()))
    mass = source[-1].read()

    logging.info("Old LM: %s", old_lm)
    logging.info("Old Pronunciations: %s", old_prons)
    logging.info("Words to add: %s", new_prons)

    if weighted:
        new_probs = ProbabilityList(meta_open(source[3].rstr()))
        logging.info("Words to add (probabilities): %s", new_probs)


    old_prons.add_entries(new_prons)
    if weighted:
        old_lm.add_unigrams_with_probs(new_probs, mass)
    else:
        old_lm.add_unigrams(new_prons.get_words(), mass)

    logging.info("New Pronunciations: %s", old_prons)
    logging.info("New LM: %s", old_lm)
    logging.info("New words have weight %s", old_lm.get_probability_of_words(new_prons.get_words()))
    logging.info("Old words have weight %s", old_lm.get_probability_of_not_words(new_prons.get_words()))

    with meta_open(target[0].rstr(), "w") as new_vocab, meta_open(target[1].rstr(), "w") as new_prons, meta_open(target[2].rstr(), "w") as new_lm:
        new_lm.write(old_lm.format())
        new_vocab.write(old_prons.format_vocabulary())
        new_prons.write(old_prons.format())
    return None


def augment_language_model_emitter(target, source, env):
    """
    Input: either a single pronunciations, or something else
    Output: given a pronunciations, set up the appropriate dependencies, otherwise pass through
    """
    # if there's more than one source, or it isn't a Python value, don't modify anything
    if len(source) != 1:
        return target, source
    else:
        try:
            config = source[0].read()
        except:
            return target, source
        base_path = env.get("BASE", "work")
        new_targets = [os.path.join(base_path, x % (config["NAME"])) for x in ["%s_vocab.txt", "%s_pronunciations.txt", "%s_lm.arpabo.gz"]]
        new_sources = [config[x] for x in ["OLD_PRONUNCIATIONS_FILE", "OLD_LANGUAGE_MODEL_FILE", "NEW_PRONUNCIATIONS_FILE"]] + [env.Value(config["PROBABILITY_MASS"])]
        return new_targets, new_sources


def collect_text_emitter(target, source, env):
    return target, source


def create_asr_experiment(target, source, env):

    # the first three sources are the original configuration dictionaries
    files, directories, parameters = [x.read() for x in source[:3]]
    files = {k : env.File(v) for k, v in files.iteritems()}
    directories = {k : env.Dir(os.path.abspath(v)) for k, v in directories.iteritems()}

    # the remainder are template files
    templates = source[3:6]

    # create one big configuration dictionary
    config = {k : v for k, v in sum([list(y) for y in [files.iteritems(), directories.iteritems(), parameters.iteritems()]], [])}
    #config["GRAPH_OFILE"] = env.File(os.path.join(config["OUTPUT_PATH"].rstr(), "dnet.bin.gz"))
    #config["CTM_OPATH"] = env.Dir(os.path.abspath(os.path.join(config["OUTPUT_PATH"].rstr(), "ctm")))
    #config["LAT_OPATH"] = env.Dir(os.path.abspath(os.path.join(config["OUTPUT_PATH"].rstr(), "lat")))
    #config["DATABASE_FILE"] = config["SEGMENTATION_FILE"]

    # print dictionary for debugging
    logging.debug("%s", "\n".join(["%s = %s" % (k, v) for k, v in config.iteritems()]))

    # perform substitution on each template file, write to appropriate location
    for template, final in zip(templates, target):
        with open(template.rstr()) as ifd, open(final.rstr(), "w") as ofd:
            ofd.write(scons_subst(ifd.read(), env=env, lvars=config))

    return None


def create_asr_experiment_emitter(target, source, env):

    # start with three configuration dictionaries    
    files, directories, parameters = [x.read() for x in source]

    directories["CONFIGURATION_PATH"] = target[0].rstr()

    for f in files.keys():
        if is_List(files[f]) and len(files[f]) > 0:
            files[f] = files[f][0]

    # all templates
    dlatsa = ["cfg.py", "construct.py", "test.py"]

    new_sources, new_targets = [], []

    # new list of targets
    new_targets = [pjoin(directories["CONFIGURATION_PATH"], x) for x in dlatsa]

    # new list of sources
    new_sources = [env.Value({k : str(v) for k, v in files.iteritems()}), env.Value({k : str(v) for k, v in directories.iteritems()}), env.Value(parameters)] + \
        [os.path.join("data", "%s.%s" % (x, parameters["LANGUAGE_ID"])) for x in dlatsa] + \
        [p for n, p in files.iteritems()]

    return new_targets, new_sources


def babelgum_lexicon(target, source, env):
    size = source[2].read()
    with meta_open(source[0].rstr()) as ifd:
        probabilities = sorted([(float(p), w) for w, p in [x.strip().split() for x in meta_open(source[0].rstr())]])[0:size]
    pronunciations = {}
    with meta_open(source[1].rstr()) as ifd:
        for w, n, p in [x.groups() for x in re.finditer(r"^(\S+)\((\d+)\) (.*?)$", ifd.read(), re.M)]:
            pronunciations[w] = pronunciations.get(w, {})
            pronunciations[w][int(n)] = p
    with meta_open(target[0].rstr(), "w") as prob_ofd, meta_open(target[1].rstr(), "w") as pron_ofd:
        prob_ofd.write("\n".join(["%s %f" % (w, p) for p, w in probabilities]))
        for w in sorted([x[1] for x in probabilities]):
            for n, p in sorted(pronunciations[w].iteritems()):
                pron_ofd.write("%s(%.2d) %s\n" % (w, n, p))
    return None


def replace_pronunciations(target, source, env):
    """
    Takes two pronunciation files, and replaces pronunciations in the first with those from the second, 
    for overlapping words.  Returns a new vocabulary file and pronunciation file.
    """
    with meta_open(source[0].rstr()) as old_fd, meta_open(source[1].rstr()) as repl_fd:
        old = Pronunciations(old_fd)
        repl = Pronunciations(repl_fd)
    logging.info("Old pronunciations: %s", old)
    logging.info("Replacement pronunciations: %s", repl)
    old.replace_by(repl)
    logging.info("New pronunciations: %s", old)
    with meta_open(target[0].rstr(), "w") as voc_ofd, meta_open(target[1].rstr(), "w") as pron_ofd:
        voc_ofd.write(old.format_vocabulary())
        pron_ofd.write(old.format())
    return None


def filter_words(target, source, env):
    """
    Takes a coherent language model, pronunciation file and vocabulary file, and a second
    vocabulary file, and returns a coherent language model, pronunciation file and vocabulary 
    file limited to the words in the second vocabulary file.

    The language model probabilities are scaled such that unigrams sum to one. ***
    """
    with meta_open(source[0].rstr()) as voc_fd, meta_open(source[1].rstr()) as pron_fd, meta_open(source[2].rstr()) as lm_fd, meta_open(source[3].rstr()) as lim_fd:
        lm = Arpabo(lm_fd)
        pron = Pronunciations(pron_fd)
        voc = Vocabulary(voc_fd)
        lim = Vocabulary(lim_fd)
    logging.info("Original vocabulary: %s", voc)
    logging.info("Original pronunciations: %s", pron)
    logging.info("Original LM: %s", lm)
    logging.info("Limiting vocabulary: %s", lim)
    logging.info("Vocabulary to remove has mass: %s", lm.get_probability_of_not_words(lim.get_words()))
    logging.info("Vocabulary to remain has mass: %s", lm.get_probability_of_words(lim.get_words()))
    lm.filter_by(lim)
    pron.filter_by(lim)
    voc.filter_by(lim)
    logging.info("New vocabulary: %s", voc)
    logging.info("New pronunciations: %s", pron)
    logging.info("New LM: %s", lm)
    with meta_open(target[0].rstr(), "w") as voc_ofd, meta_open(target[1].rstr(), "w") as pron_ofd, meta_open(target[2].rstr(), "w") as lm_ofd:
        voc_ofd.write(voc.format())
        pron_ofd.write(pron.format())
        lm_ofd.write(lm.format())
    return None


def filter_babel_gum(target, source, env):
    with meta_open(source[0].rstr()) as pron_ifd, meta_open(source[1].rstr()) as prob_ifd, meta_open(source[2].rstr()) as lim_ifd:
        pron = Pronunciations(pron_ifd)
        logging.info("Old pronunciations: %s", pron)
        prob = ProbabilityList(prob_ifd)
        logging.info("Old probabilities: %s", prob)
        filt = Vocabulary(lim_ifd)
        logging.info("Correct words: %s", filt)
        pron.filter_by(filt)
        logging.info("New pronunciations: %s", pron)
        prob.filter_by(filt)
        logging.info("New probabilities: %s", prob)
        with meta_open(target[0].rstr(), "w") as pron_ofd, meta_open(target[1].rstr(), "w") as prob_ofd:
            pron_ofd.write(pron.format())
            prob_ofd.write(prob.format())
    return None


def run_asr_experiment_torque(target, source, env):
    args = source[-1].read()
    construct_command = env.subst("${ATTILA_INTERPRETER} ${SOURCES[1].abspath}", source=source)
    out, err, success = run_command(construct_command)
    if not success:
        return out + err
    stdout = env.Dir(args.get("stdout", args["path"])).Dir("stdout").rstr()
    stderr = env.Dir(args.get("stderr", args["path"])).Dir("stderr").rstr()
    if not os.path.exists(stdout):
        os.makedirs(stdout)
    if not os.path.exists(stderr):
        os.makedirs(stderr)
    command = env.subst("${ATTILA_INTERPRETER} ${SOURCES[2].abspath} -n ${TORQUE_JOBS_PER_SCONS_INSTANCE} -j $${PBS_ARRAYID} -w ${ACOUSTIC_WEIGHT} -l 1", source=source)
    interval = args.get("interval", 10)
    job = torque.Job(args.get("name", "scons"),
                     commands=[command],
                     path=args["path"],
                     stdout_path=stdout,
                     stderr_path=stderr,
                     array=args.get("array", 0),
                     other=args.get("other", ["#PBS -W group_list=yeticcls"]),
                     )
    if env["HAS_TORQUE"]:
        job.submit(commit=True)
        while job.job_id in [x[0] for x in torque.get_jobs(True)]:
            logging.debug("sleeping...")
            time.sleep(interval)
    else:
        logging.info("no Torque server, but I would submit:\n%s" % (job))
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(time.asctime() + "\n")
    return None


def run_asr_experiment_emitter(target, source, env):
    args = {"array" : env["TORQUE_JOBS_PER_SCONS_INSTANCE"],
            "interval" : 120}
    try:
        args.update(source[0].read())
    except:
        args["path"] = source[0].get_dir().rstr()
    return target[0].File("timestamp.txt"), source + [env.Value(args)]


def score_results(target, source, env):
    """
    """    
    ctm_path = source[0].rstr()
    transcript = source[1].rstr()
    out_path = os.path.dirname(target[0].rstr())

    # Get a list of IDs from the reference.  All must appear in the CTM output
    spkD = set()
    with codecs.open(transcript, "rb", encoding="utf-8") as f:
        for line in f:
            if line.startswith(";;"):
                continue
            spkD.add(line.split()[0])

    # skip eval data
    isEval = re.compile("/eval/")

    # Merge and clean up CTM
    skipD = frozenset([u"~SIL", u"<s>", u"</s>", u"<HES>", u"<hes>"])
    ctmL = []
    for file_ in glob(pjoin(ctm_path, "*.ctm")):
        with codecs.open(file_, "rb", encoding="utf-8") as ctmF:
            for line in ctmF:
                uttid, pcm, beg, dur, token = line.split()
                if isEval.search(pcm):
                    continue
                token = token[:-4]
                if token in skipD:
                    continue
                idx = uttid.find("#")
                spk = uttid[:idx]
                spkD.discard(spk)
                ctmL.append((spk, float(beg), dur, token))
    ctmL.sort()

    # add in missing speakers
    for spk in spkD:
        bisect.insort(ctmL, (spk, 0.0, "0.0", "@"))

    with codecs.open(pjoin(out_path, "all.ctm"), "wb", encoding="utf-8") as outF:
        for ctm in sorted(ctmL):
            outF.write("%s 1 %7.3f %s %s\n" % ctm)

    args = {"SCLITE" : env["SCLITE_BINARY"],
            "TRANSCRIPT" : transcript,
            "TRANSCRIPT_FORMAT" : "stm",
            "HYPOTHESIS" : os.path.abspath(pjoin(out_path, "all.ctm")),
            "HYPOTHESIS_FORMAT" : "ctm",
            "ENCODING" : "utf-8",
            "OUTPUT_NAME" : "babel",
            "OUTPUT_ROOT" : os.path.abspath(out_path),
            "OUTPUT_TYPES" : "all dtl sgml",
            }

    # Run scoring
    cmd =env.subst("%(SCLITE)s -r %(TRANSCRIPT)s %(TRANSCRIPT_FORMAT)s -O %(OUTPUT_ROOT)s -h %(HYPOTHESIS)s %(HYPOTHESIS_FORMAT)s -n %(OUTPUT_NAME)s -o %(OUTPUT_TYPES)s -e %(ENCODING)s -D -F" % args)
    out, err, success = run_command(cmd)
    if not success:
        return out + err
    return None


def score_emitter(target, source, env):
    new_targets = [pjoin(target[0].rstr(), x) for x in ["babel.sys", "all.ctm", "babel.dtl", "babel.pra", "babel.raw", "babel.sgml"]]
    return new_targets, source


def collate_results(target, source, env):
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\t".join(["Exp", "Lang", "Pack", "Vocab", "Pron", "LM", "Sub", "Del", "Ins", "Err", "SErr"]) + "\n")
        for fname in [x.rstr() for x in source]:
            expname, language, pack, vocab, pron, lm = fname.split("/")[1:-3]
            with meta_open(fname) as ifd:
                spk, snt, wrd, corr, sub, dele, ins, err, serr = [re.split(r"\s+\|?\s*", l) for l in ifd if "aggregated" in l][0][1:-1]
            ofd.write("\t".join([expname, language, pack, vocab, pron, lm, sub, dele, ins, err, serr]) + "\n")
    return None


def split_expansion(target, source, env):
    if len(source) == 2:
        limit = source[1].read()
    else:
        limit = 0
    words = {}
    with meta_open(source[0].rstr()) as ifd:
        for l in ifd:
            toks = l.split("\t")
            assert(len(toks) == len(target) + 1)
            words[toks[0]] = [Probability(neglogprob=float(x)) for x in toks[1:]]
    for i, f in enumerate(target):
        with meta_open(f.rstr(), "w") as ofd:
            vals = [(z[0], -z[1][i].log()) for z in sorted(words.iteritems(), lambda x, y : cmp(y[1][i].log(), x[1][i].log()))]
            if limit > 0:
                vals = vals[0:limit]
            ofd.write("\n".join(["%s\t%f" % (w, p) for w, p in vals]))
    return None


def split_expansion_emitter(target, source, env):
    new_targets = [pjoin(env["BASE_PATH"], "%s.gz" % x) for x in ["morph", "lm", "lm_avg", "lm_morph"]]
    return new_targets, source


def transcripts_to_vocabulary(target, source, env):
    word_counts = FrequencyList()
    for fname in source:
        with meta_open(fname.rstr()) as ifd:
            for line in [x for x in ifd if not re.match(r"^\[.*\]\s*", x)]:
                for tok in line.split():
                    word_counts[tok] = word_counts.get(tok, 0) + 1
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(word_counts.format())
    return None


def split_train_dev(target, source, env):
    data_path = source[0].rstr()
    for type, out_fname in zip(["training", "sub-train", "dev"], target):
        with meta_open(out_fname.rstr(), "w") as ofd:
            for in_fname in env.Glob(pjoin(data_path, "*", type, "transcription", "*")):
                with meta_open(in_fname.rstr()) as ifd:
                    ofd.write(ifd.read())
    return None


def pronunciations_from_probability_list(target, source, env):
    with meta_open(source[0].rstr()) as pl_fd:
        pass
    return None


def top_words(target, source, env):
    args = source[-1].read()
    with meta_open(source[0].rstr()) as words_ifd, meta_open(source[1].rstr()) as pron_ifd:
        top = ProbabilityList(words_ifd).get_top_n(args["COUNT"])
        prons = Pronunciations(pron_ifd)
        prons.filter_by(top)
    with meta_open(target[0].rstr(), "w") as words_ofd, meta_open(target[1].rstr(), "w") as pron_ofd:
        words_ofd.write(top.format())
        pron_ofd.write(prons.format())
    return None


def pronunciation_performance(target, source, env):
    with meta_open(source[0].rstr()) as gold_fd, meta_open(source[1].rstr()) as gen_fd:
        tp, fp, fn = 0, 0, 0
        gold = Pronunciations(gold_fd)
        gen = Pronunciations(gen_fd)
        logging.info("gold phone inventory: %s", " ".join(gold.phones()))
        logging.info("generated phone inventory: %s", " ".join(gen.phones()))
        for x in gen.get_words().intersection(gold.get_words()):
            gold_prons = set(map(tuple, [map(str.lower, y) for y in gold[x].values()]))
            gen_prons = set(map(tuple, [map(str.lower, y) for y in gen[x].values()]))            
            for go_p in gold_prons:
                if go_p in gen_prons:
                    tp += 1
                else:
                    fn += 1
            for ge_p in gen_prons:
                if ge_p not in gold_prons:
                    fp += 1
        prec = float(tp) / (tp + fp)
        rec = float(tp) / (tp + fn)
        f = 2 * (prec * rec) / (prec + rec)
        with meta_open(target[0].rstr(), "w") as ofd:
            ofd.write("%f %f %f\n" % (prec, rec, f))
    return None


class CFG():
    dictFile = '${PRONUNCIATIONS_FILE}'
    vocab = '${VOCABULARY_FILE}'
    lm = '${LANGUAGE_MODEL_FILE}'
    dbFile = '${DATABASE_FILE}'
    graph = '${GRAPH_FILE}'    
    psFile = '${PHONE_FILE}'
    pssFile = '${PHONE_SET_FILE}'
    tagsFile = '${TAGS_FILE}'
    treeFile = '${TREE_FILE}'
    topoFile = '${TOPO_FILE}'
    ttreeFile = '${TOPO_TREE_FILE}'
    samplingrate = '${SAMPLING_RATE}'
    featuretype = '${FEATURE_TYPE}'
    pcmDir = '${PCM_PATH}'
    melFile = '${MEL_FILE}'
    warpFile = '${WARP_FILE}'
    ldaFile = '${LDA_FILE}'
    acweight = '${ACOUSTIC_WEIGHT}'
    ctmDir = '${CTM_PATH}'
    latDir = '${LATTICE_PATH}'
    useDispatcher = True
    priors = '${PRIORS_FILE}'
    misc.errorMax = 150000
    errorHandler.setVerbosity(errorHandler.ERROR_LOG)
    nn = None
    def __str__(self):
        return "\n".join(["%s = %s" % (k, getattr(self, k)) for k in dir(self) if not k.startswith("_")])
    def __init__(self, env, **args):
        for k in [x for x in dir(self) if not x.startswith("_")]:
            current = getattr(self, k)
            if isinstance(current, basestring) and "FILE" in current:
                try:
                    setattr(self, k, os.path.abspath(env.Glob(env.subst(current))[0].rstr()))
                except:
                    setattr(self, k, os.path.abspath(env.subst(current)))
            else:
                setattr(self, k, env.subst(current))

        try:
            self.acweight = float(self.acweight)
        except:
            self.acweight = .09
        self.db = dbase.DB(dirFn=dbase.getFlatDir)

        self.fe = frontend.FeCombo(self.db, int(self.samplingrate), self.featuretype)
        self.fe.ctx2           = frontend.FeCTX([self.fe.fmllr])
        self.fe.ctx2.spliceN   = 4
        self.fe.ctx2.db        = self.db
        self.fe.end            = self.fe.ctx2
        self.fe.pcm.pcmDir = self.pcmDir
        self.fe.norm.normDir = env.subst('${CMS_PATH}')
        self.fe.fmllr.fmllrDir = env.subst('${FMLLR_PATH}')
        self.fe.norm.normMode = 1

        self.layerL = []
        for i in range(6):
            self.l = nnet.LayerWeights()
            self.l.name = 'layer%d'%i
            self.l.isTrainable = False
            self.l.initWeightFile = env.subst('${MODEL_PATH}/layer%d') % i
            self.layerL.append(self.l)
            if i < 5:
                self.l = nnet.LayerSigmoid()
                self.l.name = 'layer%d-nonl' % i
                self.layerL.append(self.l)
        self.layerL[-1].matrixOut = True
        self.nn = nnet.NeuralNet(layerL=self.layerL, depL=[self.fe.end])
        self.nn.db = self.db

        
def asr_construct(target, source, env):
    #
    # based on Zulu LLP
    #
    vocabulary, pronunciations, language_model = source
    env.Replace(VOCABULARY_FILE=vocabulary, PRONUNCIATIONS_FILE=pronunciations, LANGUAGE_MODEL_FILE=language_model, ACOUSTIC_WEIGHT=.09)
    cfg = CFG(env)
    se = dsearch.Decoder(lmType=32)
    se.build(cfg)
    se.dnet.write(target[0].rstr())
    return None


def asr_test(target, source, env):
    #
    # based on Zulu LLP
    #
    dnet, vocabulary, pronunciations, language_model, args = source
    args = args.read()
    out_path, tail = os.path.split(os.path.dirname(target[0].rstr()))
    env.Replace(VOCABULARY_FILE=vocabulary.rstr(),
                PRONUNCIATIONS_FILE=pronunciations.rstr(),
                LANGUAGE_MODEL_FILE=language_model,
                CTM_PATH=os.path.join(out_path, "ctm"),
                LATTICE_PATH=os.path.join(out_path, "lat"),
                ACOUSTIC_WEIGHT=.09,                
                GRAPH_FILE=dnet,
    )
    cfg = CFG(env)
    
    #
    # from test.py
    #
    jid    = args.get("JOB_ID", 0)
    jnr    = int(env["ASR_JOB_COUNT"]) #args.get("JOB_COUNT", 1)
    genLat = True    
    cfg.useDispatcher = False
    
    # # ------------------------------------------
    # # Boot
    # # ------------------------------------------
    cfg.db.init(cfg.dbFile, 'utterance', cfg.useDispatcher, jid, jnr, chunkSize=5)
    cfg.fe.mel.readFilter(cfg.melFile)
    cfg.fe.mel.readWarp(cfg.warpFile)
    cfg.fe.lda.readLDA(cfg.ldaFile)    

    # # decoder
    se = dsearch.Decoder(speed=12,scale=cfg.acweight,lmType=32,genLat=genLat)
    se.initGraph(cfg)
    se.latBeam  = 7
    se.linkMax  = 700
    rescoreBeam = 2.0
    
    # # NN Scorer
    cfg.nn.configure()

    se.sc = NNScorer()
    se.dnet.scorer = se.sc
    se.sc.scale    = cfg.acweight
    se.sc.feat     = cfg.nn.feat
    se.sc.logInput = True
    se.sc.readPriors(cfg.priors)

    # # ------------------------------------------
    # # Main loop
    # # ------------------------------------------

    if genLat:
        misc.makeDir(cfg.latDir)
    with open(target[0].rstr(), "w") as ofd, open(target[1].rstr(), "w") as lattice_list_ofd:
        for utt in cfg.db:
            cfg.nn.eval(utt)
            se.search()
            key    = utt + ' ' + os.path.splitext(cfg.db.getFile(utt))[0]
            txt    = se.getHyp().strip()
            hyp    = se.getCTM(key, cfg.db.getFrom(utt))
            tscore = se.getScore()
            for c in hyp:
                print >>ofd,c
            if genLat:
                se.rescore(rescoreBeam)
                fname = os.path.abspath(os.path.join(cfg.latDir,"%s.fsm.gz" % utt))
                lattice_list_ofd.write("%s\n" % (fname))
                se.lat.write(fname, cfg.db.getFrom(utt))
    return None


def run_asr(env, root_path, vocabulary, pronunciations, language_model, *args, **kw):
    env.Replace(ROOT_PATH=root_path)
    dnet = env.ASRConstruct("${ROOT_PATH}/dnet.bin.gz", [vocabulary, pronunciations, language_model])
    tests = []
    for i in range(3): #env["ASR_JOB_COUNT"]):
        tests.append(env.ASRTest(["${ROOT_PATH}/ctm/%d.ctm" % i, "${ROOT_PATH}/lattice_list_%d.txt" % i],
                                 [dnet, vocabulary, pronunciations, language_model, env.Value({"JOB_ID" : i})]))
    return tests


def TOOLS_ADD(env):
    BUILDERS = {"ASRConstruct" : Builder(action=asr_construct),
                "ASRTest" : Builder(action=asr_test),
                "IBMTrainLanguageModel" : Builder(action=ibm_train_language_model),
                }
    env.Append(BUILDERS=BUILDERS)
    env.AddMethod(run_asr, "RunASR")
