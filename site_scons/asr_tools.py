from SCons.Builder import Builder
from SCons.Action import Action
from SCons.Subst import scons_subst
from SCons.Util import is_List
from SCons.Node.FS import Dir
from scons_tools import ThreadedBuilder, threaded_run
import re
from glob import glob
from functools import partial
import tarfile
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
from os.path import join as pjoin
#import matplotlib
#matplotlib.use("Agg")
#from matplotlib import pyplot
#import numpy
import sys
import user
import dsearch
import misc
import dbase
import frontend
import nnet
from attila import NNScorer, errorHandler, Act_Rectified, Act_Sigmoid, Act_ID, MatrixCU, Act_Tanh, Act_Softmax, HMM
from scons_tools import run_command

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
    mlpFile = '${MLP_FILE}'
    acweight = '${ACOUSTIC_WEIGHT}'
    ctmDir = '${CTM_PATH}'
    latDir = '${LATTICE_PATH}'
    useDispatcher = False
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

            
def train_language_model(target, source, env):
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


def asr_construct(target, source, env):
    #
    # based on Zulu LLP
    #
    vocabulary, pronunciations, language_model = source
    env.Replace(VOCABULARY_FILE=vocabulary, PRONUNCIATIONS_FILE=pronunciations, LANGUAGE_MODEL_FILE=language_model) #, ACOUSTIC_WEIGHT=.060)    
    cfg = CFG(env)
    #if env.subst("${BABEL_ID}") == "201":
    #    cfg.treeFile = env.subst("${MODEL_PATH}/tree2")
    se = dsearch.Decoder(lmType=32)
    se.build(cfg)
    se.dnet.write(target[0].rstr())
    return None

def asr_test(target, source, env):
    #
    # based on Zulu LLP
    #
    dnet, vocabulary, pronunciations, language_model = source
    #args = args.read()
    out_path, tail = os.path.split(os.path.dirname(target[0].rstr()))
    env.Replace(VOCABULARY_FILE=vocabulary.rstr(),
                PRONUNCIATIONS_FILE=pronunciations.rstr(),
                LANGUAGE_MODEL_FILE=language_model,
                GRAPH_FILE=dnet,
    )
        
    cfg = CFG(env)
    postThresh = 1e-04

    mlpFile = env.maybe(env.subst("${MLP_FILE}"))
    melFile = env.maybe(env.subst("${MEL_FILE}"))
    warpFile = env.maybe(env.subst("${WARP_FILE}"))
    ldaFile = env.maybe(env.subst("${LDA_FILE}"))
    priorsFile = env.maybe(env.subst("${PRIORS_FILE}"))

    mlp = os.path.exists(cfg.mlpFile) and "weights.mlp" in mlpFile
    nmlp = os.path.exists(cfg.mlpFile) and "weights.mlp" not in mlpFile
    layer = os.path.exists(env.subst("${MODEL_PATH}/layer0"))


    db = dbase.DB(dirFn=dbase.getFlatDir)
    fe = frontend.FeCombo(db, int(env["SAMPLING_RATE"]), env["FEATURE_TYPE"])
    fe.end            = fe.fmllr
    fe.pcm.pcmDir     = cfg.pcmDir
    fe.pcm.readMode   = 'speaker'
    fe.norm.normMode  = 1
    fe.norm.normDir   = env.subst("${CMS_PATH}") #cfg.cms'cms/'
    fe.fmllr.fmllrDir = env.subst("${FMLLR_PATH}") #'fmllr/'
    
    #
    # from test.py
    #
    jid    = int(env["JOB_ID"])
    jnr    = int(env["JOB_COUNT"])
    genLat = True
    genCons = True
    writeLat = False
    writeCons = True
    cfg.useDispatcher = False
    if nmlp:
        chunkSize = 10
    else:
        chunkSize = 5
    acweight = float(env.subst("${ACOUSTIC_WEIGHT}"))
    db.init(cfg.dbFile, 'utterance', False, jid, jnr, chunkSize=chunkSize)

    fe.mel.readFilter(melFile)
    fe.mel.readWarp(warpFile)
    fe.lda.readLDA(ldaFile)    

    se = dsearch.Decoder(speed=12, scale=acweight, lmType=32, genLat=genLat)
        
    se.initGraph(cfg)
    se.latBeam  = 7
    se.linkMax  = 700

    if mlp:
        fe.ctx2           = frontend.FeCTX([fe.fmllr])
        fe.ctx2.spliceN   = 4
        fe.ctx2.db        = db
        fe.mlp.depL       = [fe.ctx2]
        fe.mlp.db         = db
        fe.end            = fe.mlp
        fe.mlp.mlp.read(mlpFile)

        fe.mlp.mlp.layerL[0].afct = Act_Rectified()
        fe.mlp.mlp.layerL[1].afct = Act_Rectified()
        fe.mlp.mlp.layerL[2].afct = Act_Rectified()
        fe.mlp.mlp.layerL[3].afct = Act_Sigmoid()
        fe.mlp.mlp.layerL[4].afct = Act_ID()

        se.sc = NNScorer()
        se.dnet.scorer = se.sc
        se.sc.scale    = acweight
        se.sc.feat     = fe.end.feat
        se.sc.logInput = True
        se.sc.readPriors(priorsFile)
        
        pass
    elif layer:
        fe.ctx2           = frontend.FeCTX([fe.fmllr])
        fe.ctx2.spliceN   = 4
        fe.ctx2.db        = db
        fe.end            = fe.ctx2
        layerL = []
        for i in range(6):
            l = nnet.LayerWeights()
            l.name = 'layer%d'%i
            l.isTrainable = False
            l.initWeightFile = env.subst('${MODEL_PATH}/layer%d') % i
            layerL.append(l)
            if i < 5:
                l = nnet.LayerSigmoid()
                l.name = 'layer%d-nonl' % i
                layerL.append(l)
        layerL[-1].matrixOut = True
        nn = nnet.NeuralNet(layerL=layerL, depL=[fe.end])
        nn.db = db
        nn.configure()

        se.sc = NNScorer()
        se.dnet.scorer = se.sc
        se.sc.scale    = acweight
        se.sc.feat     = nn.feat
        se.sc.logInput = True
        se.sc.readPriors(priorsFile)
    elif nmlp:
        se.initAM(cfg)    
        mlp      = fe.mlp.mlp
        mlp.feat = MatrixCU()
        sigmoid  = Act_Sigmoid()
        tanh     = Act_Tanh()
        actid    = Act_ID()
        softmax  = Act_Softmax()
        softmax.logOutput = True

        mlp.read(mlpFile)
        for layerX in range(mlp.layerL.size()):
            mlp.layerL[layerX].afct = sigmoid
        mlp.layerL[-1].afct = actid

        se.sc = NNScorer()
        se.dnet.scorer = se.sc
        se.sc.scale    = acweight
        se.sc.logInput = True
        se.sc.feat     = mlp.layerL[-1].Y.mat
        se.sc.readPriors(priorsFile)

        se.latBeam  = 6.5
        se.linkMax  = 700

        binThresh         = 1.0e-10
        writeSIL          = 0

        totUtt    = 0
        totArc    = 0
        totNonSil = 0
        totDur    = 0.0
        totDens   = 0.0
    else:
        return "Don't know how to run ASR with these models!"

    with meta_open(target[0].rstr(), "w") as ctm_ofd, tarfile.open(target[1].rstr(), "w|gz") as tf_ofd, temp_file() as temp_fname:        
        for utt in db:

            key    = utt + ' ' + os.path.splitext(db.getFile(utt))[0]
            if mlp:
                fe.end.eval(utt)
            elif nmlp:
                fe.end.eval(utt)
                #se.voc.lmap[se.voc.pronL.index("<s>(01)")] = se.voc.wordL.index("<s>")
                #se.voc.lmap[se.voc.pronL.index("</s>(01)")] = se.voc.wordL.index("</s>")
            else:
                nn.eval(utt)

            se.search()

            txt    = se.getHyp().strip()
            hyp    = se.getCTM(key, db.getFrom(utt))
            tscore = se.getScore()
            for c in hyp:
                ctm_ofd.write("%s\n" % (c))
            se.rescore(env["RESCORE_BEAM"])
            with meta_open(temp_fname, "w") as ofd:
                pass
            if writeLat:
                
                #fname = os.path.abspath(os.path.join(cfg.latDir,"%s.fsm.gz" % utt))
                fname = "%s.fsm" % (utt)
                #lattice_list_ofd.write("%s\n" % (fname))
                se.lat.write(temp_fname, db.getFrom(utt))
            elif writeCons:
                fname = "%s.cons" % (utt)
                #fname = os.path.abspath(os.path.join(cfg.latDir, "%s.cons.gz" % utt))
                #rescoreBeam = 2.0
                #se.rescore(rescoreBeam)
                arcN = len(se.lat.arcs)
                durS = db.getTo(utt)- db.getFrom(utt)
                dens = arcN / durS
                ##se.voc.lmap[se.voc.pronL.index("<s>(01)")] = se.voc.optWordX
                ##se.voc.lmap[se.voc.pronL.index("</s>(01)")] = se.voc.optWordX
                # if (se.lat.arcs.size() < 100000):
                #     postThresh=1e-08
                # elif (se.lat.arcs.size() < 500000):
                #     postThresh=1e-06
                # else:
                #     postThresh=1e-05

                se.consensus(postThresh)
                binThresh         = 1.0e-10
                writeSIL          = 0
                #lattice_list_ofd.write("%s\n" % (fname))
                se.cons.write(temp_fname, db.getFrom(utt), binThresh, writeSIL)
            tf_ofd.add(temp_fname, arcname=fname)
        tf_ofd.close()
    return None

def run_asr(env, root_path, vocabulary, pronunciations, language_model, *args, **kw):
    env.Replace(ROOT_PATH=root_path)
    if env["RUN_ASR"]:
        dnet = env.ASRConstruct("${ROOT_PATH}/dnet.bin.gz", [vocabulary, pronunciations, language_model], PACK=env["PACK"], BABEL_ID=env["BABEL_ID"], LANGUAGE_NAME=env["LANGUAGE_NAME"])    
        tests = [dnet]
        if env["TEST_ASR"]:
            to = 1
        else:
            to = env["JOB_COUNT"]
        for i in range(to):
            tests.append(env.ASRTest(["${ROOT_PATH}/transcripts_${JOB_ID + 1}_of_${JOB_COUNT}.ctm.gz",
                                      "${ROOT_PATH}/confusion_networks_${JOB_ID + 1}_of_${JOB_COUNT}.tgz"],
                                     [dnet, vocabulary, pronunciations, language_model], PACK=env["PACK"], BABEL_ID=env["BABEL_ID"], JOB_ID=i, LANGUAGE_NAME=env["LANGUAGE_NAME"]))
    else:
        return [None] + [[None, x] for x in env.Glob("${ROOT_PATH}/*_[0-9]*_of_[0-9]*.tgz")]
    return tests

def TOOLS_ADD(env):
    BUILDERS = {"ASRConstruct" : Builder(action=asr_construct),
                "ASRTest" : Builder(action=asr_test),
                "TrainLanguageModel" : Builder(action=train_language_model),
                }
    env.Append(BUILDERS=BUILDERS)
    env.AddMethod(run_asr, "RunASR")
