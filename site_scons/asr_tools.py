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

def graph_file(target, source, env):
    vocabulary_file, pronunciations_file, language_model_file = source
    env.Replace(VOCABULARY_FILE=vocabulary_file.rstr())
    env.Replace(PRONUNCIATIONS_FILE=pronunciations_file.rstr())
    env.Replace(LANGUAGE_MODEL_FILE=language_model_file.rstr())
    
    sys.path.append(env.subst("${ATTILA_PATH}/tools/attila/pylib"))
    sys.path.append(env.subst("${ATTILA_PATH}/tools/attila/bin/linux64/"))

    from attila import errorHandler, HMM
    import dsearch
    import misc
    import dbase
    import frontend
    import misc

    misc.errorMax = 150000
    errorHandler.setVerbosity(errorHandler.INFO_LOG)
    cfg = CFG(env)
    db = dbase.DB(dirFn=dbase.getFlatDir)
    fe = frontend.FeCombo(db, int(cfg.samplingrate), cfg.featuretype)
    fe.end = fe.fmmi
    fe.pcm.pcmDir = cfg.pcmDir
    fe.norm.normMode = 1
    fe.norm.normDir = '${CMS_PATH}'
    fe.fmllr.fmllrDir = '${FMLLR_PATH}'
    HMM.silWord = '~SIL'
    HMM.bosWord = '<s>'
    HMM.eosWord = '</s>'
    HMM.variants = True
    HMM.skipBnd = False
    HMM.silProb = 1.0
    HMM.leftContext = 2
    HMM.rightContext = 2
    HMM.leftXContext = 2
    HMM.rightXContext = 2
    HMM.wordBoundaryPhone = '|'
    se = dsearch.Decoder(lmType=32)
    se.build(cfg)
    se.dnet.write(target[0].rstr())
    return None

def vtln(target, source, env):
    sys.path.append(env.subst("${ATTILA_PATH}/tools/attila/pylib"))
    sys.path.append(env.subst("${ATTILA_PATH}/tools/attila/bin/linux64/"))
    #from attila import *
    #import vcfg as cfg
    import aio
    import train
    import misc
    import dbase
    import frontend
    
    cfg = CFG(env)

    for job_id in range(env["JOB_COUNT"]):
        
        db = dbase.DB(dirFn=dbase.getFlatDir)

        db.init(cfg.dbFile, 'speaker', cfg.useDispatcher, job_id, env["JOB_COUNT"], chunkSize=1)
        continue
        # frontend
        fe = frontend.FeCombo(db, int(cfg.samplingrate), cfg.featuretype)
        fe.mel.readFilter(cfg.melFile)
        fe.lda.readLDA(cfg.ldaFile)
        fe.fmmi.init(cfg.trFile, cfg.trfsFile, cfg.ctxFile, cfg.ictx, cfg.octx)

        continue
        # Transcripts
        tx    = aio.ConfRef()
        tx.db = db
        tx.txtDir = cfg.txtDir

        # Trainer
        tr = train.Trainer()
        tr.initAM  (cfg)
        tr.initDict(cfg)
        tr.initTree(cfg)
        tr.initHMM ()
        tr.db = db
        tr.gs.setFeat(fe.end.feat)

        # get a set of states that correspond to phones we skip in VTLN
        idx  = IVector()
        silD = set()
        for topo in tr.tset:
            if topo.name not in cfg.silphoneL:
                continue
            for state in topo:
                tr.tree.getModelSet(idx,state.root)
                for i in range(idx.dimN):
                    silD.add(idx[i])
        # Voicing Model
        cfg.gsFile   = cfg.vtlgsFile
        cfg.msFile   = cfg.vtlmsFile
        cfg.treeFile = cfg.vtltreeFile
        warpLst      = range(len(fe.mel.melBank))
        vfe          = Feature()
        vtr = train.Trainer()
        vtr.initAM  (cfg)
        vtr.shareDict(tr)
        vtr.initTree(cfg)
        vtr.initHMM ()
        vtr.gs.setFeat(vfe)

        vdir = os.path.split(cfg.warpFile)[0]
        if vdir and not os.path.isdir(vdir):
            try:
                os.makedirs(vdir)
            except:
                if not os.path.isdir(vdir):
                    raise

        # ------------------------------------------
        # getNorm : speech based cms
        # ------------------------------------------

        # global arrays
        cmsA   = {}
        scA    = {}
        plpA   = {}
        scoreA = {}

        def getNorm(spk):
            global cmsA,scA,plpA,scoreA
            wght   = FVector()
            uttL   = db.getUtts(spk)
            plpA   = {}
            cmsA   = {}
            scA    = {}
            scoreA = {}
            for warp in warpLst:
                cmsA[warp]   = FMatrix()
                scA[warp]    = Scatter()
                scoreA[warp] = 0
            for utt in uttL :
                for warp in warpLst:
                    fe.mel.utt = 'unknown'
                    fe.plp.utt = 'unknown'            
                    fe.mel.warp = warp
                    try:
                        fe.plp.eval(utt)
                    except:
                        misc.error('getNorm','frontend error','%s warp= %d' % (utt, warp))
                        continue
                    plpA[utt,warp] = Feature()
                    plpA[utt,warp].copy(fe.plp.feat)
                    plpA[utt,warp].mean(cmsA[warp])

        # ------------------------------------------
        # getScore
        # ------------------------------------------

        def getScore(spk):
            global cmsA,scA,plpA,scoreA
            uttL = db.getUtts(spk)
            wght = FVector()    
            for utt  in uttL :
                try:
                    ref = tx.get(utt)               
                    fe.plp.eval(utt)
                    vtr.buildHMM(ref)
                except:
                    misc.error('getScore','accumulation error',utt)            
                    continue
                path = tr.pbox[utt]
                if len(path) == 0 :
                    continue
                path.align(vtr.hmm.sg)
                frameN = len(path)
                wght.resize(frameN)
                wght.setConst(1.0)
                for frameX in range(frameN):
                    if path[frameX][0].obsX in silD:
                        wght[frameX] = 0.0
                for warp in warpLst:
                    vfe.norm(plpA[utt,warp],cmsA[warp],CVN_NORM)
                    scA[warp].accu(vfe,wght)
                    path.update(vtr.sc)
                    for frameX in range(frameN):
                        if path[frameX][0].obsX in silD:
                            continue
                        scoreA[warp] += path[frameX][0].score
                        if frameX > 0:
                            scoreA[warp] -= path[frameX-1][0].score

        # ------------------------------------------
        # findWarp
        # ------------------------------------------

        def findWarp(spk):
            global cmsA,scA,plpA,scoreA
            bestS = 1e+20
            bestW = 10
            for warp in warpLst:
                if scA[warp].cnt < cfg.vtlnMinCount:
                    misc.warn('findWarp','insufficient counts, backing' \
                              ' off to no warping ','spk= %s'%spk)
                    break
                l = scoreA[warp] / scA[warp].cnt
                s = l-0.5*tr.sc.scale*scA[warp].logDet()
                if s < bestS:
                    bestS = s
                    bestW = warp
            if bestS < 0:
                misc.warn('findWarp','best Warp','spk= %s warp= %d score= %2.2f (will reset)'%(spk,bestW,bestS))
                bestW = 10
            misc.info('findWarp','best Warp','spk= %s warp= %d score= %2.2f'%(spk,bestW,bestS))
            return bestW

        # ------------------------------------------
        # VTLN Estimation
        # ------------------------------------------

        def process(spk,vtlnFile):
            uttL = db.getUtts(spk)
            tr.pbox.clear()
            # viterbi
            fe.mel.warp = fe.mel.nullWarp
            for utt in uttL:
                try:
                    ref  = tx.get(utt)
                    path = tr.mkPath(utt)
                    fe.end.eval(utt)
                    tr.buildHMM(ref)
                    tr.viterbi()
                except:
                    misc.error('main','accumulation error',utt)            
                    continue
            # estimate warp
            getNorm(spk)
            getScore(spk)
            warp = findWarp(spk)
            # write cms
            dir = db.getDir(spk,root=cfg.normDir,createDir=True)
            cmsA[warp].write(dir+spk+'.mat')
            # write vtln
            print >>vtlnFile,spk,warp
            return

        # ------------------------------------------
        # Main loop
        # ------------------------------------------

        vtlnFile = open(cfg.warpFile,'w')

        for spk in db:
            process(spk,vtlnFile)

        vtlnFile.close()

        
    return None

# def run_asr(target, source, env):

#     sys.path.append(env.subst("${ATTILA_PATH}/tools/attila/pylib"))
#     sys.path.append(env.subst("${ATTILA_PATH}/tools/attila/bin/linux64/"))

#     from attila import errorHandler, HMM, NNScorer
#     import dsearch
#     import misc
#     import dbase
#     import frontend
#     import misc
#     import nnet
    
#     cfg = CFG(env)

#     env.Replace(JOB_ID=0)
#     env.Replace(JOB_COUNT=1)
#     env.Replace(GENERATE_LATTICES=True)
#     jid = int(env.subst("${JOB_ID}"))
#     jnr = int(env.subst("${JOB_COUNT}"))
#     genLat = bool(env.subst("${GENERATE_LATTICES}"))

#     env.Replace(ACOUSTIC_WEIGHT=.01)
#     acweight = float(env.subst("${ACOUSTIC_WEIGHT}"))

#     # dbase
#     db = dbase.DB(dirFn=dbase.getFlatDir)
#     db.init(cfg.dbFile,'utterance',cfg.useDispatcher,jid,jnr,chunkSize=5)

#     # frontend
#     fe = frontend.FeCombo(db, int(cfg.samplingrate), cfg.featuretype)
#     fe.mel.readFilter(cfg.melFile)
#     fe.mel.readWarp(cfg.warpFile)
#     fe.lda.readLDA(cfg.ldaFile)

#     # decoder
#     se = dsearch.Decoder(speed=12,scale=acweight,lmType=32,genLat=genLat)
#     se.initGraph(cfg)

#     se.latBeam  = 7
#     se.linkMax  = 700
#     rescoreBeam = 2.0

#     # NN Scorer
#     layerL = []
#     for i in range(6):
#         l = nnet.LayerWeights()
#         l.name = 'layer%d'%i
#         l.isTrainable = False
#         l.initWeightFile = env.subst('${MODEL_PATH}/layer%d' % i)
#         layerL.append(l)
#         if i < 5:
#             l = nnet.LayerSigmoid()
#             l.name = 'layer%d-nonl' % i
#             layerL.append(l)
#     layerL[-1].matrixOut = True

#     nn    = nnet.NeuralNet(layerL=layerL,depL=[fe.end])
#     nn.db = db

#     nn.configure()


#     se.sc = NNScorer()
#     se.dnet.scorer = se.sc
#     se.sc.scale    = acweight
#     se.sc.feat     = nn.feat
#     se.sc.logInput = True
#     se.sc.readPriors(cfg.priors)

#     # ------------------------------------------
#     # Main loop
#     # ------------------------------------------

#     def process(utt,f):
#         print utt
#         nn.eval(utt)
#         return
#         se.search()
#         key    = utt + ' ' + os.path.splitext(db.getFile(utt))[0]
#         txt    = se.getHyp().strip()
#         hyp    = se.getCTM(key,db.getFrom(utt))
#         tscore = se.getScore()
#         print utt,'score= %.5f frameN= %d'%(tscore,se.dnet.state.frameN)
#         print utt,'words=',txt
#         for c in hyp: print >>f,c
#         if genLat:
#             se.rescore(rescoreBeam)
#             se.lat.write(cfg.latDir+utt+'.fsm.gz',db.getFrom(utt))
#         return

#     misc.makeDir(cfg.ctmDir)
#     if genLat:
#         misc.makeDir(cfg.latDir)

#     with open(env.subst("${CTM_PATH}/${JOB_ID}.ctm"), "w") as ofd:
#         for utt in db:
#             process(utt, ofd)

#     return None

# def run_asr_emitter(target, source, env):
#     return target, source

def pronunciations_to_vocabulary(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        d = Pronunciations(ifd)
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(d.format_vocabulary())
    return None


def appen_to_attila_old(target, source, env, for_signature):
    pron, pnsp, tag = target
    lexicons = source[0:-1]
    args = source[-1].read()
    base = "python data/makeBaseDict.py -d %s -p %s -t %s -l %s" % (pron, pnsp, tag, args["LOCALE"])
    if args.get("SKIP_ROMAN", False):
        base += " -s"
    return "%s %s" % (base, " ".join([x.rstr() for x in lexicons]))


def appen_to_attila(target, source, env):
    args = source[-1].read()

    # set the locale for sorting and getting consistent case
    locale.setlocale(locale.LC_ALL, args.get("LOCALE", "utf-8"))

    # convert BABEL SAMPA pronunciations to attila format
    #
    # In this version the primary stress ("), secondary stress (%),
    # syllable boundary (.), and word boundary within compound (#) marks
    # are simply stripped out.  We may want to try including this
    # information in the form of tags in some variants.
    #
    def attilapron(u, pnsp):
        skipD = frozenset(['"', '%', '.', '#'])
        phoneL = []
        for c in u.encode('utf-8').split():
            if c in skipD:
                continue
            #c = cfg.p2p.get(c,c) TODO
            pnsp.add(c)
            phoneL.append(c)
        phoneL.append('[ wb ]')
        if len(phoneL) > 2:
            phoneL.insert(1, '[ wb ]')
        return ' '.join(phoneL)

    # Pronunciations for the BABEL standard tags, silence, and the start
    # and end tokens
    nonlex = [ ['<UNINTELLIGIBLE>', set(['REJ [ wb ]'])],
               ['<FOREIGN>',   set(['REJ [ wb ]'])],
               ['<LAUGH>',     set(['VN [ wb ]'])],
               ['<COUGH>',     set(['VN [ wb ]'])],
               ['<BREATH>',    set(['NS [ wb ]', 'VN [ wb ]'])],
               ['<LIPSMACK>',  set(['NS [ wb ]'])],
               ['<CLICK>',     set(['NS [ wb ]'])],
               ['<RING>',      set(['NS [ wb ]'])],
               ['<DTMF>',      set(['NS [ wb ]'])],
               ['<INT>',       set(['NS [ wb ]'])],
               ['<NO-SPEECH>', set(['SIL [ wb ]'])],
               ['~SIL',        set(['SIL [ wb ]'])], 
               ['<s>',         set(['SIL [ wb ]'])],
               ['</s>',        set(['SIL [ wb ]'])], ]

    # Get the right dictionaries
    dictL = [x.rstr() for x in source[:-1]]

    # read in the dictionaries, normalizing the case of the word tokens to
    # all lowercase.  Normalize <hes> to <HES> so the LM tools don't think
    # it is an XML tag.
    voc = {}
    pnsp = set()
    for name in dictL:
        with codecs.open(name,'rb',encoding='utf-8') as f:
            for line in f:
                pronL = line.strip().split(u'\t')
                token = pronL.pop(0).lower()
                if args.get("SKIP_ROMAN", False):
                    pronL.pop(0)
                if token == '<hes>':
                    token = '<HES>'
                prons = voc.setdefault(token, set())
                prons.update([attilapron(p,pnsp) for p in pronL])

    # need a collation function as a workaround for a Unicode bug in
    # locale.xtrxfrm (bug is fixed in Python 3.0)
    def collate(s):
        return locale.strxfrm(s.encode('utf-8'))

    odict, opnsp, otags = [x.rstr() for x in target]

    # write the pronunciations, and collect the phone set
    with open(odict, 'w') as f:
        for token in sorted(voc.iterkeys(),key=collate):
            for pronX, pron in enumerate([x for x in voc[token]]):
                f.write("%s(%02d) %s\n" % (token.encode('utf-8'), 1+pronX, pron))
                
        for elt in nonlex:
            token = elt[0]
            for pronX, pron in enumerate(elt[1]):
                f.write("%s(%02d) %s\n" % (token, 1+pronX, pron))

    # generate and write a list of phone symbols (pnsp)
    with open(opnsp, 'w') as f:
        for pn in sorted(pnsp):
            f.write("%s\n" % pn)
        f.write("\n".join(["SIL", "NS", "VN", "REJ", "|", "-1"]) + "\n")

    # generate and write a list of tags
    with open(otags,'w') as f:
        f.write("wb\n")
    return None


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


# def collect_text(target, source, env):
#     words = set()
#     with meta_open(target[0].rstr(), "w") as ofd:
#         for dname in source:
#             for fname in glob(os.path.join(dname.rstr(), "*.txt")) + glob(os.path.join(dname.rstr(), "*.txt.gz")):
#                 with meta_open(fname) as ifd:
#                     for line in ifd:
#                         if not line.startswith("["):
#                             toks = []
#                             for x in line.lower().split():
#                                 if x == "<hes>":
#                                     toks.append("<HES>")
#                                 elif not x.startswith("<"):
#                                     toks.append(x)
#                             for t in toks:
#                                 words.add(t)
#                             if len(toks) > 0:
#                                 ofd.write("%s </s>\n" % (" ".join(toks)))
#     with meta_open(target[1].rstr(), "w") as ofd:
#         ofd.write("# BOS: <s>\n# EOS: </s>\n# UNK: <UNK>\n<s>\n</s>\n<UNK>\n")
#         ofd.write("\n".join(sorted(words)) + "\n")                                      
#     return None


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

#from ctypes import cdll
#l = [cdll.LoadLibrary("/home/tom/projects/babel_data/VT-2-5-babel/tools/attila/bin/linux64/intel64/libmkl_mc.so"),
#     cdll.LoadLibrary("/home/tom/projects/babel_data/VT-2-5-babel/tools/attila/bin/linux64/intel64/libmkl_def.so"),
#     cdll.LoadLibrary("/home/tom/projects/babel_data/VT-2-5-babel/tools/attila/bin/linux64/intel64/libmkl_core.so"),
#     cdll.LoadLibrary("/home/tom/projects/babel_data/VT-2-5-babel/tools/attila/bin/linux64/intel64/libmkl_intel_lp64.so"),
#     cdll.LoadLibrary("/home/tom/projects/babel_data/VT-2-5-babel/tools/attila/bin/linux64/libattila.so")
# ]


def cfg(env, **args):
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


# def replace_probabilities(target, source, env):
#     """
#     Takes a probability list and a language model, and creates a new probability list
#     where each word has the probability from the language model, for overlapping words.

#     Either the unspecified word probabilities are scaled (False), or all word probabilities 
#     are scaled (True), such that the unigram probabilities sum to one.
#     """
#     with meta_open(source[0].rstr()) as pl_fd, meta_open(source[1].rstr()) as lm_fd:
#         pass
#     return None


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


def run_asr_experiment(target, source, env):
    args = source[-1].read()
    #construct_command = env.subst("${ATTILA_INTERPRETER} ${SOURCES[1].abspath}", source=source)
    #out, err, success = run_command(construct_command)
    #if not success:
    #    return out + err
    #command = env.subst("${ATTILA_INTERPRETER} ${SOURCES[2].abspath} -n ${LOCAL_JOBS_PER_SCONS_INSTANCE} -j %d -w ${ACOUSTIC_WEIGHT} -l 1", source=source)
    #procs = [subprocess.Popen(shlex.split(command % i)) for i in range(env["LOCAL_JOBS_PER_SCONS_INSTANCE"])]
    #for p in procs:
    #    p.wait()
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


def plot_probabilities(target, source, env):
    p = ProbabilityList(meta_open(source[0].rstr()))
    ps = sorted([x.prob() for x in p.values()])
    pyplot.plot(ps)
    pyplot.savefig(target[0].rstr())
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


def plot_reduction(target, source, env):
    args = source[-1].read()
    bins = args["bins"]
    with meta_open(source[-3].rstr()) as in_voc_fd, meta_open(source[-2].rstr()) as all_voc_fd:
        in_vocabulary = FrequencyList(in_voc_fd).make_conservative()
        other_vocabulary = FrequencyList(all_voc_fd).make_conservative()
        all_vocabulary = other_vocabulary.join(in_vocabulary)
        out_of_vocabulary = set([x for x in all_vocabulary.keys() if x not in in_vocabulary])
        num_iv_types = len(in_vocabulary)
        num_iv_tokens = sum([all_vocabulary.get(x, 0) for x in in_vocabulary])
        num_types = len(all_vocabulary)
        num_tokens = sum(all_vocabulary.values())
        num_oov_types = num_types - num_iv_types
        num_oov_tokens = num_tokens - num_iv_tokens
        logging.info("%d/%d in-vocabulary types", num_iv_types, num_types)
        logging.info("%d/%d in-vocabulary tokens", num_iv_tokens, num_tokens)
        #pyplot.figure(figsize=(8 * 2, 7))
        pyplot.figure(figsize=(8, 7))
        for expansion_fname in source[0:-3]:
            good_tokens = 0
            good_types = 0
            token_based = numpy.empty(shape=(bins + 1))
            type_based = numpy.empty(shape=(bins + 1))
            token_based[0] = float(0.0)
            type_based[0] = float(0.0)
            name = {"morph" : "just Morfessor",
                    "lm" : "reranking by ngrams",
                    "lm_avg" : "reranking by ngram average",
                    "lm_morph" : "reranking by boundary-ngrams",
                    }[os.path.splitext(os.path.basename(expansion_fname.rstr()))[0]]            
            method = os.path.dirname(expansion_fname.rstr()).split("/")[-1]
            name = "%s - %s" % (method, name)

            with meta_open(expansion_fname.rstr()) as expansion_fd:
                expansions = [(w, p) for w, p in [x.strip().split() for x in expansion_fd]]
                bin_size = len(expansions) / bins
                for i in range(bins):
                    correct = [x for x in expansions[i*bin_size:(i+1)*bin_size] if x[0] in all_vocabulary]
                    good_types += len(correct)
                    good_tokens += sum([all_vocabulary.get(x[0], 0) for x in correct])
                    type_based[i + 1] = good_types
                    token_based[i + 1] = good_tokens
                logging.info("%d recovered types", good_types)
                logging.info("%d recovered tokens", good_tokens)
            #pyplot.subplot(1, 2, 1)
            logging.info("%s at %d, %d/%d recovered types", name, (type_based.shape[0] / 2) * bin_size, type_based[type_based.shape[0] / 2], num_oov_types)
            pyplot.plot(100 * type_based / float(num_oov_types), label=name)
            #pyplot.subplot(1, 2, 2)
            #pyplot.plot(token_based, label=name)

        #pyplot.subplot(1, 2, 1)
        #pyplot.title("Type-based")
        #pyplot.xlabel("Expansion threshold (in 1000s of words)")        
        pyplot.ylabel("% OOV reduction")
        pyplot.legend(loc="lower right", fontsize=10)
        pyplot.xticks([x * bin_size for x in range(11)], [(x * bin_size) / 10 for x in range(11)])
        #print type_based.max()
        yinc = float(type_based.max()) / 9
        #yinc = 2409.0 / 9
        #pyplot.yticks([x * yinc for x in range(10)], ["%d" % (int(100 * x * yinc / float(num_oov_types))) for x in range(10)])
        yinc = 35.0 / 9
        pyplot.yticks([x * yinc  for x in range(10)], ["%d" % (x * yinc) for x in range(10)])
        #pyplot.yticks([x * yinc for x in range(10)], ["%d/%d" % (int(100 * x * yinc / float(num_oov_types)), int(100 * x * yinc / float(num_iv_types))) for x in range(10)], fontsize=8)
        pyplot.grid()

        #pyplot.subplot(1, 2, 2)
        #pyplot.title("Token-based")
        pyplot.xlabel("1000s of words")
        #pyplot.ylabel("%% OOV reduction/IV increase (%d initially OOV tokens)" % (num_oov_tokens))
        #pyplot.legend(loc="lower right", fontsize=10)
        #pyplot.xticks([x * bin_size for x in range(11)], [(x * bin_size) / 10 for x in range(11)])
        #yinc = float(token_based.max()) / 9
        #pyplot.yticks([x * yinc for x in range(10)], ["%d" % (int(100 * x * yinc / float(num_oov_tokens))) for x in range(10)])
        #pyplot.yticks([x * yinc for x in range(10)], ["%d/%d" % (int(100 * x * yinc / float(num_oov_tokens)), int(100 * x * yinc / float(num_iv_tokens))) for x in range(10)], fontsize=8)
        #pyplot.grid()

        pyplot.savefig(target[0].rstr())
        pyplot.cla()
        pyplot.clf()
    return None


def plot_reduction_emitter(target, source, env):
    args = source[-1].read()
    new_targets = pjoin(os.path.dirname(source[0].rstr()), "%d_reduction.png" % (args["bins"]))
    return new_targets, source


def plot_unigram_probabilities(target, source, env):
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


# def run_g2p(target, source, env):
#     with temp_file() as tfname, meta_open(source[0].rstr()) as pl_fd:
#         words = set([x.split()[0].split("(")[0] for x in pl_fd])
#         with meta_open(tfname, "w") as t_fd:
#             t_fd.write("\n".join(words))
#         out, err, success = run_command(env.subst("%s %s/bin/g2p.py --model %s --encoding=%s --apply %s --variants-mass=%f  --variants-number=%d" % (env["PYTHON"], env["OVERLAY"], source[1].rstr(), "utf-8", tfname, .9, 4)),
#                                         env={"PYTHONPATH" : env.subst("${OVERLAY}/lib/python2.7/site-packages")},
#                                         )
#         if not success:
#             return err
#         else:
#             with meta_open(target[0].rstr(), "w") as out_fd:
#                 out_fd.write(out)
#     return None


# def g2p_to_babel(target, source, env):
#     # include accuracy
#     swap = source[1].read()
#     myWords = {}
#     with meta_open(source[0].rstr()) as in_fd, meta_open(target[0].rstr(), "w") as out_fd:
#         for tokens in [x.strip().replace('\t\t','\t').split('\t') for x in in_fd]:
#             if len(tokens) > 3:
#                 word, pronunciation = tokens[0], tokens[-1]
#                 pronunciation = pronunciation.replace('\"', '').replace('%','').replace('.','').strip().replace('#', '').replace('  ',' ').replace('  ',' ').replace('  ',' ')
#                 for k, v in swap.iteritems():
#                     pronunciation = pronunciation.replace(k, v)

#                 phonemes = pronunciation.split(" ")
#                 if len(phonemes) > 1:
#                     new_phonemes = [phonemes[0], "[ wb ]"] + phonemes[1:] + ["[ wb ]"]
#                     new_pronunciation = " ".join(new_phonemes)
#                 elif ':' in phonemes[0] or phonemes[0] in ['a', 'e', 'o']:
#                     new_pronunciation = "%s [ wb ]" % (pronunciation)
#                 else:
#                     new_pronunciation = pronunciation
#                 new_pronunciation = new_pronunciation.strip()
#                 myWords[word] = myWords.get(word, []) + [new_pronunciation]
#             else:
#                 logging.info("couldn't process G2P output: %s", "\t".join(tokens))
#         for word in myWords.keys():
#             for count, pronun in enumerate(myWords[word]):
#                 out_fd.write("%s(%.2d) %s\n" % (word, count, pronun))
#     return None


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
    jnr    = int(env["JOB_COUNT"]) #args.get("JOB_COUNT", 1)
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
    with open(target[0].rstr(), "w") as ofd:
        for utt in cfg.db:
            cfg.nn.eval(utt)
            se.search()
            key    = utt + ' ' + os.path.splitext(cfg.db.getFile(utt))[0]
            txt    = se.getHyp().strip()
            hyp    = se.getCTM(key, cfg.db.getFrom(utt))
            tscore = se.getScore()
            #print utt,'score= %.5f frameN= %d'%(tscore, se.dnet.state.frameN)
            #print utt,'words=',txt
            for c in hyp:
                print >>ofd,c
            if genLat:
                se.rescore(rescoreBeam)
                se.lat.write(os.path.join(cfg.latDir,"%s.fsm.gz" % utt), cfg.db.getFrom(utt))
    return None

def asr_dummy(target, source, env):
    time.sleep(10)
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("test")
    return None

def run_asr(env, root_path, vocabulary, pronunciations, language_model, *args, **kw):
    env.Replace(ROOT_PATH=root_path)
    dnet = env.ASRConstruct("${ROOT_PATH}/dnet.bin.gz", [vocabulary, pronunciations, language_model])
    tests = []
    for i in range(env["JOB_COUNT"]):
        tests.append(env.ASRTest(["${ROOT_PATH}/ctm/%d.ctm" % i],
                                 [dnet, vocabulary, pronunciations, language_model, env.Value({"JOB_ID" : i})]))
    return [dnet] + tests


def TOOLS_ADD(env):
    env["FLOOKUP"] = "flookup"
    BUILDERS = {"ASRConstruct" : Builder(action=asr_construct),
                
                #"SplitTrainDev" : Builder(action=split_train_dev),
                #"GraphFile" : Builder(action=graph_file),
                #"VTLN" : Builder(action=vtln),
                #"AppenToAttila" : Builder(action=appen_to_attila),
                #"PronunciationsToVocabulary" : Builder(action=pronunciations_to_vocabulary),
                "IBMTrainLanguageModel" : Builder(action=ibm_train_language_model),
                #"MissingVocabulary" : Builder(action=missing_vocabulary),
                #"AugmentLanguageModel" : Builder(action=augment_language_model, emitter=augment_language_model_emitter),
                #"AugmentLanguageModelFromBabel" : Builder(action=augment_language_model_from_babel),
                #"TranscriptVocabulary" : Builder(action=transcript_vocabulary),
                #"TrainPronunciationModel" : Builder(action=train_pronunciation_model),
                #"CollectText" : Builder(action=collect_text, emitter=collect_text_emitter),
                #"BabelGumLexicon" : Builder(action=babelgum_lexicon),
                #"ReplacePronunciations" : Builder(action=replace_pronunciations),
                #"ReplaceProbabilities" : Builder(action=replace_probabilities),
                #"FilterWords" : Builder(action=filter_words),                           
                #"FilterBabelGum" : Builder(action=filter_babel_gum),
                #"ScoreResults" : Builder(action=score_results, emitter=score_emitter),
                #"CollateResults" : Builder(action=collate_results),
                #"SplitExpansion" : Builder(action=split_expansion, emitter=split_expansion_emitter),
                #"PlotProbabilities" : Builder(action=plot_probabilities),
                #"PlotReduction" : Builder(action=plot_reduction, emitter=plot_reduction_emitter),
                #"PlotUnigramProbabilities" : Builder(action=plot_unigram_probabilities),
                #"TranscriptsToVocabulary" : Builder(action=transcripts_to_vocabulary),
                
                #"CreateASRExperiment" : Builder(action=create_asr_experiment, emitter=create_asr_experiment_emitter),
                #"RunASR" : Builder(action=run_asr, emitter=run_asr_emitter),
                #"RunASRExperiment" : Builder(action=run_asr_experiment, emitter=run_asr_experiment_emitter),
                #"PronunciationsFromProbabilityList" : Builder(action=pronunciations_from_probability_list),
                #"TopWords" : Builder(action=top_words),
                #"RunG2P" : Builder(action=run_g2p),
                #"G2PToBabel" : Builder(action=g2p_to_babel),
                #"PronunciationPerformance" : Builder(action=pronunciation_performance),
                }

    if env["HAS_TORQUE"]:
        BUILDERS["ASRTest"] = Builder(action=torque_run)
    elif env["IS_THREADED"]:
        BUILDERS["ASRTest"] = Builder(action=threaded_run)
    else:
        BUILDERS["ASRTest"] = Builder(action=asr_test)
    env.Append(BUILDERS=BUILDERS)
    env.AddMethod(run_asr, "RunASR")
