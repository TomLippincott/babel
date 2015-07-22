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
import pickle
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
import sys
import user
import dsearch
import misc
import dbase
import frontend
import nnet
from attila import NNScorer, errorHandler, Act_Rectified, Act_Sigmoid, Act_ID, MatrixCU, Act_Tanh, Act_Softmax, HMM, Window, Audio, Decimator
from frontend import FeFFT, FeMEL, FeGamma, FeMFCC, FeLogMel, FeRootMel, FePLP, FeFusion, FeNorm, FeDeltaCTX, FeCTX, FeLDA, FeFMLLR, FeFMPE, FeFMMI, FeNormAccu, FeIIRFilter, FeFile, SpeechDetector, FeSilenceDetector, FePad, FeCache, FeMLP, FeFFV, FeOnlyFFV, FeCombo
import frontenditf
from scons_tools import run_command


class TarAudio(Audio):
    pass


class TarFeAudio(frontend.FeAudio):

    def __init__(self,win):
        frontenditf.Fe.__init__(self)
        self.readMode    = "speaker"
        self.readSwitch  = False
        self.key         = 0,0,0,0
        self.pcmDir      = ''
        self.pcmVar      = ''
        self.pcmSpk      = TarAudio()
        self.pcmUtt      = TarAudio()
        self.win         = win
        self.feat        = self.pcmUtt
        self.socket      = None
        self.dec         = Decimator()
        self.dec.factor  = 1
        self.preemphasis = 1
    
    def readFile(self,utt):
        spk   = self.db.getSpk(utt)
        name  = self.db.getDir(spk,root=self.pcmDir)
        name += os.sep + self.pcmVar + os.sep + self.db.getFile(utt)
        ext   = os.path.splitext(name)[1].lower()
        real  = misc.findExt(name)
        real  = os.path.normpath(real)
        ch    = self.db.getChannel(utt)
        beg   = self.win.s2f(self.db.getFrom(utt))
        beg   = self.dec.factor * self.win.f2p(beg)
        end   = self.dec.factor * self.win.s2p(self.db.getTo(utt))        
        key   = real,ch,beg,end
        name = self.db.getFile(utt)
        
        # same audio segment
        if self.key == key:
            return       
        # same audio file
        if self.key[:2] == key[:2] and self.pcmSpk.dimN > 0:
            self.key = key
            if beg == 0 and end < 0:
                self.feat = self.pcmSpk
                return
            if end < 0:
                end = self.pcmSpk.dimN
            if end > self.pcmSpk.dimN:
                misc.warn('FeAudio::readFile','invalid end time','%s to= %s sampleN= %d'%(utt,self.db.getTo(utt),self.pcmSpk.dimN))
                end = self.pcmSpk.dimN
            self.pcmUtt.slice(beg,end,self.pcmSpk)
            self.feat = self.pcmUtt
            return                            
        misc.info('FeAudio::readFile','read','%s beg= %d end= %d'%(real,beg,end))
        self.pcmSpk.resize(0,0)
        self.pcmUtt.resize(0,0)        
        self.key = key
        if self.readMode == 'speaker':
            rbeg,rend = 0,-1
            self.feat = self.pcmSpk
        else:
            rbeg,rend = beg,end
            self.feat = self.pcmUtt
        # file formats that can read segments
        if ext == '.sph':            
            self.feat.readSphere(real,rbeg,rend,ch)
        # if ext == '.flac':            
        #     self.feat.readFlac(real,rbeg,rend,ch)
        # if ext == '.pcm8':            
        #     self.feat.readPCM(real,1,rbeg,rend)
        # if ext == '.pcm16' or ext == '.pcm':            
        #     self.feat.readPCM(real,2,rbeg,rend)
        if self.feat.dimN > 0:
            # extract segment from entire audio file when in speakerMode
            if self.feat == self.pcmSpk:
                if beg == 0 and end < 0:
                    self.feat = self.pcmSpk
                    return
                if end < 0:
                    end = self.pcmSpk.dimN
                if end > self.pcmSpk.dimN:
                    misc.warn('FeAudio::readFile','invalid end time','%s to= %s sampleN= %d'%(utt,self.db.getTo(utt),self.pcmSpk.dimN))
                    end = self.pcmSpk.dimN
                self.pcmUtt.slice(beg,end,self.pcmSpk)
                self.feat = self.pcmUtt
            return
        # file formats that cannot read segments
        self.feat = self.pcmSpk
        # if ext == '.ulaw':
        #     self.feat.readUlaw(real)
        # elif ext == '.vox':
        #     self.feat.readVOX(real)
        if ext == '.wav':
            self.feat.readWAV(real)
        else:
            raise RunTimeError, 'FeAudio::readFile unknown file extension %s'%ext
        # extract segment from entire audio file
        if beg == 0 and end < 0:
            self.feat = self.pcmSpk
            return
        if end < 0:
            end = self.pcmSpk.dimN
        if end > self.pcmSpk.dimN:
            misc.warn('FeAudio::readFile','invalid end time','%s to= %s sampleN= %d'%(utt,self.db.getTo(utt),self.pcmSpk.dimN))
            end = self.pcmSpk.dimN
        self.pcmUtt.slice(beg,end,self.pcmSpk)
        self.feat = self.pcmUtt
        return


class TarFeCombo(frontend.FeCombo):
    def __init__(self,db,sr=8000,type='plp',ctxtype='splice',streaming=False):
        self.win       = Window()
        self.pcm       = d = TarFeAudio(self.win)
        self.fft       = d = FeFFT([d],self.win)        
        self.mel       = d = FeMEL([self.fft])
        self.gamma     = d = FeGamma([self.fft])        
        self.mfcc      = d = FeMFCC([self.mel])
        self.logmel    = d = FeLogMel([self.mel])
        self.rootmel   = d = FeRootMel([self.mel])
        self.plp       = d = FePLP([self.mel])
        self.fusion    = d = FeFusion([self.plp,self.mfcc])
        self.norm      = d = FeNorm([self.plp])
        self.deltactx  = d = FeDeltaCTX([self.norm])
        self.ctx       = d = FeCTX([self.norm])
        self.lda       = d = FeLDA([d])
        self.fmllr     = d = FeFMLLR([d])
        self.fmpe      = d = FeFMPE([d])
        self.fmmi      = d = FeFMMI([self.fmllr])
        self.accu      = d = FeNormAccu(self.mel,[self.plp])
        self.iirfilter = FeIIRFilter([self.pcm])
        self.file      = FeFile()            
        self.sdet      = SpeechDetector([self.pcm])
        self.sildet    = FeSilenceDetector([self.pcm])
        self.pad       = FePad([])
        self.cache     = FeCache([])
        self.mlp       = FeMLP([])
        self.nn        = nnet.NeuralNet()
        self.ffv       = FeFFV(sr,[])      
        self.ffvonly   = FeOnlyFFV(self.win,[])
        self.end       = ''
        self.win.sr    = sr
        self.win.win   = 200 * sr / 8000
        self.win.shift = 80  * sr / 8000
        self.win.segs  = 1
        # ensure FFT length is a power of 2
        m, e = math.frexp(sr / 8000.0)
        if m == 0.5:
            e -= 1
        self.fft.fft.fft = int(math.pow(2,e) * self.fft.fft.fft)
        self.connect(type,ctxtype)
        self.assignDB(db)
        for e in vars(self).itervalues():
            if hasattr(e,'streaming') : e.streaming = streaming
        return


class CFG():
    """
    This class captures the configuration information that is provided by cfg.py in
    IBM's ASR pipelines.  It expands any variable ending in "FILE" using a regular
    expression glob on the file system.
    """
    dictFile = '${PRONUNCIATIONS_FILE}'
    vocab = '${VOCABULARY_FILE}'
    lm = '${LANGUAGE_MODEL_FILE}'
    dbFile = '${DATABASE_FILE}'
    graph = '${NETWORK_FILE}'    
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
    """Train an n-gram language model using a plain text transcript.

    Uses IBM's compiled LM tools that ship with Attila.  This can also be used on a segmented transcript,
    in which case the n-grams are over morphs rather than words.

    Sources: transcript file, n
    Targets: language model file
    """
    text_file = source[0].rstr()
    n = source[1].read()
    with temp_dir() as prefix_dir, temp_file() as vocab_file, temp_file(suffix=".txt") as sentence_file, meta_open(text_file) as text_fd:
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
        
        lm = ".".join(target[0].rstr().split(".")[0:-2])
        cmd = "${ATTILA_PATH}/tools/lm_64/BuildNGram.sh -n %d -arpabo %s %s" % (n, prefix, lm)
        out, err, success = run_command(env.subst(cmd), env={"SFCLMTOOLS" : env.subst("${ATTILA_PATH}/tools/lm_64")})
        if not success:
            return err
        
    return None


def word_error_rate(target, source, env):
    """Calculate the word error rate of CTM transcripts with respect to an STM  gold standard.

    Uses the government's SCLite scoring tools, and is essentially a cut-and-paste from the
    "score.py" files included with the models IBM shipped to us.  It generates several files,
    but the important one (AFAIK) is "babel.sys", which contains the word error rate (WER).  
    Note that the WER is not necessarily a good indicator of KWS performance, but it should
    at least demonstrate that the ASR is running correctly.  This can only be run on output
    from a baseline experiment (i.e. not morph-space).

    Sources: ctm file #1, ctm file #2, ..., stm transcript file
    Targets: "babel.sys", "all.ctm", "babel.dtl", "babel.pra", "babel.raw", "babel.sgml"
    """
    ctms = [x.rstr() for x in source[0:-1]]
    transcript = source[-1].rstr()
    out_path = os.path.dirname(target[0].rstr())

    # Get a list of IDs from the reference.  All must appear in the CTM output
    spkD = set()
    with meta_open(transcript) as ifd:
        for line in ifd:
            if line.startswith(";;"):
                continue
            spkD.add(line.split()[0])

    # skip eval data
    isEval = re.compile("/eval/")

    # Merge and clean up CTM
    skipD = frozenset([u"~SIL", u"<s>", u"</s>", u"<HES>", u"<hes>"])
    ctmL = []
    for ctm in ctms:
        with meta_open(ctm) as ifd:
            for line in ifd:
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


def decoding_network(target, source, env):
    """Prepare a decoding network for an ASR experiment (based on Zulu LLP scripts from IBM).

    This is a fast builder that creates a single, small file that is shared by all parallel jobs
    of a given experiment.

    Sources: vocabulary file, pronunciation file, language model
    Targets: decoding network file
    """
    vocabulary, pronunciations, language_model = source
    env.Replace(VOCABULARY_FILE=vocabulary, PRONUNCIATIONS_FILE=pronunciations, LANGUAGE_MODEL_FILE=language_model)
    cfg = CFG(env)
    se = dsearch.Decoder(lmType=32)
    se.build(cfg)
    se.dnet.write(target[0].rstr())
    return None


def decode(target, source, env):
    """Decode some audio using a decoding network and some models (based on the example pipelines from IBM).

    This is the heart, and by far the most complicated and error-prone part, of the pipeline.  Basically,
    the models IBM sent us are similar, but have small variations so that some need to be run
    differently.  This builder tries to figure out what to do based on what model files exist, and
    then run the appropriate code.  If it can't figure out what to run, it throws an error.  It is also 
    aware of how many jobs the experiment has been split into, and only runs the job it was told to.  Most
    of the code was just slightly-adapted from the cfg.py, construct.py, and test.py files in the acoustic 
    models IBM sent us.

    Sources: decoding network file, vocabulary file, pronunciation file, language model file
    Targets: ctm transcript file, consensus network file
    """
    dnet, vocabulary, pronunciations, language_model = source
    out_path, tail = os.path.split(os.path.dirname(target[0].rstr()))
    env.Replace(VOCABULARY_FILE=vocabulary.rstr(),
                PRONUNCIATIONS_FILE=pronunciations.rstr(),
                LANGUAGE_MODEL_FILE=language_model,
                NETWORK_FILE=dnet,
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

    fe = FeCombo(db, int(env["SAMPLING_RATE"]), env["FEATURE_TYPE"])
    fe.end            = fe.fmllr
    fe.pcm.pcmDir     = cfg.pcmDir
    fe.pcm.readMode   = 'speaker'
    fe.norm.normMode  = 1
    fe.norm.normDir   = env.subst("${CMS_PATH}")
    fe.fmllr.fmllrDir = env.subst("${FMLLR_PATH}")
    
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
            if mlp or nmlp:
                fe.end.eval(utt)
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
                fname = "%s.fsm" % (utt)
                se.lat.write(temp_fname, db.getFrom(utt))
            elif writeCons:
                fname = "%s.cons" % (utt)
                arcN = len(se.lat.arcs)
                durS = db.getTo(utt)- db.getFrom(utt)
                dens = arcN / durS
                se.consensus(postThresh)
                binThresh         = 1.0e-10
                writeSIL          = 0
                se.cons.write(temp_fname, db.getFrom(utt), binThresh, writeSIL)
            tf_ofd.add(temp_fname, arcname=fname)
        tf_ofd.close()
    return None


def run_asr(env, root_path, vocabulary, pronunciations, language_model, *args, **kw):
    """Set up an ASR experiment using the computational resources available.
    
    This is not an actual builder, but a "method" that sets up an ASR experiment according 
    to its inputs and the resources available on the computer/specified in the SCons variables.
    If RUN_ASR = False, just return any existing consensus network files already in place.

    Inputs: output path, vocabulary file, pronunciation file, language model file
    Outputs: (transcript file 1, consensus file 1), (transcript file 2, consensus file 2), ...
    """
    env.Replace(ROOT_PATH=root_path)
    tests = []
    resources = kw.get("TORQUE_RESOURCES", {})
    env.Replace(JOB_COUNT=resources.get("JOB_COUNT", env.get("JOB_COUNT")))
    env.Replace(TORQUE_TIME=resources.get("TORQUE_TIME", env.get("TORQUE_TIME")))
    env.Replace(TORQUE_MEMORY=resources.get("TORQUE_MEMORY", env.get("TORQUE_MEMORY")))
    if env["RUN_ASR"]:
        decoding_network = env.DecodingNetwork("${ROOT_PATH}/decoding_network.gz", [vocabulary, pronunciations, language_model],
                                               PACK=env["PACK"], BABEL_ID=env["BABEL_ID"], LANGUAGE_NAME=env["LANGUAGE_NAME"])    
        to = 1 if env["DEBUG"] else env["JOB_COUNT"]
        for i in range(to):
            tests.append(env.Decode(["${ROOT_PATH}/transcripts_${JOB_ID + 1}_of_${JOB_COUNT}.ctm.gz",
                                     "${ROOT_PATH}/confusion_networks_${JOB_ID + 1}_of_${JOB_COUNT}.tgz"],
                                    [decoding_network, vocabulary, pronunciations, language_model], PACK=env["PACK"],
                                    BABEL_ID=env["BABEL_ID"], JOB_ID=i, JOB_COUNT=env.get("JOB_COUNT"), LANGUAGE_NAME=env["LANGUAGE_NAME"],
                                    TORQUE_TIME=env.get("TORQUE_TIME"), TORQUE_MEMORY=env.get("TORQUE_MEMORY")))
    else:
        tests = [[None, x] for x in env.Glob("${ROOT_PATH}/*_[0-9]*_of_[0-9]*.tgz")]
    return tests


def TOOLS_ADD(env):
    """Conventional way to add the four builders and one method to an SCons environment."""
    env.Append(BUILDERS = {
        "DecodingNetwork" : Builder(action=decoding_network),
        "Decode" : Builder(action=decode),
        "TrainLanguageModel" : Builder(action=train_language_model),
        "WordErrorRate" : Builder(action=word_error_rate),
    })
    env.AddMethod(run_asr, "RunASR")
