# imports from the default Python libraries
import os
import os.path
import sys
from glob import glob
import logging
import time
import re
from os.path import join as pjoin

# imports from the SCons library
from SCons.Tool import textfile

# imports from this (the "babel") repository
import babel_tools
import asr_tools
import kws_tools

# imports from the "python" git repository
import morfessor_tools
import pycfg_tools
import torque_tools
import vocabulary_tools
import g2p_tools
import openfst_tools
from scons_tools import make_threaded_builder
from torque_tools import make_torque_builder
import scons_tools
from common_tools import meta_open


# Everything is configured by variables described below, where you can see (NAME, HELP_TEXT, DEFAULT_VALUE)
# These shouldn't be edited here, but rather overridden in the custom.py file
# Many variables point to specific files, but sometimes the file naming we receive is a bit strange, and
# instead the variable will be a regular expression that will (hopefully) match the needed file
vars = Variables("custom.py")
vars.AddVariables(

    # these variables are just cosmetic or for debugging purposes and don't affect the actual build process
    ("OUTPUT_WIDTH", "Controls the maximum line width SCons will print before truncating", 130),
    BoolVariable("DEBUG", "Only run the first job in a given parallel set (e.g. to make sure everything works before using Torque)", False),
    ("LOG_LEVEL", "How much information to display while building", logging.INFO),
    ("LOG_DESTINATION", "Where to display log output", sys.stdout),
    ("COMMAND_LINE_SUFFIX", "This is added to the end of command-line builders (e.g. to redirect stderr/stdout to /dev/null)", ""),
    
    # ideally, BASE_PATH is the only path variable you need to define, since all the others are defined relative to it, 
    # but you can also override other path variables on a case-by-case basis in custom.py
    ("BASE_PATH", "Directory containing lots of resources needed for experiments", None),
    ("OVERLAY", "Location of lots of binaries and libraries (installed e.g. with 'make PREFIX=...')", "${BASE_PATH}/local"),
    
    # these variables determine what experiments are performed
    ("LANGUAGES", "A dictionary describing each language: see custom.py.template for examples", {}),
    ("RUN_ASR", "Whether to perform ASR: otherwise, use dummy builders", True),
    ("RUN_KWS", "Whether to perform KWS: otherwise, use dummy builders", True),
    ("RUN_SEGMENTATION", "Whether to learn morphological models and apply them to other word lists", True),

    # parameters related to how Adaptor Grammars are trained (i.e. using the py-cfg tool)
    ("NUM_ITERATIONS", "Number of training iterations", 1000),
    ("NUM_SAMPLES", "Number of samples to gather at the end of sampling", 1),
    ("ANNEAL_INITIAL", "Initial value of annealing parameter", 3),
    ("ANNEAL_FINAL", "Final value of annealing parameter", 1),
    ("ANNEAL_ITERATIONS", "Number of iterations over which to decrease annealing value", 500),
    
    # these variables determine how parallelism is exploited    
    ("LONG_RUNNING", "Builders that should be considered for parallel execution, in tuples of form (name, #targets, #sources, combine)", []),
    ("JOB_COUNT", "How many jobs to split a single experiment into", 1),
    BoolVariable("THREADED_SUBMIT_NODE", "Run parallel jobs using multiple cores", False),
    BoolVariable("TORQUE_SUBMIT_NODE", "Run parallel jobs using Torque (e.g. invoking scons on yetisubmit.cc.columbia.edu)", False),

    # Torque-specific variables
    ("TORQUE_TIME", "Maximum running time for each torque job", "11:30:00"),
    ("TORQUE_MEMORY", "Maximum memory usage for each Torque job", "3500mb"),
    ("TORQUE_INTERVAL", "How often to check whether all pending Torque jobs have finished", 60),
    ("TORQUE_LOG", "Where Torque will create log files", "work/"),
    ("BASHRC_TXT", "Bash script that sets up environment variables", "${BASE_PATH}/bashrc.txt"),
    ("TORQUE_RESOURCES", "Dictionary mapping experiment properties to Torque parallelization properties", {}),
    
    # Torque bookkeeping variables (don't set these yourself)
    BoolVariable("WORKER_NODE", "Tracks whether this is a worker node", False),
    ("JOB_ID", "Tracks which parallel job this is", 0),
    ("SCONSIGN_FILE", "Redirects dependency database when running on a worker node", None),
    
    # these variables define the locations of various tools and data
    ("LANGUAGE_PACK_PATH", "", "${BASE_PATH}/language_packs"),
    ("IBM_MODELS", "Acoustic models and related files provided by IBM", "${BASE_PATH}/ibm_models"),
    ("LORELEI_SVN", "Checkout of SVN repository hosted on lorelei", "${BASE_PATH}/lorelei_svn"),
    ("ATTILA_PATH", "IBM's ASR system", "${BASE_PATH}/attila"),
    ("F4DE_PATH", "NIST software for evaluating keyword search output", "${BASE_PATH}/lorelei_resources/F4DE"),
    ("INDUSDB_PATH", "Babel resource with keyword lists, transcripts, segmentations, and so forth", "${BASE_PATH}/lorelei_resources/IndusDB"),
    ("LIBRARY_OVERLAY", "", "${OVERLAY}/lib:${OVERLAY}/lib64:${LORELEI_TOOLS}/boost_1_49_0/stage/lib/"),
    ("PYCFG_PATH", "Mark Johnson's tool for training Adaptor Grammars", "${BASE_PATH}/local/bin"),
    
    # these variables all have default definitions in terms of the previous, but may be overridden as needed
    ("G2P", "Location of main Sequitur script", "g2p.py"),
    ("BABEL_BIN_PATH", "Location of IBM's compiled executables", "${LORELEI_SVN}/tools/kws/bin64"),
    ("BABEL_SCRIPT_PATH", "Location of IBM's KWS scripts", "${LORELEI_SVN}/tools/kws/scripts"),
    ("BABEL_CN_SCRIPT_PATH", "Location of IBM's confusion network KWS scripts", "${LORELEI_SVN}/tools/cn-kws/scripts"),
    ("WRD2PHLATTICE", "", "${BABEL_BIN_PATH}/wrd2phlattice"),
    ("BUILDINDEX", "", "${BABEL_BIN_PATH}/buildindex"),
    ("BUILDPADFST", "", "${BABEL_BIN_PATH}/buildpadfst"),
    ("MAKE_INDEX", "", "${LORELEI_SVN}/tools/cn-kws/scripts/make_index.pl"),
    ("LAT2IDX", "", "${BABEL_BIN_PATH}/lat2idx"),
    ("OPENFST_BINARIES", "", "${OVERLAY}/bin"),
    ("QUERY2PHONEFST", "", "${BABEL_BIN_PATH}/query2phonefst"),
    ("STDSEARCH", "", "${BABEL_BIN_PATH}/stdsearch"),
    ("MERGESEARCHFROMPARINDEXPRL", "", "${BABEL_SCRIPT_PATH}/mergeSearchFromParIndex.prl"),
    ("MERGESCORESSUMPOSTNORMPL", "", "${BABEL_SCRIPT_PATH}/merge.scores.sumpost.norm.pl"),
    ("PRINTQUERYTERMLISTPRL", "", "${BABEL_SCRIPT_PATH}/printQueryTermList.prl"),
    ("F4DENORMALIZATIONPY", "", "${BABEL_SCRIPT_PATH}/F4DENormalization.py"),
    ("JAVA_NORM", "", "${BABEL_REPO}/examples/babel-dryrun/javabin"),    
    ("KWSEVALPL", "", "${F4DE}/KWSEval/tools/KWSEval/KWSEval.pl"),    
    ("SUMTOONENORMALIZE", "", "${BABEL_SCRIPT_PATH}/applySTONormalization.prl"),
    ("MERGEIVOOVCASCADE", "", "${BABEL_SCRIPT_PATH}/merge_iv_oov_cascade.prl"),
    ("APPLYRESCALEDDTPIPE", "", "${BABEL_SCRIPT_PATH}/applyRescaledDTpipe.py"),
    ("BABELSCORER", "", "${F4DE_PATH}/KWSEval/tools/KWSEval/KWSEval.pl"),

    # KWS-related variables
    ("TRANSPARENT", "Symbols that don't correspond to an actual sound", "'<s>,</s>,~SIL,<epsilon>'"),
    ("ADD_DELETE", "", 5),
    ("ADD_INSERT", "", 5),
    ("NBESTP2P_IV", "How many expanded terms per IV query term according to P2P similarities", 2000),
    ("NBESTP2P_OOV", "How many expanded terms per OOV query term according to P2P similarities", 20000),
    ("MINPHLENGTH", "How many phones a word must have to be considered", 2),
    ("PRINT_WORDS_THRESH", "", "1e-10"),
    ("PRINT_EPS_THRESH", "", "1e-03"),
    ("PRUNE", "", 10),
    ("RESCORE_BEAM", "(increasing this variable can greatly increase memory/time usage)", 1.5),
    ("LOWER_CASE", "Whether the language should, in general, be converted to lower case", False),
    ("CN_KWS_SCRIPTS", "", "${BASE_PATH}/lorelei_svn/tools/cn-kws/scripts"),
    ("JAVA_NORM", "", "${BABEL_REPO}/KWS/examples/babel-dryrun/javabin"),
    ("SCLITE_BINARY", "", "${BASE_PATH}/sctk-2.4.5/bin/sclite"),    
    
    # ASR-related variables
    ("MODEL_PATH", "Location of acoustic models provided by IBM", "${IBM_MODELS}/${BABEL_ID}/${PACK}/models"),
    ("PHONE_FILE", "", "${MODEL_PATH}/pnsp"),
    ("PHONE_SET_FILE", "", "${MODEL_PATH}/phonesset"),
    ("TAGS_FILE", "", "${MODEL_PATH}/tags"),
    ("TREE_FILE", "", "${MODEL_PATH}/tree"),
    ("MLP_FILE", "", "${MODEL_PATH}/*.mlp"),
    ("TOPO_FILE", "", "${MODEL_PATH}/topo.*"),
    ("TOPO_TREE_FILE", "", "${MODEL_PATH}/topotree"),
    ("PRONUNCIATIONS_FILE", "Pronunciations shipped with IBM's acoustic models", "${MODEL_PATH}/dict.test"),
    ("VOCABULARY_FILE", "Vocabulary shipped with IBM's acoustic models", "${MODEL_PATH}/vocab"),
    ("LANGUAGE_MODEL_FILE", "Language model shipped with IBM's acoustic models", "${MODEL_PATH}/lm.2gm.arpabo.gz"),
    ("SAMPLING_RATE", "Sample rate of all audio files", 8000),
    ("FEATURE_TYPE", "", "plp"),
    ("MEL_FILE", "", "${MODEL_PATH}/mel"),
    ("LDA_FILE", "", "${MODEL_PATH}/30.mat"),
    ("PRIORS_FILE", "", "${MODEL_PATH}/priors"),
    ("P2P_FILE", "", "${MODEL_PATH}/P2P.fst"),
    ("OUTPUT_PATH", "", "work/asr/${LANGUAGE_NAME}/${EXPERIMENT_NAME}"),
    ("WARP_FILE", "", "${IBM_MODELS}/${BABEL_ID}/${PACK}/adapt/warp.lst"),
    ("GRAPH_FILE", "", "${OUTPUT_PATH}/dnet.bin.gz"),
    ("CTM_PATH", "", "${OUTPUT_PATH}/ctm"),
    ("LATTICE_PATH", "", "${OUTPUT_PATH}/lat"),
    ("TEXT_PATH", "", "${OUTPUT_PATH}/text"),
    ("PCM_PATH", "", "${LANGUAGE_PACK_PATH}/${BABEL_ID}_${LANGUAGE_NAME}"),
    ("DATABASE_FILE", "", "${IBM_MODELS}/${BABEL_ID}/${PACK}/segment/*db"),
    ("CMS_PATH", "", "${IBM_MODELS}/${BABEL_ID}/${PACK}/adapt/cms"),
    ("FMLLR_PATH", "", "${IBM_MODELS}/${BABEL_ID}/${PACK}/adapt/fmllr"),
    ("ECF_FILE", "", "${INDUSDB_PATH}/*babel${BABEL_ID}*conv-dev/*.scoring.ecf.xml"),
    ("DEV_KEYWORD_FILE", "File pattern for development keyword list", "${INDUSDB_PATH}/*babel${BABEL_ID}*conv-dev.kwlist2.xml"),
    ("EVAL_KEYWORD_FILE", "File pattern for evaluation keyword list", "data/eval_keyword_lists/*babel${BABEL_ID}*conv-eval.kwlist*.xml"),
    ("RTTM_FILE", "File pattern for the MIT audio segmentation", "${INDUSDB_PATH}/*babel${BABEL_ID}*conv-dev/*mit*rttm"),
)

# initialize logging system
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.ERROR)

# initialize build environment
env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default", "textfile"] + [x.TOOLS_ADD for x in [babel_tools, morfessor_tools,
                                                                         pycfg_tools,
                                                                         asr_tools, kws_tools, vocabulary_tools, g2p_tools, scons_tools,
                                                                         openfst_tools,
                                                                     ]],
                  )

def dummy_segmentation(target, source, env):
    words = set()
    with meta_open(source[0].rstr()) as ifd:
        for line in ifd:
            word, count = line.strip().split()
            if "_" not in word:
                words.add(word)
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(words) + "\n")            
    return None

env["BUILDERS"]["DummySegmentation"] = Builder(action=dummy_segmentation)

# this is the ugly mechanism that handles replacing a builder with a threaded/torque proxy
for b, t, s, ss in env["LONG_RUNNING"]:
    if env["WORKER_NODE"]:
        pass
    elif env["THREADED_SUBMIT_NODE"]:
        env["BUILDERS"][b] = make_threaded_builder(env["BUILDERS"][b], t, s, ss)
    elif env["TORQUE_SUBMIT_NODE"]:
        env["BUILDERS"][b] = make_torque_builder(env["BUILDERS"][b], t, s, ss)
if (env["WORKER_NODE"] or env["WORKER_NODE"]) and env.get("SCONSIGN_FILE", False):
    env.SConsignFile(env["SCONSIGN_FILE"])

# generate help text
Help(vars.GenerateHelpText(env))

# don't print out lines longer than the terminal width
def print_cmd_line(s, target, source, env):
    lines = []
    for line in s.split("\n"):
        if len(line) > int(env["OUTPUT_WIDTH"]):
            print line[:int(env["OUTPUT_WIDTH"]) - 10] + "..." + line[-7:]
        else:
            print line

env['PRINT_CMD_LINE_FUNC'] = print_cmd_line

# use time stamps to determine if a target needs to be rebuilt
env.Decider("timestamp-newer")

# aggregate targets by language, ASR/KWS, pack, and morphology (for more flexible selective building)
pseudo_targets = {}

#
# Begin defining the actual experiments and dependencies
#
model_names = ["prefix_suffix", "agglutinative"]
all_texts = []
for language, properties in env["LANGUAGES"].iteritems():
    
    env.Replace(BABEL_ID=properties["BABEL_ID"])
    env.Replace(LANGUAGE_NAME=language)
    env.Replace(LOCALE=properties.get("LOCALE"))
    env.Replace(NON_ACOUSTIC_GRAPHEMES=properties.get("NON_ACOUSTIC_GRAPHEMES", []))
    env.Replace(NON_WORD_PATTERN=".*(_|\<).*")
    env.Replace(FORCE_SPLIT=["-"])
    env.Replace(GRAPHEMIC=properties.get("GRAPHEMIC", False))
    
    packs = {}
    stripped_pack = env.FilterTar("work/stripped_packs/${BABEL_ID}.tgz", ["${LANGUAGE_PACK_PATH}/${BABEL_ID}_${LANGUAGE_NAME}.tgz", env.Value(r".*transcription.*")])
    
    if "FLP" in properties.get("PACKS", []):
        packs["FLP"] = env.CollectText("work/texts/${LANGUAGE_NAME}_FLP.txt",
                                       [stripped_pack, env.Value(".*transcription.*txt")],
                                   )
    
    if "LLP" in properties.get("PACKS", []):
        packs["LLP"] = env.CollectText("work/texts/${LANGUAGE_NAME}_LLP.txt",
                                       [stripped_pack, env.Value(".*sub-train/transcription.*txt")],
        )
    
    if "VLLP" in properties.get("PACKS", []):
        packs["VLLP"] = env.StmToData("work/texts/${LANGUAGE_NAME}_VLLP.txt",
                                      ["${BASE_PATH}/LPDefs.20141006.tgz", env.Value(env.subst(".*IARPA-babel${BABEL_ID}.*.VLLP.training.transcribed.stm"))]
        )
    
    for pack, data in packs.iteritems():

        env.Replace(PACK=pack)        
        baseline_vocabulary = env.File("${VOCABULARY_FILE}")
        baseline_pronunciations = env.File("${PRONUNCIATIONS_FILE}")
        segmentations = {}

        if os.path.exists(pjoin(env.subst("${IBM_MODELS}/${BABEL_ID}/${PACK}"))):
            baseline_language_model = env.Glob("${IBM_MODELS}/${BABEL_ID}/${PACK}/models/*.arpabo.gz")[0]
            env.Replace(ACOUSTIC_WEIGHT=properties.get("ACOUSTIC_WEIGHT", .10))
            baseline_asr_output = env.RunASR("work/asr_experiments/${LANGUAGE_NAME}/${PACK}/baseline",
                                             baseline_vocabulary,
                                             baseline_pronunciations,
                                             baseline_language_model,
                                             TORQUE_RESOURCES=env["TORQUE_RESOURCES"].get(("asr", "words"), {}))

            if not properties.get("GRAPHEMIC", False):
                g2p_model = env.TrainG2P("work/pronunciations/${LANGUAGE_NAME}_${PACK}_${MODEL}_baseline_model_1.txt", baseline_pronunciations)
            else:
                g2p_model = None
            
            keyword_file = env.maybe("${DEV_KEYWORD_FILE}")

            baseline_asr_word_error_rate = env.WordErrorRate(
                ["work/word_error_rates/${LANGUAGE_NAME}/${PACK}/baseline/%s" % x for x in ["babel.sys", "all.ctm", "babel.dtl", "babel.pra", "babel.raw", "babel.sgml"]],
                [x[0] for x in baseline_asr_output] + env.Glob("${INDUSDB_PATH}/IARPA-babel${BABEL_ID}*/*dev.stm"))
            
            baseline_kws_output = env.RunKWS("work/kws_experiments/${LANGUAGE_NAME}/${PACK}/baseline",
                                             [x[1] for x in baseline_asr_output], baseline_vocabulary, baseline_pronunciations, env.Glob("${DEV_KEYWORD_FILE}"),
                                             G2P_MODEL=g2p_model)

            word_list = env.WordList("work/word_lists/${LANGUAGE_NAME}_${PACK}.txt", data)
            keywords_for_models = env.KeywordListToModelInput("work/word_lists/${LANGUAGE_NAME}_${PACK}_keywords.txt", keyword_file)
            morfessor, morfessor_model = env.TrainMorfessor(["work/morfessor/${LANGUAGE_NAME}_${PACK}.txt", "work/morfessor/${LANGUAGE_NAME}_${PACK}.model"], word_list)
            segmented_keyword_vocab = env.ApplyMorfessor("work/morfessor/${LANGUAGE_NAME}_${PACK}_segmented_keyword_vocab.txt", 
                                                         [morfessor_model, keywords_for_models])
            segmented_keywords = env.ReconstructSegmentedKeywords("work/morfessor/${LANGUAGE_NAME}_${PACK}_segmented_keywords.xml", 
                                                                  [keyword_file, segmented_keyword_vocab])

            segmentations["morfessor"] = (morfessor, segmented_keywords)

            for model_name in ["prefix_suffix"]:
                
                (ag, segmented_keyword_vocab) = env.AdaptorGrammarBabelExperiment("work/adaptor_grammars/${LANGUAGE_NAME}_${PACK}_${MODEL_NAME}/",
                                                                                  model_name,
                                                                                  properties.get("NON_ACOUSTIC_GRAPHEMES", []),
                                                                                  [word_list, keywords_for_models])

                segmented_keywords = env.ReconstructSegmentedKeywords("work/adaptor_grammars/${LANGUAGE_NAME}_${PACK}_${MODEL_NAME}_segmented_keywords.xml", 
                                                                      [keyword_file, segmented_keyword_vocab])
                segmentations[model_name] = (ag, segmented_keywords)
            
            for model_name,(segmentation, segmented_keywords) in segmentations.iteritems():
                env.Replace(MODEL=model_name)
                segmented_pronunciations_training, morphs = env.SegmentedPronunciations(["work/pronunciations/${LANGUAGE_NAME}_${PACK}_${MODEL}_segmented.txt",
                                                                                         "work/pronunciations/${LANGUAGE_NAME}_${PACK}_${MODEL}_morphs.txt"],
                                                                                        [baseline_pronunciations, segmentation])

                if properties.get("GRAPHEMIC", False):
                    morph_pronunciations = env.GraphemicPronunciations("work/pronunciations/${LANGUAGE_NAME}_${PACK}_${MODEL}_morph_pronunciations.txt", morphs)
                    g2p_segmented_model = None
                else:
                    g2p_segmented_model = env.TrainG2P("work/pronunciations/${LANGUAGE_NAME}_${PACK}_${MODEL}_segmented_model_1.txt", segmented_pronunciations_training)
                    morph_pronunciations = env.ApplyG2P("work/pronunciations/${LANGUAGE_NAME}_${PACK}_${MODEL}_morph_pronunciations.txt", [g2p_segmented_model, morphs])

                segmented_vocabulary, segmented_pronunciations = env.PronunciationsToVocabDict(
                    ["work/asr_input/${LANGUAGE_NAME}_${PACK}_${MODEL}_vocabulary.txt", "work/asr_input/${LANGUAGE_NAME}_${PACK}_${MODEL}_pronunciations.txt"],
                    [morph_pronunciations, baseline_pronunciations, env.Value(properties.get("GRAPHEMIC", False))])

                segmented_training_text = env.SegmentTranscripts("work/segmented_training/${LANGUAGE_NAME}_${PACK}_${MODEL}.txt", [data, segmentation])

                segmented_language_model = env.TrainLanguageModel("work/asr_input/${LANGUAGE_NAME}_${PACK}_${MODEL}_languagemodel_segmented.arpabo.gz",
                                                                  [segmented_training_text, Value(2)])

                segmented_asr_output = env.RunASR("work/asr_experiments/${LANGUAGE_NAME}/${PACK}/${MODEL}",
                                                  segmented_vocabulary,
                                                  segmented_pronunciations,
                                                  segmented_language_model,
                                                  TORQUE_RESOURCES=env["TORQUE_RESOURCES"].get(("asr", "morphs"), {}))

                segmented_kws_output = env.RunKWS("work/kws_experiments/${LANGUAGE_NAME}/${PACK}/${MODEL}", [x[1] for x in segmented_asr_output], segmented_vocabulary, segmented_pronunciations, segmented_keywords, G2P_MODEL=g2p_segmented_model)

                cascaded_kws_output = env.RunCascade("work/kws_experiments/${LANGUAGE_NAME}/${PACK}/${MODEL}_cascaded",
                                                     baseline_kws_output, segmented_kws_output)
