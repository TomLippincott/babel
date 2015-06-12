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

# imports from the "python" repository
#import scala_tools
import morfessor_tools
#import emma_tools
#import trmorph_tools
#import sfst_tools
#import evaluation_tools
#import almor_tools
#import mila_tools
import pycfg_tools
import torque_tools
import vocabulary_tools
import g2p_tools
import openfst_tools
from scons_tools import make_threaded_builder
from torque_tools import make_torque_builder
import scons_tools

# Everything is configured by variables described below, where you can see (NAME, HELP_TEXT, DEFAULT_VALUE)
# These shouldn't be edited here, but rather overridden in the custom.py file
vars = Variables("custom.py")
vars.AddVariables(

    # these variables are just cosmetic or for debugging purposes and don't affect the actual build process
    ("OUTPUT_WIDTH", "Controls the maximum line width SCons will print before truncating", 130),
    BoolVariable("DEBUG", "Not really used, but should control debugging output", True),
    ("LOG_LEVEL", "How much information to display while building", logging.INFO),
    ("LOG_DESTINATION", "Where to display log output", sys.stdout),

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
    ("LONG_RUNNING", "Names of builders that should be considered for parallel execution", []),
    ("ASR_JOB_COUNT", "How many jobs to split a single ASR experiment into", 1),
    ("KWS_JOB_COUNT", "How many jobs to split a single KWS experiment into", 1),
    ("TEST_ASR", "Just run the first parallel job, for testing purposes", False),
    BoolVariable("THREADED_SUBMIT_NODE", "Run parallel jobs using multiple cores", False),
    BoolVariable("TORQUE_SUBMIT_NODE", "Run parallel jobs using Torque (e.g. invoking scons on yetisubmit.cc.columbia.edu)", False),
    
    # Torque-specific variables
    ("TORQUE_TIME", "Maximum running time for each torque job", "11:30:00"),
    ("TORQUE_MEMORY", "Maximum memory usage for each Torque job", "3500mb"),
    ("TORQUE_INTERVAL", "How often to check whether all pending Torque jobs have finished", 60),
    ("TORQUE_LOG", "Where Torque will create log files", "work/"),

    # Torque bookkeeping variables (don't set these yourself)
    BoolVariable("WORKER_NODE", "Tracks whether this is a worker node", False),
    ("JOB_ID", "Tracks which parallel job this is", 0),
    ("SCONSIGN_FILE", "Redirects dependency database when running on a worker node", None),
    
    # these variables define the locations of various tools and data
    ("IBM_MODELS", "Acoustic models and related files provided by IBM", "${BASE_PATH}/ibm_models"),
    ("LORELEI_SVN", "Checkout of SVN repository hosted on lorelei", "${BASE_PATH}/lorelei_svn"),
    ("ATTILA_PATH", "IBM's ASR system", "${BASE_PATH}/VT-2-5-babel"),
    ("ATTILA_INTERPRETER", "", "${ATTILA_PATH}/tools/attila/attila"),
    ("F4DE_PATH", "", None),
    ("INDUSDB_PATH", "", "${BASE_PATH}/lorelei_resources/IndusDB"),
    ("SEQUITUR_PATH", "", ""),
    ("JAVA_NORM", "", "${BABEL_REPO}/KWS/examples/babel-dryrun/javabin"),
    ("LIBRARY_OVERLAY", "", "${OVERLAY}/lib:${OVERLAY}/lib64:${LORELEI_TOOLS}/boost_1_49_0/stage/lib/"),
    ("PYTHON_INTERPRETER", "", None),
    ("SCORE_SCRIPT", "", None),
    ("SCLITE_BINARY", "", "${BASE_PATH}/sctk-2.4.5/bin/sclite"),
    ("LORELEI_TOOLS", "", "${BASE_PATH}/lorelei_tools"),
    ("CN_KWS_SCRIPTS", "", "${BASE_PATH}/lorelei_svn/tools/cn-kws/scripts"),
    ("PYCFG_PATH", "", "${BASE_PATH}/py-cfg"),
    
    # these variables all have default definitions in terms of the previous, but may be overridden as needed
    ("LANGUAGE_PACK_PATH", "", "${BASE_PATH}/language_packs"),
    ("STRIPPED_LANGUAGE_PACK_PATH", "", "${BASE_PATH}/stripped_language_packs"),
    ("BABEL_RESOURCES", "", "${BASE_PATH}/lorelei_resources"),
    ("PYTHON", "", "/usr/bin/python"),
    ("PERL", "", "/usr/bin/perl"),    
    ("PERL_LIBRARIES", "", os.environ.get("PERL5LIB", "")),
    ("G2P", "", "g2p.py"),
    ("G2P_PATH", "", "/home/tom/local/python/lib/python2.7/site-packages/"),
    ("BABEL_BIN_PATH", "", "${LORELEI_SVN}/tools/kws/bin64"),
    ("BABEL_SCRIPT_PATH", "", "${LORELEI_SVN}/tools/kws/scripts"),
    ("F4DE_PATH", "", "${BABEL_RESOURCES}/F4DE"),
    ("INDUS_DB", "", "${BABEL_RESOURCES}/IndusDB"),
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
    ("TRANSPARENT", "", "'<s>,</s>,~SIL,<epsilon>'"),
    ("ADD_DELETE", "", 5),
    ("ADD_INSERT", "", 5),
    ("NBESTP2P", "", 2000),
    ("MINPHLENGTH", "", 2),
    ("PRINT_WORDS_THRESH", "", "1e-10"),
    ("PRINT_EPS_THRESH", "", "1e-03"),
    ("PRUNE", "", 10),
    ("RESCORE_BEAM", "", 1.5),
    ("LOWER_CASE", "", False),
    
    # ASR-related variables
    ("MODEL_PATH", "", "${IBM_MODELS}/${BABEL_ID}/${PACK}/models"),
    ("PHONE_FILE", "", "${MODEL_PATH}/pnsp"),
    ("PHONE_SET_FILE", "", "${MODEL_PATH}/phonesset"),
    ("TAGS_FILE", "", "${MODEL_PATH}/tags"),
    ("TREE_FILE", "", "${MODEL_PATH}/tree"),
    ("MLP_FILE", "", "${MODEL_PATH}/*.mlp"),
    ("TOPO_FILE", "", "${MODEL_PATH}/topo.*"),
    ("TOPO_TREE_FILE", "", "${MODEL_PATH}/topotree"),
    ("PRONUNCIATIONS_FILE", "", "${MODEL_PATH}/dict.test"),
    ("VOCABULARY_FILE", "", "${MODEL_PATH}/vocab"),
    ("LANGUAGE_MODEL_FILE", "", "${MODEL_PATH}/lm.2gm.arpabo.gz"),
    ("SAMPLING_RATE", "", 8000),
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
    ("PCM_PATH", "", "${LANGUAGE_PACK_PATH}/${BABEL_ID}_resampled"),
    ("DATABASE_FILE", "", "${IBM_MODELS}/${BABEL_ID}/${PACK}/segment/*db"),
    ("CMS_PATH", "", "${IBM_MODELS}/${BABEL_ID}/${PACK}/adapt/cms"),
    ("FMLLR_PATH", "", "${IBM_MODELS}/${BABEL_ID}/${PACK}/adapt/fmllr"),
    ("ECF_FILE", "", "${INDUSDB_PATH}/*babel${BABEL_ID}*conv-dev/*.scoring.ecf.xml"),
    ("DEV_KEYWORD_FILE", "", "${INDUSDB_PATH}/*babel${BABEL_ID}*conv-dev.kwlist2.xml"),
    ("EVAL_KEYWORD_FILE", "", "data/eval_keyword_lists/*babel${BABEL_ID}*conv-eval.kwlist*.xml"),
    ("RTTM_FILE", "", "${INDUSDB_PATH}/*babel${BABEL_ID}*conv-dev/*mit*rttm"),
)

# initialize logging system
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.ERROR)

# initialize build environment
env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default", "textfile"] + [x.TOOLS_ADD for x in [babel_tools, morfessor_tools,
                                                                         #mila_tools, almor_tools, 
                                                                         pycfg_tools,
                                                                         asr_tools, kws_tools, vocabulary_tools, g2p_tools, scons_tools,
                                                                         openfst_tools,
                                                                     ]],
                  )

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




#
# Begin defining the actual experiments and dependencies
#
model_names = ["prefix_suffix", "agglutinative"]
all_texts = []
for language, properties in env["LANGUAGES"].iteritems():
    
    env.Replace(BABEL_ID=properties["BABEL_ID"])
    env.Replace(LANGUAGE_NAME=language)
    env.Replace(LOCALE=properties.get("LOCALE"))
    env.Replace(NON_ACOUSTIC_GRAPHEMES=properties.get("NON_ACOUSTIC_GRAPHEMES"))
    env.Replace(NON_WORD_PATTERN=".*(_|\<).*")
    env.Replace(FORCE_SPLIT=["-"])
    
    packs = {}
    #resampled_pack = env.Resample("work/resampled_packs/${BABEL_ID}.tgz", ["${LANGUAGE_PACK_PATH}/${BABEL_ID}.tgz", "data/down6x.filt"])
    stripped_pack = env.FilterTar("work/stripped_packs/${BABEL_ID}.tgz", ["${LANGUAGE_PACK_PATH}/${BABEL_ID}.tgz", env.Value(r".*transcription.*")])
    
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

    # all_texts += packs.values()
    
    # dev_keyword_file = env.Glob(env.subst("${DEV_KEYWORD_FILE}"))
    # #dev_keyword_list = env.KeywordsToList("work/keywords/${LANGUAGE_NAME}_dev.txt", dev_keyword_file)
    # #dev_keyword_text_file = env.KeywordXMLToText("work/adaptor_grammar/${LANGUAGE_NAME}_dev_keywords.txt", dev_keyword_file)
    # #eval_keyword_file = env.Glob(env.subst("${EVAL_KEYWORD_FILE}"))
    # #eval_keyword_list = env.KeywordsToList("work/keywords/${LANGUAGE_NAME}_eval.txt", eval_keyword_file)
    
    # keyword_lists = []
    # for kwfile in env.Glob("${INDUSDB_PATH}/IARPA-babel${BABEL_ID}*kwlist*xml") + env.Glob(env.subst("${EVAL_KEYWORD_FILE}")):
    #     basename = os.path.splitext(os.path.basename(kwfile.rstr()))[0]
    #     keyword_lists.append(env.KeywordsToList("work/keyword_lists/${LANGUAGE_NAME}/%s.txt" % basename, kwfile))
            
    # testing_lists = env.Glob("data/testing_lists/${BABEL_ID}*")
    
    # for training_words in env.Glob("data/training_lists/${BABEL_ID}*"):
    #     continue
    #     target_base = os.path.splitext(training_words.name)[0]
        
    #     cleaned_training_words = env.CleanWords("work/word_lists/${TARGET_BASE}.txt", training_words, TARGET_BASE=target_base, LOWER_CASE=True)
        
    #     segmented = env.MorfessorBabelExperiment(target_base,
    #                                              properties.get("NON_ACOUSTIC_GRAPHEMES", []),
    #                                              "web-data",
    #                                              cleaned_training_words + keyword_lists + testing_lists,
    #                                         )
        
    #     for model_name in model_names:            
    #         continue
    #         segmented = env.AdaptorGrammarBabelExperiment(target_base,
    #                                                       model_name,
    #                                                       properties.get("NON_ACOUSTIC_GRAPHEMES", []),
    #                                                       cleaned_training_words + keyword_lists
    #         )
    
    for pack, data in packs.iteritems():

        env.Replace(PACK=pack)        
        baseline_vocabulary = env.File("${VOCABULARY_FILE}")
        baseline_pronunciations = env.File("${PRONUNCIATIONS_FILE}")
        segmentations = {}

        if os.path.exists(pjoin(env.subst("${IBM_MODELS}/${BABEL_ID}/${PACK}"))):
            baseline_language_model = env.Glob("${IBM_MODELS}/${BABEL_ID}/${PACK}/models/*.arpabo.gz")[0]
            env.Replace(ACOUSTIC_WEIGHT=properties.get("ACOUSTIC_WEIGHT", .10))
            baseline_asr_output = env.RunASR("work/asr_experiments/${LANGUAGE_NAME}/${PACK}/baseline", baseline_vocabulary, baseline_pronunciations, baseline_language_model)
            #baseline_kws_output = env.RunKWS("work/kws_experiments/${LANGUAGE_NAME}/${PACK}/baseline",
            #                                 [x[1] for x in baseline_asr_output[1:]], baseline_vocabulary, baseline_pronunciations, dev_keyword_file)
            
            # for model_name, segmentation in segmentations.iteritems():
            #     env.Replace(MODEL=model_name)
            #     segmented_pronunciations_training, morphs = env.SegmentedPronunciations(["work/pronunciations/${LANGUAGE_NAME}_${PACK}_${MODEL}_segmented.txt",
            #                                                                              "work/pronunciations/${LANGUAGE_NAME}_${PACK}_${MODEL}_morphs.txt"],
            #                                                                             [baseline_pronunciations, segmentation])
            #     if properties.get("GRAPHEMIC", False):
            #         morph_pronunciations = env.GraphemicPronunciations("work/pronunciations/${LANGUAGE_NAME}_${PACK}_${MODEL}_morph_pronunciations.txt", morphs)
            #     else:
            #         g2p_segmented_model = env.TrainG2P("work/pronunciations/${LANGUAGE_NAME}_${PACK}_${MODEL}_segmented_model_1.txt", segmented_pronunciations_training)
            #         morph_pronunciations = env.ApplyG2P("work/pronunciations/${LANGUAGE_NAME}_${PACK}_${MODEL}_morph_pronunciations.txt", [g2p_segmented_model, morphs])
            #     segmented_vocabulary, segmented_pronunciations = env.PronunciationsToVocabDict(
            #         ["work/asr_input/${LANGUAGE_NAME}_${PACK}_${MODEL}_vocabulary.txt", "work/asr_input/${LANGUAGE_NAME}_${PACK}_${MODEL}_pronunciations.txt"],
            #         [morph_pronunciations, baseline_pronunciations, env.Value(properties.get("GRAPHEMIC", False))])

            #     segmented_training_text = env.SegmentTranscripts("work/segmented_training/${LANGUAGE_NAME}_${PACK}_${MODEL}.txt", [data, segmentation])
            #     segmented_language_model = env.TrainLanguageModel("work/asr_input/${LANGUAGE_NAME}_${PACK}_${MODEL}_languagemodel_segmented.arpabo.gz",
            #                                                       [segmented_training_text, Value(2)])
                #morfessor_asr_output = env.RunASR("work/asr_experiments/${LANGUAGE_NAME}/${PACK}/morfessor", segmented_vocabulary, segmented_pronunciations, segmented_language_model)
                #morfessor_kws_output = env.RunKWS("work/kws_experiments/${LANGUAGE_NAME}/${PACK}/morfessor", [x[1] for x in morfessor_asr_output[1:]], segmented_vocabulary, segmented_pronunciations, dev_keyword_file)
                # training_vocabulary_file = env.TextToVocabulary("work/vocabularies/${LANGUAGE_NAME}/training.txt.gz",
    #                                                 training_text)
    # dev_vocabulary_file = env.TextToVocabulary("work/vocabularies/${LANGUAGE_NAME}/development.txt.gz",
    #                                            dev_text)
    #language_model_file = env.Glob("${LANGUAGE_MODEL_FILE}")[0]
    #    dnet = env.GraphFile("${GRAPH_FILE}", [vocabulary_file, pronunciations_file, language_model_file])
        #warp = env.VTLN("${WARP_FILE}", [])
        #env.RunASR([dnet])
        #(asr_output, asr_score) = env.RunASR("baseline", LANGUAGE_ID=babel_id, ACOUSTIC_WEIGHT=properties["ACOUSTIC_WEIGHT"])    
    #full_transcripts = env.ExtractTranscripts("work/full_transcripts/${LANGUAGE_NAME}.xml.gz", ["${STRIPPED_LANGUAGE_PACK_PATH}/${BABEL_ID}.tgz", Value({})])
    #limited_transcripts = env.ExtractTranscripts("work/training_transcripts/${LANGUAGE_NAME}_training.xml.gz", ["${STRIPPED_LANGUAGE_PACK_PATH}/${BABEL_ID}.tgz",
    #Value({"PATTERN" : r".*sub-train.*transcription.*txt"})])
    #limited_data = env.TranscriptsToData("work/training_data/${LANGUAGE_NAME}.xml.gz", [limited_transcripts, Value({})])
    #full_data = env.TranscriptsToData("work/full_data/${LANGUAGE_NAME}.xml.gz", [full_transcripts, Value({})])
    #morfessor, morfessor_model = env.TrainMorfessor(["work/morfessor/${LANGUAGE_NAME}.xml.gz", "work/morfessor/${LANGUAGE_NAME}.model"], limited_data)
    #continue
    #terms = env.Glob("${INDUSDB_PATH}/IARPA-babel${BABEL_ID}*-dev.kwlist*.xml")[0]
    #segmented_terms = env.ApplyMorfessor(["work/segmented_terms/${LANGUAGE_NAME}.txt"], [morfessor_model, terms])
    #segmented_pronunciations_training, morphs = env.SegmentedPronunciations(["work/pronunciations/${LANGUAGE_NAME}_${PACK}_segmented.txt",
    #                                                                         "work/pronunciations/${LANGUAGE_NAME}_${PACK}_morphs.txt"], [pronunciations_file, morfessor])
    #g2p_segmented_model = env.TrainG2P("work/pronunciations/${LANGUAGE_NAME}_${PACK}_morfessor_model_1.txt", segmented_pronunciations_training)
    #for i in range(2, 5):
    #    g2p_segmented_model = env.TrainG2P("work/pronunciations/${LANGUAGE_NAME}_segmented_model_%d.txt" % (i), [g2p_segmented_model, segmented_pronunciations_training])
    #g2p_model = env.TrainG2P("work/pronunciations/${LANGUAGE_NAME}_morfessor_model_1.txt", pronunciations_file)
    #for i in range(2, 5):
    #    g2p_model = env.TrainG2P("work/pronunciations/${LANGUAGE_NAME}_model_%d.txt" % (i), [g2p_model, pronunciations_file])
    #ibm_g2p_model = env.File("${LORELEI_SVN}/${BABEL_ID}/LimitedLP/models/g2p.4.model")
    #morph_pronunciations = env.ApplyG2P("work/pronunciations/${LANGUAGE_NAME}_${PACK}_morfessor_pronunciations.txt", [g2p_segmented_model, morphs])
    #segmented_vocabulary, segmented_pronunciations = env.PronunciationsToVocabDict(
    #    ["work/asr_input/${LANGUAGE_NAME}_${PACK}_morfessor_vocabulary.txt", "work/asr_input/${LANGUAGE_NAME}_${PACK}_morfessor_pronunciations.txt"],
    #    [morph_pronunciations, pronunciations_file])
    #segmented_training_text = env.SegmentTranscripts("work/segmented_training/${LANGUAGE_NAME}.txt.gz", [training_text, morfessor])
    #segmented_language_model = env.IBMTrainLanguageModel("work/asr_input/${LANGUAGE_NAME}/languagemodel_segmented.arpabo.gz", [segmented_training_text, Value(2)])
    #morfessor_asr_output = env.RunASR("work/asr_experiments/${LANGUAGE_NAME}/morfessor", segmented_vocabulary, segmented_pronunciations, segmented_language_model)
    #morfessor_kws_output = env.RunCascadeKWS("work/kws_experiments/${LANGUAGE_NAME}/morfessor", [baseline_asr_output, morfessor_asr_output], segmented_vocabulary, segmented_pronunciations)
    #
    # Adaptor Grammar Experiments
    #
    #query_file = env.QueryFile() # just search terms, one per line
    #word_dictionary = env.WordDictionary() # running(01) runn+ +ing
    #hyb_dict = env.HybDict() # runn+(01) runn+
    #seg_transcripts, seg_vocabulary, seg_pronunciations, seg_language_model = env.ApplySegmentations(morfessor, training_vocabulary_file, pronunciations_file)
    # IBMTrainLanguageModel
#     for run in range(1, env["RUNS"] + 1):
#         env.Replace(RUN=str(run))
#         if properties.get("VLLP", False):
#             very_limited_data = env.StmToData("work/very_limited_data/${LANGUAGE}_${RUN}.xml.gz", 
#                                               ["${BABEL_DATA_PATH}/LPDefs.20141006.tgz", env.Value(".*IARPA-babel%d.*VLLP.training.transcribed.stm" % babel_id)])
#         else:
#             very_limited_data = env.GenerateDataSubset("work/very_limited_data/${LANGUAGE}_${RUN}.xml.gz", [full_data, Value({"RANDOM" : True, "WORDS" : 30000})])
#         has_morphology = os.path.exists(env.subst("data/${LANGUAGE}_morphology.txt"))
#         if has_morphology:
#             very_limited_data_morph = env.AddMorphology("work/data/${LANGUAGE}/very_limited_morph_${RUN}.xml.gz", [very_limited_data, "data/${LANGUAGE}_morphology.txt"])
#             limited_data_morph = env.AddMorphology("work/data/${LANGUAGE}/limited_morph_${RUN}.xml.gz", [limited_data, "data/${LANGUAGE}_morphology.txt"])
#         templates = env.Glob("data/grammar_fragments/templates/${LANGUAGE}_*.txt")
#         print templates
#         continue
#         characters = env.CharacterProductions("work/character_productions/${LANGUAGE}_${RUN}.txt", very_limited_data)
#         data = env.MorphologyData("work/ag_data/${LANGUAGE}_${RUN}.txt", very_limited_data)
#         arguments = Value({"LANGUAGE" : language, "SSN" : "None", "RUN" : str(run)})
#         cfg = env.ComposeGrammars("work/ag_models/${LANGUAGE}_${RUN}_plain.txt", [template, characters])
#         parses, grammar, trace_file = env.RunPYCFG([cfg, data, arguments])
#         py = env.MorphologyOutputToEMMA([parses, arguments])                        
#         guess, gold = env.PrepareDatasetsForEMMA([py, very_limited_data_morph, arguments])                                                                 
#         results.append(env.RunEMMA([guess, gold, arguments]))
#         for ssn in env.Glob("data/grammar_fragments/${LANGUAGE}_*.txt"):
#             stem = re.match(r"^%s_(.*).txt$" % language, os.path.basename(ssn.rstr())).groups()[0]
#             arguments = Value({"LANGUAGE" : language,
#                                "SSN" : stem,
#                                "RUN" : str(run),
#                            })
#             cfg = env.ComposeGrammars("work/ag_models/${LANGUAGE}_${RUN}_%s.txt" % stem, [template, ssn, characters])
#             parses, grammar, trace_file = env.RunPYCFG([cfg, data, arguments])
#             py = env.MorphologyOutputToEMMA([parses, arguments])                        
#             guess, gold = env.PrepareDatasetsForEMMA([py, very_limited_data_morph, arguments])                                                                 
#             results.append(env.RunEMMA([guess, gold, arguments]))
#         morfessor = env.TrainMorfessor("work/morfessor/${LANGUAGE}_${RUN}.xml.gz", very_limited_data_morph)        
#         if has_morphology:
#             guess, gold = env.PrepareDatasetsForEMMA([morfessor, very_limited_data_morph, Value({"LANGUAGE" : "$LANGUAGE", "MODEL" : "MORFESSOR", "RUN" : str(run)})])
#             results.append(env.RunEMMA([guess, gold, Value({"LANGUAGE" : "$LANGUAGE", "MODEL" : "MORFESSOR", "RUN" : str(run)})]))
# env.CollateResults("work/results.txt", results)
# env.VocabularyComparison("work/vocabulary_comparison.txt", all_texts)
