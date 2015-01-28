import os
import os.path
import sys
from SCons.Tool import textfile
from glob import glob
import logging
import time
import re
from os.path import join as pjoin
import babel_tools
import scala_tools
import morfessor_tools
import emma_tools
import trmorph_tools
import sfst_tools
import evaluation_tools
import almor_tools
import mila_tools
import pycfg_tools
import torque_tools
import asr_tools
import kws_tools
import vocabulary_tools
import g2p_tools
from scons_tools import threaded_run
from torque_tools import torque_run

vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 130),
    ("LOCAL_PATH", "", False),
    ("LANGUAGES", "", {}),
    BoolVariable("DEBUG", "", True),
    BoolVariable("HAS_TORQUE", "", False),
    ("ANNEAL_INITIAL", "", 5),
    ("ANNEAL_FINAL", "", 1),
    ("ANNEAL_ITERATIONS", "", 0),
    ("LANGUAGE_PACK_PATH", "", None),
    ("STRIPPED_LANGUAGE_PACK_PATH", "", None),
    ("BABEL_DATA_PATH", "", None),
    ("RUNS", "", 1),
    ("TORQUE_INTERVAL", "", 60),

    ("INDUSDB_PATH", "", None),

    ("CABAL", "", "/home/tom/.cabal"),
    ("PYCFG_PATH", "", ""),
    ("MAXIMUM_SENTENCE_LENGTH", "", 20),
    ("HASKELL_PATH", "", ""),
    ("PENN_TREEBANK_PATH", "", ""),
    ("LPSOLVE_PATH", "", ""),
    
    # parameters shared by all models
    ("NUM_ITERATIONS", "", 1),
    ("NUM_SAMPLES", "", 1),
    #("SAVE_EVERY", "", 1),
    ("TOKEN_BASED", "", [True, False]),

    # tagging parameters
    ("NUM_TAGS", "", 45),
    ("MARKOV", "", 1),
    ("TRANSITION_PRIOR", "", .1),
    ("EMISSION_PRIOR", "", .1),
    BoolVariable("SYMMETRIC_TRANSITION_PRIOR", "", True),
    BoolVariable("SYMMETRIC_EMISSION_PRIOR", "", True),


    # morphology parameters
    ("PREFIX_PRIOR", "", 1),
    ("SUFFIX_PRIOR", "", 1),
    ("SUBMORPH_PRIOR", "", 1),
    ("WORD_PRIOR", "", 1),
    ("TAG_PRIOR", "", .1),
    ("BASE_PRIOR", "", 1),    
    ("ADAPTOR_PRIOR_A", "", 0),
    ("ADAPTOR_PRIOR_B", "", 100),
    ("CACHE_PROBABILITY", "", 100),
    ("RULE_PRIOR", "", 1),

    # these variables determine what experiments are performed
    ("LANGUAGES", "", {}),
    ("RUN_ASR", "", False),
    ("RUN_KWS", "", False),
    ("EVALUATE_PRONUNCIATIONS", "", False),
    ("EXPANSION_SIZES", "", []),
    ("EXPANSION_WEIGHTS", "", []),

    # these variables determine how parallelism is exploited
    BoolVariable("HAS_TORQUE", "", True),
    BoolVariable("IS_THREADED", "", True),
    ("MAXIMUM_LOCAL_JOBS", "", 4),
    ("MAXIMUM_TORQUE_JOBS", "", 200),

    # these variables define the locations of various tools and data
    ("BASE_PATH", "", None),
    ("OVERLAY", "", "${BASE_PATH}/local"),
    #("LANGUAGE_PACKS", "", "${BASE_PATH}/language_transcripts"),
    ("IBM_MODELS", "", "${BASE_PATH}/ibm_models"),
    ("LORELEI_SVN", "", "${BASE_PATH}/lorelei_svn"),
    ("ATTILA_PATH", "", "${BASE_PATH}/VT-2-5-babel"),
    ("VOCABULARY_EXPANSION_PATH", "", "${BASE_PATH}/vocabulary_expansions"),
    ("INDUSDB_PATH", "", "${BASE_PATH}/lorelei_resources/IndusDB"),
    ("SEQUITUR_PATH", "", ""),
    ("ATTILA_INTERPRETER", "", "${ATTILA_PATH}/tools/attila/attila"),
    ("F4DE_PATH", "", None),
    ("JAVA_NORM", "", "${BABEL_REPO}/KWS/examples/babel-dryrun/javabin"),
    ("OVERLAY", "", None),
    ("LIBRARY_OVERLAY", "", "${OVERLAY}/lib:${OVERLAY}/lib64:${LORELEI_TOOLS}/boost_1_49_0/stage/lib/"),
    ("LOG_LEVEL", "", logging.INFO),
    ("LOG_DESTINATION", "", sys.stdout),    
    ("PYTHON_INTERPRETER", "", None),
    ("SCORE_SCRIPT", "", None),
    ("SCLITE_BINARY", "", "${BASE_PATH}/sctk-2.4.5/bin/sclite"),
    ("LORELEI_TOOLS", "", "${BASE_PATH}/lorelei_tools"),
    
    # these variables all have default definitions in terms of the previous, but may be overridden as needed
    ("PYTHON", "", "/usr/bin/python"),
    ("PERL", "", "/usr/bin/perl"),    
    ("PERL_LIBRARIES", "", os.environ.get("PERL5LIB", "")),
    ("G2P", "", "g2p.py"),
    ("G2P_PATH", "", "/home/tom/local/python/lib/python2.7/site-packages/"),
    ("BABEL_BIN_PATH", "", "${LORELEI_SVN}/tools/kws/bin64"),
    ("BABEL_SCRIPT_PATH", "", "${LORELEI_SVN}/tools/kws/scripts"),
    ("F4DE", "", "${BABEL_RESOURCES}/F4DE"),
    ("INDUS_DB", "", "${BABEL_RESOURCES}/IndusDB"),
    ("WRD2PHLATTICE", "", "${BABEL_BIN_PATH}/wrd2phlattice"),
    ("BUILDINDEX", "", "${BABEL_BIN_PATH}/buildindex"),
    ("BUILDPADFST", "", "${BABEL_BIN_PATH}/buildpadfst"),
    ("LAT2IDX", "", "${BABEL_BIN_PATH}/lat2idx"),
    ("FSTCOMPILE", "", "${OVERLAY}/bin/fstcompile"),
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
    ("BABELSCORER", "", "${F4DE_PATH}/bin/BABEL13_Scorer"),
    
    # all configuration information for ASR
    BoolVariable("TORQUE_SUBMIT_NODE", "", False),
    BoolVariable("TORQUE_WORKER_NODE", "", False),
    BoolVariable("THREADED_SUBMIT_NODE", "", False),
    BoolVariable("THREADED_WORKER_NODE", "", False),
    
    ("ASR_JOB_COUNT", "", 1),
    ("KWS_JOB_COUNT", "", 1),
    ("JOB_ID", "", 0),
    
    ("MODEL_PATH", "", "${IBM_MODELS}/${BABEL_ID}/LLP/models"),
    ("PHONE_FILE", "", "${MODEL_PATH}/pnsp"),
    ("PHONE_SET_FILE", "", "${MODEL_PATH}/phonesset"),
    ("TAGS_FILE", "", "${MODEL_PATH}/tags"),
    ("TREE_FILE", "", "${MODEL_PATH}/tree"),
    ("TOPO_FILE", "", "${MODEL_PATH}/topo.tied"),
    ("TOPO_TREE_FILE", "", "${MODEL_PATH}/topotree"),
    ("PRONUNCIATIONS_FILE", "", "${MODEL_PATH}/dict.test"),
    ("VOCABULARY_FILE", "", "${MODEL_PATH}/vocab"),
    ("LANGUAGE_MODEL_FILE", "", "${MODEL_PATH}/lm.2gm.arpabo.gz"),
    ("SAMPLING_RATE", "", 8000),
    ("FEATURE_TYPE", "", "plp"),
    ("MEL_FILE", "", "${MODEL_PATH}/mel"),
    ("LDA_FILE", "", "${MODEL_PATH}/30.mat"),
    ("PRIORS_FILE", "", "${MODEL_PATH}/priors"),
    
    ("OUTPUT_PATH", "", "work/asr/${LANGUAGE_NAME}/${EXPERIMENT_NAME}"),
    #("WARP_FILE", "", "${OUTPUT_PATH}/warp.lst.${JOB_ID}"),
    ("WARP_FILE", "", "${IBM_MODELS}/${BABEL_ID}/LLP/adapt/warp.lst"),
    ("GRAPH_FILE", "", "${OUTPUT_PATH}/dnet.bin.gz"),
    ("CTM_PATH", "", "${OUTPUT_PATH}/ctm"),
    ("LATTICE_PATH", "", "${OUTPUT_PATH}/lat"),
    ("TEXT_PATH", "", "${OUTPUT_PATH}/text"),
    ("PCM_PATH", "", "${LANGUAGE_PACK_PATH}/${BABEL_ID}"),
    # ("TRFS_FILE", "", "${MODEL_PATH}/phoneset"),
    # ("TR_FILE", "", "${MODEL_PATH}/phoneset"),
    # ("CTX_FILE", "", "${MODEL_PATH}/phoneset"),
    # ("GS_FILE", "", "${MODEL_PATH}/phoneset"),
    # ("MS_FILE", "", "${MODEL_PATH}/phoneset"),
    # ("FS_FILE", "", "${MODEL_PATH}/phoneset"),

    # ("LAT_OPATH", "", "${MODEL_PATH}/phoneset"),
    # ("TXT_PATH", "", "${MODEL_PATH}/phoneset"),
    ("DATABASE_FILE", "", "${IBM_MODELS}/${BABEL_ID}/LLP/segment/*db"),
     #"${LORELEI_SVN}/${BABEL_ID}/LimitedLP/asr/genSeg/babel${BABEL_ID}.dev.seg.*db"),
    ("CMS_PATH", "", "${IBM_MODELS}/${BABEL_ID}/LLP/adapt/cms"),
    ("FMLLR_PATH", "", "${IBM_MODELS}/${BABEL_ID}/LLP/adapt/fmllr"),


    ("KEYWORD_FILE", "", "${INDUSDB_PATH}/*babel${BABEL_ID}*.kwlist.xml"),
)

# initialize logging system
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.ERROR)

# initialize build environment
env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default", "textfile"] + [x.TOOLS_ADD for x in [babel_tools, evaluation_tools, scala_tools, morfessor_tools, emma_tools, 
                                                                         trmorph_tools, sfst_tools, mila_tools, almor_tools, pycfg_tools, torque_tools,
                                                                         asr_tools, kws_tools, vocabulary_tools, g2p_tools
                                                                     ]],
                  )


long_running = ["ASRTest", "LatticeToIndex", "TrainMorfessor", "RunPYCFG"]
if env["THREADED_SUBMIT_NODE"]:
    for b in long_running:
        env["BUILDERS"][b] = Builder(action=threaded_run)
elif env["TORQUE_SUBMIT_NODE"]:
    for b in long_running:
        env["BUILDERS"][b] = Builder(action=Action(torque_run, batch_key=True))


Help(vars.GenerateHelpText(env))


# env.Tag(node, X=y)
#"ASRTest" : Builder(action=Action(torque_run, batch_key=True)),


# don't print out lines longer than the terminal width
def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print s[:int(env["OUTPUT_WIDTH"]) - 10] + "..." + s[-7:]
    else:
        print s

env['PRINT_CMD_LINE_FUNC'] = print_cmd_line
env.Decider("timestamp-newer")


for language, properties in env["LANGUAGES"].iteritems():
    # for Zulu, replace "(\S)-(\S)" with "\1=\2"
    env.Replace(BABEL_ID=properties["BABEL_ID"])
    env.Replace(LANGUAGE_NAME=language)
    env.Replace(LOCALE=properties.get("LOCALE"))

    training_text = env.CollectText("work/texts/${LANGUAGE_NAME}/training_text.txt.gz",
                                    [env.subst("${STRIPPED_LANGUAGE_PACK_PATH}/${BABEL_ID}.tgz"), env.Value(".*sub-train/transcription.*txt")],
                                    )
    
    if os.path.exists(env.subst("data/${BABEL_ID}.txt")):
        #arguments = Value({"LANGUAGE" : language, "SSN" : "None", "RUN" : str(run)})
        segs, models = env.TrainMorfessor(["work/web_data_morphology/${LANGUAGE_NAME}_morfessor.xml.gz",
                                           "work/web_data_morphology/${LANGUAGE_NAME}_morfessor.model"], "data/${BABEL_ID}.txt")
        characters = env.CharacterProductions("work/character_productions/${LANGUAGE_NAME}.txt", "data/${BABEL_ID}.txt")
        cfg = env.ComposeGrammars("work/web_data_morphology/${LANGUAGE_NAME}_cfg.txt", ["data/grammar_templates/simple_prefix_suffix.txt", characters])
        data = env.MorphologyData("work/web_data_morphology/${LANGUAGE_NAME}_data.txt", "data/${BABEL_ID}.txt")
        #parses, grammar, trace_file =
        env.RunPYCFG("work/web_data_morphology/${LANGUAGE_NAME}_output.txt", [cfg, data])
    
        #continue
    # combined_text = env.CollectText("work/texts/${LANGUAGE_NAME}/combined.txt.gz",
    #                                 [env.subst("${STRIPPED_LANGUAGE_PACK_PATH}/${BABEL_ID}.tgz"), env.Value(".*(sub-train|dev)/transcription.*txt")],
    #                             )


    # dev_text = env.CollectText("work/texts/${LANGUAGE_NAME}/dev_text.txt.gz",
    #                            [env.subst("${STRIPPED_LANGUAGE_PACK_PATH}/${BABEL_ID}.tgz"), env.Value(".*dev/transcription.*txt")],
    #                            )

    # combined_vocabulary_file = env.TextToVocabulary("work/vocabularies/${LANGUAGE_NAME}/combined.txt.gz",
    #                                                 combined_text)

    # training_vocabulary_file = env.TextToVocabulary("work/vocabularies/${LANGUAGE_NAME}/training.txt.gz",
    #                                                 training_text)

    # dev_vocabulary_file = env.TextToVocabulary("work/vocabularies/${LANGUAGE_NAME}/development.txt.gz",
    #                                            dev_text)

    #if os.path.exists(pjoin(env.subst("${IBM_MODELS}/${BABEL_ID}"))):
    pronunciations_file = env.File("${PRONUNCIATIONS_FILE}")
    vocabulary_file = env.File("${VOCABULARY_FILE}")
    #language_model_file = env.Glob("${LANGUAGE_MODEL_FILE}")[0]
    #    dnet = env.GraphFile("${GRAPH_FILE}", [vocabulary_file, pronunciations_file, language_model_file])
        #warp = env.VTLN("${WARP_FILE}", [])
        #env.RunASR([dnet])
        #(asr_output, asr_score) = env.RunASR("baseline", LANGUAGE_ID=babel_id, ACOUSTIC_WEIGHT=properties["ACOUSTIC_WEIGHT"])    

    #if language == "english":
    #    full_transcripts = env.PennToTranscripts("work/full_transcripts/${LANGUAGE}.xml.gz", ["${PENN_TREEBANK_PATH}", Value({})])
    #    full_data = env.TranscriptsToData("work/full_data/${LANGUAGE}.xml.gz", [full_transcripts, Value({})])
    #    limited_data = env.GenerateDataSubset("work/training_data/${LANGUAGE}.xml.gz", [full_data, Value({"RANDOM" : True, "WORDS" : 100000})])

    baseline_vocabulary = env.File("${IBM_MODELS}/${BABEL_ID}/LLP/models/vocab")
    baseline_pronunciations = env.File("${IBM_MODELS}/${BABEL_ID}/LLP/models/dict.test")
    baseline_language_model = env.File("${IBM_MODELS}/${BABEL_ID}/LLP/models/lm.2gm.arpabo.gz")
    
    baseline_asr_output = env.RunASR("work/asr_experiments/${LANGUAGE_NAME}/baseline", baseline_vocabulary, baseline_pronunciations, baseline_language_model)
    baseline_kws_output = env.RunKWS("work/kws_experiments/${LANGUAGE_NAME}/baseline", baseline_asr_output)


    continue
    #elif os.path.exists(env.subst("${LANGUAGE_PACK_PATH}/${BABEL_ID}.tgz")):
    full_transcripts = env.ExtractTranscripts("work/full_transcripts/${LANGUAGE_NAME}.xml.gz", ["${STRIPPED_LANGUAGE_PACK_PATH}/${BABEL_ID}.tgz", Value({})])
    limited_transcripts = env.ExtractTranscripts("work/training_transcripts/${LANGUAGE_NAME}_training.xml.gz", ["${STRIPPED_LANGUAGE_PACK_PATH}/${BABEL_ID}.tgz",
                                                                                                                Value({"PATTERN" : r".*sub-train.*transcription.*txt"})])
    limited_data = env.TranscriptsToData("work/training_data/${LANGUAGE_NAME}.xml.gz", [limited_transcripts, Value({})])
        
    full_data = env.TranscriptsToData("work/full_data/${LANGUAGE_NAME}.xml.gz", [full_transcripts, Value({})])


    morfessor, morfessor_model = env.TrainMorfessor(["work/morfessor/${LANGUAGE_NAME}.xml.gz", "work/morfessor/${LANGUAGE_NAME}.model"], limited_data)
    terms = env.Glob("${INDUSDB_PATH}/IARPA-babel${BABEL_ID}*-dev.kwlist*.xml")[0]
    segmented_terms = env.ApplyMorfessor(["work/segmented_terms/${LANGUAGE_NAME}.txt"], [morfessor_model, terms])
    
    segmented_pronunciations_training, morphs = env.SegmentedPronunciations(["work/pronunciations/${LANGUAGE_NAME}_segmented.txt",
                                                                             "work/pronunciations/${LANGUAGE_NAME}_morphs.txt"], [pronunciations_file, morfessor])

    g2p_segmented_model = env.TrainG2P("work/pronunciations/${LANGUAGE_NAME}_segmented_model_1.txt", segmented_pronunciations_training)
    #for i in range(2, 5):
    #    g2p_segmented_model = env.TrainG2P("work/pronunciations/${LANGUAGE_NAME}_segmented_model_%d.txt" % (i), [g2p_segmented_model, segmented_pronunciations_training])

    g2p_model = env.TrainG2P("work/pronunciations/${LANGUAGE_NAME}_model_1.txt", pronunciations_file)
    #for i in range(2, 5):
    #    g2p_model = env.TrainG2P("work/pronunciations/${LANGUAGE_NAME}_model_%d.txt" % (i), [g2p_model, pronunciations_file])

    ibm_g2p_model = env.File("${LORELEI_SVN}/${BABEL_ID}/LimitedLP/models/g2p.4.model")
    
    morph_pronunciations = env.ApplyG2P("work/pronunciations/${LANGUAGE_NAME}_morph_pronunciations.txt", [g2p_segmented_model, morphs])
    
    segmented_vocabulary, segmented_pronunciations = env.PronunciationsToVocabDict(
        ["work/asr_input/${LANGUAGE_NAME}/vocabulary.txt", "work/asr_input/${LANGUAGE_NAME}/pronunciations.txt"],
        [morph_pronunciations, pronunciations_file])

    segmented_training_text = env.SegmentTranscripts("work/segmented_training/${LANGUAGE_NAME}.txt.gz", [training_text, morfessor])
    segmented_language_model = env.IBMTrainLanguageModel("work/asr_input/${LANGUAGE_NAME}/languagemodel_segmented.arpabo.gz", [segmented_training_text, Value(2)])

    morfessor_asr_output = env.RunASR("work/asr_experiments/${LANGUAGE_NAME}/morfessor", segmented_vocabulary, segmented_pronunciations, segmented_language_model)
    morfessor_kws_output = env.RunKWS("work/kws_experiments/${LANGUAGE_NAME}/morfessor", morfessor_asr_output)
    
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
