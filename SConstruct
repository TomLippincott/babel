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
from scons_tools import make_threaded_builder
from torque_tools import make_torque_builder
import scons_tools

vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 130),
    ("LOCAL_PATH", "", False),
    BoolVariable("DEBUG", "", True),
    ("BABEL_DATA_PATH", "", None),
    ("PYCFG_PATH", "", None),
    ("LONG_RUNNING", "", []),
    
    ("TRANSPARENT", "", "'<s>,</s>,~SIL,<epsilon>'"),
    ("ADD_DELETE", "", 5),
    ("ADD_INSERT", "", 5),
    ("NBESTP2P", "", 2000),
    ("MINPHLENGTH", "", 2),
    ("PRINT_WORDS_THRESH", "", "1e-10"),
    ("PRINT_EPS_THRESH", "", "1e-03"),
    ("PRUNE", "", 10),
    ("RESCORE_BEAM", "", 1.5),
    
    # these variables determine what experiments are performed
    ("LANGUAGES", "", {}),
    ("RUN_ASR", "", True),
    ("RUN_KWS", "", True),
    ("PROCESS_PACK", "", None),
    
    # py-cfg parameters
    ("NUM_SAMPLES", "", 1),
    ("NUM_ITERATIONS", "", 1000),
    ("ANNEAL_INITIAL", "", 3),
    ("ANNEAL_FINAL", "", 1),
    ("ANNEAL_ITERATIONS", "", 500),
    
    # these variables determine how parallelism is exploited    
    BoolVariable("WORKER_NODE", "", False),
    BoolVariable("TORQUE_SUBMIT_NODE", "", False),
    ("TORQUE_TIME", "", "11:30:00"),
    ("TORQUE_MEMORY", "", "3500mb"),
    ("TORQUE_INTERVAL", "", 60),
    ("TORQUE_LOG", "", "work/"),
    BoolVariable("THREADED_SUBMIT_NODE", "", False),
    ("ASR_JOB_COUNT", "", 1),
    ("TEST_ASR", "", False),
    ("KWS_JOB_COUNT", "", 1),
    ("JOB_ID", "", 0),
    ("SCONSIGN_FILE", "", None),
    
    # these variables define the locations of various tools and data
    ("BASE_PATH", "", None),
    ("LOCAL_PATH", "", "${BASE_PATH}/local"),
    ("OVERLAY", "", "${BASE_PATH}/local"),
    ("IBM_MODELS", "", "${BASE_PATH}/ibm_models"),
    ("LORELEI_SVN", "", "${BASE_PATH}/lorelei_svn"),
    ("ATTILA_PATH", "", "${BASE_PATH}/VT-2-5-babel"),
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
    ("CN_KWS_SCRIPTS", "", "${BASE_PATH}/lorelei_svn/tools/cn-kws/scripts"),
    
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
    ("BABELSCORER", "", "${F4DE_PATH}/KWSEval/tools/KWSEval/KWSEval.pl"),
    
    # all configuration information for ASR
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
                  tools=["default", "textfile"] + [x.TOOLS_ADD for x in [babel_tools, evaluation_tools, scala_tools, morfessor_tools, emma_tools, 
                                                                         trmorph_tools, sfst_tools, mila_tools, almor_tools, pycfg_tools,
                                                                         asr_tools, kws_tools, vocabulary_tools, g2p_tools, scons_tools,
                                                                     ]],
                  )

for b, t, s, ss in env["LONG_RUNNING"]:
    if env["WORKER_NODE"]:
        pass
    elif env["THREADED_SUBMIT_NODE"]:
        env["BUILDERS"][b] = make_threaded_builder(env["BUILDERS"][b], t, s, ss)
    elif env["TORQUE_SUBMIT_NODE"]:
        env["BUILDERS"][b] = make_torque_builder(env["BUILDERS"][b], t, s, ss)
if (env["WORKER_NODE"] or env["WORKER_NODE"]) and env.get("SCONSIGN_FILE", False):
    env.SConsignFile(env["SCONSIGN_FILE"])

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
env.Decider("timestamp-newer")

all_texts = []

for language, properties in env["LANGUAGES"].iteritems():
    # for Zulu, replace "(\S)-(\S)" with "\1=\2"
    env.Replace(BABEL_ID=properties["BABEL_ID"])
    env.Replace(LANGUAGE_NAME=language)
    env.Replace(LOCALE=properties.get("LOCALE"))

    packs = {}
    if "FLP" in properties.get("PACKS", []):
        packs["FLP"] = env.CollectText("work/texts/${LANGUAGE_NAME}_FLP.txt",
                                       [env.subst("${STRIPPED_LANGUAGE_PACK_PATH}/${BABEL_ID}.tgz"), env.Value(".*transcription.*txt")],
                                   )

    if "LLP" in properties.get("PACKS", []):
        packs["LLP"] = env.CollectText("work/texts/${LANGUAGE_NAME}_LLP.txt",
                                       [env.subst("${STRIPPED_LANGUAGE_PACK_PATH}/${BABEL_ID}.tgz"), env.Value(".*sub-train/transcription.*txt")],
                                   )
    
    if "VLLP" in properties.get("PACKS", []):
        packs["VLLP"] = env.StmToData("work/texts/${LANGUAGE_NAME}_VLLP.txt",
                                      ["${BASE_PATH}/LPDefs.20141006.tgz", env.Value(env.subst(".*IARPA-babel${BABEL_ID}.*.VLLP.training.transcribed.stm"))])
    all_texts += packs.values()
    
    dev_keyword_file = env.Glob(env.subst("${DEV_KEYWORD_FILE}"))[0]
    dev_keyword_text_file = env.KeywordXMLToText("work/ag_input/${LANGUAGE_NAME}_keywords.txt", dev_keyword_file)
    # eval_keyword_file = env.Glob("data/eval_keyword_lists/IARPA-babel${BABEL_ID}*eval.kwlist*.xml")

    # alp_vocab = env.File("data/op2/${BABEL_ID}/ALP/morphology/vocab")
    # segs, models = env.TrainMorfessor(["work/morfessor_segmentations/${LANGUAGE_NAME}_ALP.txt",
    #                                    "work/morfessor_models/${LANGUAGE_NAME}_ALP.model"], alp_vocab)
    # unseg = env.Unsegment("work/unsegmented/${LANGUAGE_NAME}_ALP_unseg.txt", segs)
    # segs = env.ApplyMorfessor("work/morfessor_segmentations/${LANGUAGE_NAME}_ALP_reseg.txt",
    #                           [models, unseg])
    # env.NormalizeMorfessorOutput("work/morphology/morfessor/${LANGUAGE_NAME}_ALP.txt", segs)
    # segs = env.ApplyMorfessor("work/morfessor_segmentations/${LANGUAGE_NAME}_ALP_dev_keywords.txt",
    #                           [models, dev_keyword_file])
    # env.NormalizeMorfessorOutput("work/morphology/morfessor/${LANGUAGE_NAME}_ALP_dev_keywords.txt", segs)
    # segs = env.ApplyMorfessor("work/morfessor_segmentations/${LANGUAGE_NAME}_ALP_eval_keywords.txt",
    #                           [models, eval_keyword_file])
    # env.NormalizeMorfessorOutput("work/morphology/morfessor/${LANGUAGE_NAME}_ALP_eval_keywords.txt", segs)
    
    for pack, data in packs.iteritems():
        if env.get("PROCESS_PACK", pack) != pack:
            continue
        env.Replace(PACK=pack)
        baseline_vocabulary = env.File("${VOCABULARY_FILE}")
        
        # temp_segs, models = env.TrainMorfessor(["work/morfessor_models/${LANGUAGE_NAME}_${PACK}.txt",
        #                                         "work/morfessor_models/${LANGUAGE_NAME}_${PACK}.model"], env.Glob("syl/babel${BABEL_ID}.${PACK}*bz2")[0])
        # dev_kw_segs = env.ApplyMorfessor("work/morfessor_segmentations/${LANGUAGE_NAME}_${PACK}_dev_keywords.txt",
        #                                  [models, dev_keyword_file])
        # unseg = env.Unsegment("work/unsegmented/${LANGUAGE_NAME}_${PACK}_unseg.txt", temp_segs)
        # segs = env.ApplyMorfessor("work/morfessor_segmentations/${LANGUAGE_NAME}_${PACK}.txt",
        #                           [models, unseg])
        # env.NormalizeMorfessorOutput("work/morphology/morfessor/${LANGUAGE_NAME}_${PACK}.txt", segs)
        # env.NormalizeMorfessorOutput("work/morphology/morfessor/${LANGUAGE_NAME}_${PACK}_dev_keywords.txt", dev_kw_segs)
        #eval_kw_segs = env.ApplyMorfessor("work/morfessor_segmentations/${LANGUAGE_NAME}_${PACK}_eval_keywords.txt",
        #                                  [models, eval_keyword_file])
        #env.NormalizeMorfessorOutput("work/morphology/morfessor/${LANGUAGE_NAME}_${PACK}_eval_keywords.txt", eval_kw_segs)

        # for other_vocab in env.Glob("data/web_data/${LANGUAGE_NAME}/${PACK}/*"):
        #     base = os.path.splitext(os.path.basename(other_vocab.rstr()))[0]
        #     other_segs = env.ApplyMorfessor("work/morfessor_segmentations/${LANGUAGE_NAME}_${PACK}_web_data.txt",
        #                                     [models, other_vocab])
        #     env.NormalizeMorfessorOutput("work/morphology/morfessor/${LANGUAGE_NAME}_${PACK}_web_data.txt", other_segs)
        #     pass
        
        #if len(eval_keyword_file) == 1:
            #eval_keyword_text_file = env.KeywordXMLToText("work/ag_input/${LANGUAGE_NAME}_keywords.txt", eval_keyword_file[0])
        #    kw_segs = env.ApplyMorfessor("work/eval_morfessor_segmentations/${LANGUAGE_NAME}_${PACK}_eval_keywords.txt",
        #                                 [models, eval_keyword_file[0]])
        #    env.NormalizeMorfessorOutput("work/morphology/morfessor/${LANGUAGE_NAME}_${PACK}_eval_keywords.txt", kw_segs)


        if pack in ["LLP", "VLLP", "FLP"] and False:
            characters = env.CharacterProductions("work/character_productions/${LANGUAGE_NAME}_${PACK}.txt", [data, dev_keyword_text_file])
            pycfg_data = env.MorphologyData("work/ag_input/${LANGUAGE_NAME}_${PACK}_data.txt", [data, Value(properties.get("LOWER_CASE", True))])
            for model in ["prefix_suffix"]: #, "prefix", "suffix"]: #, "agglutinative"]:
                env.Replace(MODEL=model)
                cfg = env.ComposeGrammars("work/ag_morphology/${LANGUAGE_NAME}/${MODEL}/${PACK}_cfg.txt",
                                          ["data/grammar_templates/simple_${MODEL}.txt", characters])
                pycfg = env.RunPYCFG(["work/ag_morphology/${LANGUAGE_NAME}_${MODEL}_${PACK}_output.txt",
                                      "work/ag_morphology/${LANGUAGE_NAME}_${MODEL}_${PACK}_grammar.txt",
                                      "work/ag_morphology/${LANGUAGE_NAME}_${MODEL}_${PACK}_trace.txt",
                                      "work/ag_morphology/${LANGUAGE_NAME}_${MODEL}_${PACK}_keyword_output.txt"],                                  
                                      [cfg, pycfg_data, dev_keyword_text_file])

                env.NormalizePYCFGOutput("work/segmentations/simple_adaptor_grammar/${LANGUAGE_NAME}_${PACK}_dev.txt", pycfg[0])
                env.NormalizePYCFGOutput("work/segmentations/simple_adaptor_grammar/${LANGUAGE_NAME}_${PACK}_dev_keywords.txt", pycfg[-1])
        if os.path.exists(pjoin(env.subst("${IBM_MODELS}/${BABEL_ID}/${PACK}"))):

            baseline_vocabulary = env.File("${VOCABULARY_FILE}")
            baseline_pronunciations = env.File("${PRONUNCIATIONS_FILE}")

            #baseline_vocabulary = env.File("${IBM_MODELS}/${BABEL_ID}/${PACK}/models/vocab")
            #baseline_pronunciations = env.File("${IBM_MODELS}/${BABEL_ID}/${PACK}/models/dict.test")
            #baseline_language_model = env.TrainLanguageModel("work/language_models/${LANGUAGE_NAME}_${PACK}_baseline.arpabo.gz", [data, env.Value(2)])
            baseline_language_model = env.Glob("${IBM_MODELS}/${BABEL_ID}/${PACK}/models/*.arpabo.gz")[0]
            env.Replace(ACOUSTIC_WEIGHT=properties.get("ACOUSTIC_WEIGHT", .09))
            baseline_asr_output = env.RunASR("work/asr_experiments/${LANGUAGE_NAME}/${PACK}/baseline", baseline_vocabulary, baseline_pronunciations, baseline_language_model)
            baseline_kws_output = env.RunKWS("work/kws_experiments/${LANGUAGE_NAME}/${PACK}/baseline", baseline_asr_output[1:], baseline_vocabulary, baseline_pronunciations, dev_keyword_file)
            continue
            segmented_pronunciations_training, morphs = env.SegmentedPronunciations(["work/pronunciations/${LANGUAGE_NAME}_${PACK}_morfessor_segmented.txt",
                                                                                     "work/pronunciations/${LANGUAGE_NAME}_${PACK}_morfessor_morphs.txt"],
                                                                                    [baseline_pronunciations, segs])
            if properties.get("GRAPHEMIC", False):
                morph_pronunciations = env.GraphemicPronunciations("work/pronunciations/${LANGUAGE_NAME}_${PACK}_morfessor_morph_pronunciations.txt", morphs)
            else:
                g2p_segmented_model = env.TrainG2P("work/pronunciations/${LANGUAGE_NAME}_${PACK}_morfessor_segmented_model_1.txt", segmented_pronunciations_training)        
                morph_pronunciations = env.ApplyG2P("work/pronunciations/${LANGUAGE_NAME}_${PACK}_morfessor_morph_pronunciations.txt", [g2p_segmented_model, morphs])
                
            segmented_vocabulary, segmented_pronunciations = env.PronunciationsToVocabDict(
                ["work/asr_input/${LANGUAGE_NAME}_${PACK}_morfessor_vocabulary.txt", "work/asr_input/${LANGUAGE_NAME}_${PACK}_morfessor_pronunciations.txt"],
                [morph_pronunciations, baseline_pronunciations, env.Value(properties.get("GRAPHEMIC", False))])

            segmented_training_text = env.SegmentTranscripts("work/segmented_training/${LANGUAGE_NAME}_${PACK}.txt", [data, segs])
            segmented_language_model = env.TrainLanguageModel("work/asr_input/${LANGUAGE_NAME}_${PACK}_languagemodel_segmented.arpabo.gz",
                                                              [segmented_training_text, Value(2)])

            morfessor_asr_output = env.RunASR("work/asr_experiments/${LANGUAGE_NAME}/${PACK}/morfessor", segmented_vocabulary, segmented_pronunciations, segmented_language_model)
            morfessor_kws_output = env.RunKWS("work/kws_experiments/${LANGUAGE_NAME}/${PACK}/morfessor", morfessor_asr_output[1:], segmented_vocabulary, segmented_pronunciations, dev_keyword_file)
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
    continue
    terms = env.Glob("${INDUSDB_PATH}/IARPA-babel${BABEL_ID}*-dev.kwlist*.xml")[0]
    segmented_terms = env.ApplyMorfessor(["work/segmented_terms/${LANGUAGE_NAME}.txt"], [morfessor_model, terms])
    
    segmented_pronunciations_training, morphs = env.SegmentedPronunciations(["work/pronunciations/${LANGUAGE_NAME}_${PACK}_segmented.txt",
                                                                             "work/pronunciations/${LANGUAGE_NAME}_${PACK}_morphs.txt"], [pronunciations_file, morfessor])

    g2p_segmented_model = env.TrainG2P("work/pronunciations/${LANGUAGE_NAME}_${PACK}_morfessor_model_1.txt", segmented_pronunciations_training)
    #for i in range(2, 5):
    #    g2p_segmented_model = env.TrainG2P("work/pronunciations/${LANGUAGE_NAME}_segmented_model_%d.txt" % (i), [g2p_segmented_model, segmented_pronunciations_training])

    #g2p_model = env.TrainG2P("work/pronunciations/${LANGUAGE_NAME}_morfessor_model_1.txt", pronunciations_file)
    #for i in range(2, 5):
    #    g2p_model = env.TrainG2P("work/pronunciations/${LANGUAGE_NAME}_model_%d.txt" % (i), [g2p_model, pronunciations_file])

    #ibm_g2p_model = env.File("${LORELEI_SVN}/${BABEL_ID}/LimitedLP/models/g2p.4.model")
    
    morph_pronunciations = env.ApplyG2P("work/pronunciations/${LANGUAGE_NAME}_${PACK}_morfessor_pronunciations.txt", [g2p_segmented_model, morphs])
    
    segmented_vocabulary, segmented_pronunciations = env.PronunciationsToVocabDict(
        ["work/asr_input/${LANGUAGE_NAME}_${PACK}_morfessor_vocabulary.txt", "work/asr_input/${LANGUAGE_NAME}_${PACK}_morfessor_pronunciations.txt"],
        [morph_pronunciations, pronunciations_file])

    segmented_training_text = env.SegmentTranscripts("work/segmented_training/${LANGUAGE_NAME}.txt.gz", [training_text, morfessor])
    #segmented_language_model = env.IBMTrainLanguageModel("work/asr_input/${LANGUAGE_NAME}/languagemodel_segmented.arpabo.gz", [segmented_training_text, Value(2)])

    morfessor_asr_output = env.RunASR("work/asr_experiments/${LANGUAGE_NAME}/morfessor", segmented_vocabulary, segmented_pronunciations, segmented_language_model)
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
env.VocabularyComparison("work/vocabulary_comparison.txt", all_texts)
