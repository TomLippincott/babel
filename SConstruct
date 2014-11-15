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
    ("BABEL_DATA_PATH", "", None),

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
    )

# initialize logging system
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.ERROR)

# initialize build environment
env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default", "textfile"] + [x.TOOLS_ADD for x in [babel_tools, evaluation_tools, scala_tools, morfessor_tools, emma_tools, 
                                                                         trmorph_tools, sfst_tools, mila_tools, almor_tools, pycfg_tools, torque_tools]],
                  )

# don't print out lines longer than the terminal width
def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print s[:int(env["OUTPUT_WIDTH"]) - 10] + "..." + s[-7:]
    else:
        print s

env['PRINT_CMD_LINE_FUNC'] = print_cmd_line
env.Decider("timestamp-newer")

results = []
for language, properties in env["LANGUAGES"].iteritems():
    env.Replace(LANGUAGE=language)
    babel_id = properties.get("BABEL_ID", None)
    env.Replace(BABEL_ID=babel_id)
    if language == "english":
        full_transcripts = env.PennToTranscripts("work/full_transcripts/${LANGUAGE}.xml.gz", ["${PENN_TREEBANK_PATH}", Value({})])
        full_data = env.TranscriptsToData("work/full_data/${LANGUAGE}.xml.gz", [full_transcripts, Value({})])
        limited_data = env.GenerateDataSubset("work/training_data/${LANGUAGE}.xml.gz", [full_data, Value({"RANDOM" : True, "WORDS" : 100000})])

    elif os.path.exists(env.subst("${LANGUAGE_PACK_PATH}/${BABEL_ID}.tgz")):
        full_transcripts = env.ExtractTranscripts("work/full_transcripts/${LANGUAGE}.xml.gz", ["${LANGUAGE_PACK_PATH}/${BABEL_ID}.tgz", Value({})])
        limited_transcripts = env.ExtractTranscripts("work/training_transcripts/${LANGUAGE}_training.xml.gz", ["${LANGUAGE_PACK_PATH}/${BABEL_ID}.tgz",
                                                                                                                Value({"PATTERN" : r".*sub-train.*transcription.*txt"})])
        limited_data = env.TranscriptsToData("work/training_data/${LANGUAGE}.xml.gz", [limited_transcripts, Value({})])
        
        full_data = env.TranscriptsToData("work/full_data/${LANGUAGE}.xml.gz", [full_transcripts, Value({})])

    if properties.get("VLLP", False):
        very_limited_data = env.StmToData("work/very_limited_data/${LANGUAGE}.xml.gz", 
                                          ["${BABEL_DATA_PATH}/LPDefs.20141006.tgz", env.Value(".*IARPA-babel%d.*VLLP.training.transcribed.stm" % babel_id)])
    else:
        very_limited_data = env.GenerateDataSubset("work/very_limited_data/${LANGUAGE}.xml.gz", [full_data, Value({"RANDOM" : True, "WORDS" : 30000})])
    

    has_morphology = os.path.exists(env.subst("data/${LANGUAGE}_morphology.txt"))
    if has_morphology:
        very_limited_data_morph = env.AddMorphology("work/data/${LANGUAGE}/very_limited_morph.xml.gz", [very_limited_data, "data/${LANGUAGE}_morphology.txt"])
        limited_data_morph = env.AddMorphology("work/data/${LANGUAGE}/limited_morph.xml.gz", [limited_data, "data/${LANGUAGE}_morphology.txt"])

    for has_prefixes in [True, False]:
        for has_suffixes in [True, False]:
            for is_agglutinative in [True, False]:
                for name, train, train_morph in [
                        ("VLLP", very_limited_data, very_limited_data_morph),
                        #("LLP", limited_data, limited_data_morph)
                ]:
                    arguments = Value({"MODEL" : "morphology",
                                       "DATA" : name,
                                       "LANGUAGE" : language,
                                       "HAS_PREFIXES" : has_prefixes,
                                       "HAS_SUFFIXES" : has_suffixes,
                                       "IS_AGGLUTINATIVE" : is_agglutinative,
                                   })
                    

                    cfg, data = env.MorphologyCFG([train, arguments])

                    parses, grammar, trace_file = env.RunPYCFG([cfg, data, arguments])

                    if has_morphology:
                        stem = "%s_%s_%s" % (has_prefixes, has_suffixes, name)
                        py = env.MorphologyOutputToEMMA("work/ag_output/${LANGUAGE}_%s.txt" % stem, [parses, arguments])
                        
                        guess, gold = env.PrepareDatasetsForEMMA([py, very_limited_data_morph, arguments])
                                                                 
                        results.append(env.RunEMMA([guess, gold, arguments]))

    morfessor = env.TrainMorfessor("work/morfessor/${LANGUAGE}.xml.gz", very_limited_data_morph)        
    if has_morphology:
        guess, gold = env.PrepareDatasetsForEMMA([morfessor, very_limited_data_morph, Value({"LANGUAGE" : "$LANGUAGE", "MODEL" : "MORFESSOR", "DATA" : name})])
        results.append(env.RunEMMA([guess, gold, Value({"LANGUAGE" : "$LANGUAGE", "MODEL" : "MORFESSOR", "DATA" : name})]))
        
env.CollateResults("work/results.txt", results)
