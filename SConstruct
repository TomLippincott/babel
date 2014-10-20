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
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

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

for language, properties in env["LANGUAGES"].iteritems():
    env.Replace(LANGUAGE=language)
    babel_id = properties.get("BABEL_ID", None)
    env.Replace(BABEL_ID=babel_id)
    if not os.path.exists(env.subst("${LANGUAGE_PACK_PATH}/${BABEL_ID}.tgz")):
        continue
    


    transcripts = env.ExtractTranscripts("work/transcripts/${LANGUAGE}.xml.gz", "${LANGUAGE_PACK_PATH}/${BABEL_ID}.tgz")
    full_data = env.TranscriptsToData("work/full_data/${LANGUAGE}.xml.gz", [transcripts, Value({})])

    if not properties.get("VLLP", False):
        very_limited_data = env.GenerateDataSubset("work/very_limited_data/${LANGUAGE}.xml.gz", [full_data, Value({"RANDOM" : True, "WORDS" : 30000})])
    else:
        very_limited_data = env.StmToData("work/very_limited_data/${LANGUAGE}.xml.gz", 
                                          ["${BABEL_DATA_PATH}/LPDefs.20141006.tgz", env.Value(".*IARPA-babel%d.*VLLP.training.transcribed.stm" % babel_id)])

    arguments = Value({"MODEL" : "morphology",
                       "LANGUAGE" : language,
                       "HAS_PREFIXES" : properties.get("HAS_PREFIXES", True),
                       "HAS_SUFFIXES" : properties.get("HAS_SUFFIXES", True),
                   })

    has_morphology = os.path.exists(env.subst("data/${LANGUAGE}_morphology.txt"))
    if has_morphology:
        training = env.AddMorphology("work/data/${LANGUAGE}/train_morph.xml.gz", [very_limited_data, "data/${LANGUAGE}_morphology.txt"])

    cfg, data = env.MorphologyCFG(["work/models/${LANGUAGE}_VLLP_model.txt", "work/models/${LANGUAGE}_VLLP_data.txt"], [very_limited_data, arguments])
    parses, grammar, trace_file = env.RunPYCFG([cfg, data, arguments])

    #if has_morphology:                
    #    results = getattr(env, "EvaluateMorphology")(parses, training, "data/${LANGUAGE}_morphology.txt")
