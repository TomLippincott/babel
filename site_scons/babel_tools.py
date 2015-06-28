from SCons.Builder import Builder
from SCons.Script import *
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
import cPickle as pickle
import math
try:
    import lxml.etree as et
except:
    import xml.etree.ElementTree as et
import xml.sax
import sys
import gzip
from os.path import join as pjoin
from os import listdir
import tarfile
from random import randint, shuffle
from common_tools import DataSet, meta_open, pairs, temp_file, temp_dir
import time
import codecs
from attila import Audio, Decimator

def bad_word(w):
    return (w.startswith("*") and w.endswith("*")) or (w.startswith("<") and w.endswith(">")) or (w.startswith("(") and w.endswith(")"))

def load_analyses(fname):
    retval = {}
    with meta_open(fname) as ifd:
        for i, l in enumerate(ifd):
            word, analyses = l.split("\t")
            retval[word] = (i, set())
            for a in analyses.split(","):
                i, ss = retval[word]
                ss.add(tuple([x.split(":")[0].strip() for x in a.split() if not x.startswith("~")]))
                retval[word] = (i, ss)
    return retval

def dummy(sources):
    return []

def collate_results(target, source, env):
    data = {}
    for k, v in env["MORPHOLOGY_RESULTS"].iteritems():
        with meta_open(v[0].rstr()) as ifd:
            names, values = [x.strip().split("\t") for x in ifd][0:2]
            data[k] = {n : "%.3f" % float(v) for n, v in zip(names, values)}
    for k, v in env["TAGGING_RESULTS"].iteritems():
        with meta_open(v[0].rstr()) as ifd:
            names, values = [x.strip().split("\t") for x in ifd][0:2]
            if k in data:
                data[k].update({n : "%.3f" % float(v) for n, v in zip(names, values)})
            else:
                data[k] = {n : "%.3f" % float(v) for n, v in zip(names, values)}
    with meta_open(target[0].rstr(), "w") as ofd:
        properties = ["Lang", "Method", "Units"]
        names = sorted(set(sum([x.keys() for x in data.values()], [])))
        ofd.write("\t".join(properties + names) + "\n")
        for k, v in data.iteritems():
            k = {"METHOD" : k[1], "LANG" : k[0], "UNITS" : k[2].split("-")[0]}
            ofd.write("\t".join([k.get(p.upper(), "").title() for p in properties] + [v.get(n, "") for n in names]) + "\n")
    return None

def conllish_to_xml(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        sentences = [[(w, t, []) for w, t in [re.split(r"\s+", x) for x in s.split("\n") if not re.match(r"^\s*$", x)]] for s in re.split(r"\n\n", ifd.read(), flags=re.M)]
    data = DataSet.from_sentences(sentences)
    with meta_open(target[0].rstr(), "w") as ofd:
        data.write(ofd)
    return None

def rtm_to_data(target, source, env):
    sentences = []
    with meta_open(source[0].rstr()) as ifd:
        for sentence in ifd:
            words = [w for w in sentence.split()[5:] if w not in ["(())", "IGNORE_TIME_SEGMENT_IN_SCORING"]]
            if len(words) > 0:
                sentences.append(words)
    dataset = DataSet.from_sentences([[(w, None, []) for w in s] for s in sentences])
    with meta_open(target[0].rstr(), "w") as ofd:
        dataset.write(ofd)
    return None

def extract_transcripts(target, source, env):
    args = source[-1].read()
    data = {}    
    tb = et.TreeBuilder()
    tb.start("xml", {})
    with tarfile.open(source[0].rstr()) as tf:
        for name in [n for n in tf.getnames() if re.match(args.get("PATTERN", r".*transcription.*\.txt"), n)]:
            text = tf.extractfile(name).read()
            try:
                tb.start("file", {"name" : name})
                tb.data(text.decode("utf-8"))
                tb.end("file")        
            except:
                print name, text
                raise
    tb.end("xml")
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(et.tostring(tb.close()))
    return None

def penn_to_transcripts(target, source, env):
    args = source[-1].read()
    data = {}    
    tb = et.TreeBuilder()
    tb.start("xml", {})
    with tarfile.open(source[0].rstr()) as tf:
        for name in [n for n in tf.getnames() if re.match(args.get("PATTERN", r".*parsed/mrg/wsj/.*mrg"), n)]:
            text = tf.extractfile(name).read()
            try:
                tb.start("file", {"name" : name})
                for sentence in text.split("(S "):
                    words = [m.group(1) for m in re.finditer(r"\(\S+ (\S+)\)", sentence)]
                    if len(words) > 0:
                        tb.data(" ".join(words) + "\n")
                tb.end("file")        
            except:
                print name, text
                raise
    tb.end("xml")
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(et.tostring(tb.close()))
    return None

def transcripts_to_data(target, source, env):
    args = source[-1].read()
    sentences = []
    with meta_open(source[0].rstr()) as ifd:
        for f in et.parse(ifd).getiterator("file"):
            name = f.get("name")
            for line in [l for l in f.text.split("\n") if not l.startswith("[")]:
                words = [(w, None, []) for w in line.split()]
                if len(words) > 0:
                    sentences.append(words)
    with meta_open(target[0].rstr(), "w") as ofd:
        DataSet.from_sentences(sentences).write(ofd)
    return None

def stm_to_data(target, source, env):
    if source[0].rstr().endswith("tgz"):
        pattern = source[1].read()
        with tarfile.open(source[0].rstr()) as tf:
            names = [n for n in tf.getnames() if re.match(env.subst(pattern), n)]
            if len(names) == 0:
                return "No file in archive %s matched pattern %s" % (source[0].rstr(), pattern)
            elif len(names) > 1:
                return "More than one file in archive %s matched pattern %s" % (source[0].rstr(), pattern)
            else:
                text = codecs.decode(tf.extractfile(names[0]).read(), "utf-8")
    else:
        return "Must provide an archive and file name pattern"
    sentences = [[w for w in x.split()[3:] if not (w.startswith("(") or w.startswith("<"))] for x in text.split("\n")]
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join([" ".join(s) for s in sentences if len(s) > 0]) + "\n")
    return None

def generate_data_subset(target, source, env):
    args = source[-1].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[0]
    indices = range(len(data.sentences))
    if args.get("RANDOM", False):
        shuffle(indices)
    to_keep = [1]
    total_words = 0
    while total_words < args.get("WORDS", 0):
        total_words += len(data.sentences[indices[0]])
        to_keep.append(indices[0])
        indices = indices[1:]
    with meta_open(target[0].rstr(), "w") as ofd:
        data.get_subset(to_keep).write(ofd)
    return None
#
#
#

asr_defaults = sum(
    [
        [("%s_FILE" % (k), "${IBM_MODELS}/${LANGUAGE_ID}/${PACK}/models/%s" % (v)) for k, v in 
         [("MEL", "mel"),
          ("PHONE", "pnsp"),
          ("PHONE_SET", "phonesset"),
          ("TAGS", "tags"),
          ("PRIORS", "priors"),
          ("TREE", "tree"),
          ("TOPO", "topo.tied"),
          ("TOPO_TREE", "topotree"),
          ("LDA", "30.mat"),
          ("TRFS", "*.trfs"),
          ("TR", "*.tr"),
          ("CTX", "*.ctx"),
          ("GS", "*.gs"),
          ("FS", "*.fs"),
          ("MS", "*.ms"),
          ("PRONUNCIATIONS", "dict.test"),
          ("LANGUAGE_MODEL", "lm*"),
          ("VOCABULARY", "vocab"),
          ("PRONUNCIATIONS", "dict.test"),
          ]],
                  
        [("%s_FILE" % (k), "${IBM_MODELS}/${LANGUAGE_ID}/${PACK}/segment/%s" % (v)) for k, v in
         [("SEGMENTATION", "*.dev.*")
          ]],

        [("%s_FILE" % (k), "${IBM_MODELS}/${LANGUAGE_ID}/${PACK}/adapt/%s" % (v)) for k, v in
         [("WARP", "warp.lst")
          ]],

        [("%s_FILE" % (k), "${INDUSDB_PATH}/*babel${LANGUAGE_ID}*/%s" % (v)) for k, v in
         [("STM", "*.stm"),
          ("RTTM", "*.rttm"),
          ]],
        
        [(k, v) for k, v in
         [("SAMPLING_RATE", 8000),
          ("FEATURE_TYPE", "plp"),
          ("MAX_ERROR", 15000),
          ("USE_DISPATCHER", False),          
          ]],

        [("%s_PATH" % (k), v) for k, v in
         [("PCM", "${LANGUAGE_PACKS}/${LANGUAGE_ID}"),
          ("MODEL", "${IBM_MODELS}/${LANGUAGE_ID}/${PACK}/models"),
          ("CMS", "${IBM_MODELS}/${LANGUAGE_ID}/${PACK}/adapt/cms"),
          ("FMLLR", "${IBM_MODELS}/${LANGUAGE_ID}/${PACK}/adapt/fmllr"),
          ("TXT", "${IBM_MODELS}/${LANGUAGE_ID}/${PACK}/SI/cons"),          
          ]]
        ],    
    []
    )

id_to_language = {
    102 : "assamese",
    106 : "tagalog",
    206 : "zulu",
    }

def run_asr(env, name, *args, **kw):
    #language = id_to_language[kw["LANGUAGE_ID"]]
    files = {}
    directories = {}
    parameters = {}
    searched = {}
    #renv = env.Clone(**kw)
    env.Replace(EXPERIMENT_NAME=name)
    for k, v in asr_defaults:
        searched[k] = searched.get(k, []) + [env.subst(v)]
        if k.endswith("FILE"):
            files[k] = env.Glob(v)
        elif k.endswith("PATH"):
            directories[k] = env.Dir(v)
        else:            
            parameters[k] = v

    for k, v in kw.iteritems():
        searched[k] = searched.get(k, []) + [env.subst(v)]
        if k.endswith("FILE"):
            files[k] = files.get(k, []) + env.Glob(str(v))
        elif k.endswith("PATH"):
            directories[k] = env.Dir(v)
        else:            
            parameters[k] = v

    #for k, v in files.iteritems():
    #    if len(v) == 0:
    #        env.Exit("ERROR: couldn't find suitable file for '%s' when searching %s" % (k, searched[k]))
    env.Replace(OUTPUT_PATH="work/asr/${LANGUAGE_NAME}/${EXPERIMENT_NAME}")
    #directories["OUTPUT_PATH"] = 
    #pjoin("work", "asr", "output", language, parameters["PACK"], name)
    #directories["ASR_OUTPUT_PATH"] = directories["OUTPUT_PATH"]
    #try:
    #    os.makedirs(directories["OUTPUT_PATH"])
    #except:
    #    pass
    
    logging.debug("\n".join(
            ["Files:"] +
            ["%s = %s" % (k, v) for k, v in files.iteritems()] +
            ["Directories:"] +
            ["%s = %s" % (k, v) for k, v in directories.iteritems()] +
            ["Parameters:"] +
            ["%s = %s" % (k, v) for k, v in parameters.iteritems()]
            ))

    # create the configuration files for running the experiment
    experiment = env.CreateASRExperiment(env.Dir("work/asr/configurations/${LANGUAGE_NAME}/${EXPERIMENT_NAME}"),
        #pjoin("work", "asr", "configurations", language, parameters["PACK"], name)), 
                                         [env.Value(x) for x in [files, directories, parameters]])

    if env["RUN_ASR"]:
        # run the experiment
        asr_output = None #env.RunASRExperiment(target=env.Dir(directories["OUTPUT_PATH"]), source=experiment, ACOUSTIC_WEIGHT=parameters["ACOUSTIC_WEIGHT"])

        # evaluate the output
        asr_score = None #env.ScoreResults(env.Dir(pjoin("work", "asr", "scoring", language, parameters["PACK"], name)),
                          #           [env.Dir(os.path.abspath(pjoin(directories["OUTPUT_PATH"], "ctm"))), files["STM_FILE"], asr_output])[0]
    else:
        asr_output = env.Textfile("${OUTPUT_PATH}/fake_output.txt", time.asctime())
        asr_score = env.Textfile("${OUTPUT_PATH}/fake_score.txt", time.asctime())
    return env.Flatten([asr_output, asr_score])

kws_defaults = sum(
    [
        [("%s_FILE" % (k), v) for k, v in
         [("VOCABULARY", "${IBM_MODELS}/${LANGUAGE_ID}/${PACK}/models/vocab"),
          ("IV_DICTIONARY", "${IBM_MODELS}/${LANGUAGE_ID}/${PACK}/kws-resources/kws-resources*/dict.IV*"),
          ("OOV_DICTIONARY", "${IBM_MODELS}/${LANGUAGE_ID}/${PACK}/kws-resources/kws-resources*/dict.OOV*"),
          ("SEGMENTATION", "${IBM_MODELS}/${LANGUAGE_ID}/${PACK}/segment/babel${LANGUAGE_ID}.*dev.*db"),
          ("KEYWORDS", "${INDUSDB_PATH}/*${LANGUAGE_ID}*.kwlist.xml"),
          ("RTTM", "${INDUSDB_PATH}/*${LANGUAGE_ID}*/*${LANGUAGE_ID}*dev.rttm"),
          ("ECF", "${INDUSDB_PATH}/*${LANGUAGE_ID}*.ecf.xml"),
          ]],
        ],
    []
    )


def run_kws(env, name, asr_output, *args, **kw):
    language_id = kw["LANGUAGE_ID"]
    language = id_to_language[language_id]
    files = {}
    directories = {}
    parameters = {}

    directories["OUTPUT_PATH"] = pjoin("work", "kws", language, kw["PACK"], name)
    directories["LATTICE_PATH"] = pjoin(asr_output.get_dir().rstr(), "lat")
    asr_output_path = asr_output.get_dir()
    renv = env.Clone(**kw)

    for k, v in kws_defaults:
        if k.endswith("FILE"):
            files[k] = renv.Glob(v)
        elif k.endswith("PATH"):
            directories[k] = renv.Dir(v)
        else:
            parameters[k] = v
    
    for k, v in kw.iteritems():
        if k.endswith("FILE"):
            files[k] = renv.Glob(v)
        elif k.endswith("PATH"):
            directories[k] = renv.Dir(v)
        else:
            parameters[k] = v

    for k in files.keys():
        files[k] = files[k][0]

    g = re.match(r".*(babel%.3d.*)_conv-dev.ecf.xml" % (language_id), files["ECF_FILE"].rstr()).groups()[0]
    parameters["EXPID"] = "KWS13_IBM_%s_conv-dev_BaDev_KWS_LimitedLP_BaseLR_NTAR_p-test-STO_1" % (g)

    logging.debug("\n\t".join(
            ["Files:"] +
            ["%s = %s" % (k, v) for k, v in files.iteritems()] +
            ["Directories:"] +
            ["%s = %s" % (k, v) for k, v in directories.iteritems()] +
            ["Parameters:"] +
            ["%s = %s" % (k, v) for k, v in parameters.iteritems()]
            ))

    if not env["RUN_KWS"]:
         return None
    else:

        # just make some local variables from the experiment definition (for convenience)
        iv_dict = files["VOCABULARY_FILE"]
        oov_dict = files["OOV_DICTIONARY_FILE"]
        segmentation_file = files["SEGMENTATION_FILE"]
        kw_file = files["KEYWORDS_FILE"]

        iv_query_terms, oov_query_terms, term_map, word_to_word_fst, kw_file = env.QueryFiles([pjoin(directories["OUTPUT_PATH"], x) for x in ["iv_queries.txt", 
                                                                                                                                              "oov_queries.txt",
                                                                                                                                              "term_map.txt",
                                                                                                                                              "word_to_word.fst",
                                                                                                                                              "kwfile.xml"]], 
                                                                                              [kw_file, iv_dict, env.Value(language_id), env.Value("")])
        # JOBS LATTICE_DIRECTORY KW_FILE RTTM_FILE
        base_path = directories["OUTPUT_PATH"]
        lattice_directory = pjoin(asr_output_path.rstr(), "lat")


        full_lattice_list = env.LatticeList(pjoin(directories["OUTPUT_PATH"], "lattice_list.txt"),
                                            [segmentation_file, env.Value(lattice_directory)])
        env.Depends(full_lattice_list, asr_output)

        lattice_lists = env.SplitList([pjoin(directories["OUTPUT_PATH"], "lattice_list_%d.txt" % (n + 1)) for n in range(env["LOCAL_JOBS_PER_SCONS_INSTANCE"])], full_lattice_list)

        wordpron = env.WordPronounceSymTable(pjoin(directories["OUTPUT_PATH"], "in_vocabulary_symbol_table.txt"),
                                             iv_dict)

        isym = env.CleanPronounceSymTable(pjoin(directories["OUTPUT_PATH"], "cleaned_in_vocabulary_symbol_table.txt"),
                                          wordpron)

        mdb = env.MungeDatabase(pjoin(directories["OUTPUT_PATH"], "munged_database.txt"),
                                [segmentation_file, full_lattice_list])

        padfst = env.BuildPadFST(pjoin(directories["OUTPUT_PATH"], "pad_fst.txt"),
                                 wordpron)

        full_data_list = env.CreateDataList(pjoin(directories["OUTPUT_PATH"], "full_data_list.txt"),
                                            [mdb] + [env.Value({"oldext" : "fsm.gz", 
                                                                "ext" : "fst",
                                                                "subdir_style" : "hub4",
                                                                "LATTICE_DIR" : directories["LATTICE_PATH"],
                                                                })], BASE_PATH=directories["OUTPUT_PATH"])        

        ecf_file = env.ECFFile(pjoin(directories["OUTPUT_PATH"], "ecf.xml"), mdb)

        data_lists = env.SplitList([pjoin(directories["OUTPUT_PATH"], "data_list_%d.txt" % (n + 1)) for n in range(env["LOCAL_JOBS_PER_SCONS_INSTANCE"])], full_data_list)

        p2p_fst = env.FSTCompile(pjoin(directories["OUTPUT_PATH"], "p2p_fst.txt"),
                                 [isym, word_to_word_fst])

        

        #word_to_phone_lattices = env.WordToPhoneLattice(target=env.Dir(pjoin(directories["OUTPUT_PATH"], "lattices")), 
        #                                               source=[full_lattice_list, wordpron, iv_dict, env.Value({"PRUNE_THRESHOLD" : -1, 
        #                                                                                                        "FSMGZ_FORMAT" : "",
        #                                                                                                        "CONFUSION_NETWORK" : "",
        #                                                                                                        "EPSILON_SYMBOLS" : "'<s>,</s>,~SIL,<HES>'"})])
        #
        #
        #return None
        wtp_lattices = []
        for i, (data_list, lattice_list) in enumerate(zip(data_lists, lattice_lists)):
            wp = env.WordToPhoneLattice(pjoin(directories["OUTPUT_PATH"], "lattices", "lattice_generation-%d.stamp" % (i + 1)), 
                                        [data_list, lattice_list, wordpron, iv_dict, env.Value({"PRUNE_THRESHOLD" : -1,
                                                                                                "EPSILON_SYMBOLS" : "'<s>,</s>,~SIL,<HES>'",
                                                                                                })])

            fl = env.GetFileList(pjoin(directories["OUTPUT_PATH"], "file_list-%d.txt" % (i + 1)), 
                                 [data_list, wp])
            idx = env.BuildIndex(pjoin(directories["OUTPUT_PATH"], "index-%d.fst" % (i + 1)),
                                 fl)

            wtp_lattices.append((wp, data_list, lattice_list, fl, idx))

        merged = {}
        for query_type, query_file in zip(["in_vocabulary", "out_of_vocabulary"], [iv_query_terms, oov_query_terms]):
            queries = env.QueryToPhoneFST(pjoin(directories["OUTPUT_PATH"], query_type, "query.fst"), 
                                          [p2p_fst, isym, iv_dict, query_file, env.Value({"n" : 1, "I" : 1, "OUTDIR" : pjoin(directories["OUTPUT_PATH"], query_type, "queries")})])
            searches = []
            for i, (wtp_lattice, data_list, lattice_list, fl, idx) in enumerate(wtp_lattices):
                searches.append(env.StandardSearch(pjoin(directories["OUTPUT_PATH"], query_type, "search_output-%d.txt" % (i + 1)),
                                                   [data_list, isym, idx, padfst, queries, env.Value({"PRECISION" : "'%.4d'", "TITLE" : "std.xml", "LANGUAGE_ID" : language_id})]))



            qtl, res_list, res, ures = env.Merge([pjoin(directories["OUTPUT_PATH"], query_type, x) for x in ["ids_to_query_terms.txt", "result_file_list.txt", "search_results.xml", "unique_search_results.xml"]], 
                                                 [query_file] + searches + [env.Value({"MODE" : "merge-default",
                                                                                       "PADLENGTH" : 4,                                    
                                                                                       "LANGUAGE_ID" : language_id})])

            merged[query_type] = ures
            om = env.MergeScores(pjoin(directories["OUTPUT_PATH"], query_type, "results.xml"), 
                                 res)

        iv_oov = env.MergeIVOOV(pjoin(directories["OUTPUT_PATH"], "iv_oov_results.xml"), 
                                [merged["in_vocabulary"], merged["out_of_vocabulary"], term_map, files["KEYWORDS_FILE"]])

        norm = env.Normalize(pjoin(directories["OUTPUT_PATH"], "norm.kwslist.xml"), 
                             [iv_oov, kw_file])

        normSTO = env.NormalizeSTO(pjoin(directories["OUTPUT_PATH"], "normSTO.kwslist.xml"), 
                                   norm)

        kws_score = env.Score(pjoin(directories["OUTPUT_PATH"], "scoring", "Full-Occur-MITLLFA3-AppenWordSeg.sum.txt"), 
                              [normSTO, kw_file, env.Value({"RTTM_FILE" : str(files["RTTM_FILE"]), "ECF_FILE" : ecf_file[0].rstr(), "EXPID" : parameters["EXPID"]})])

        return kws_score


def build_extrinsic_tables(target, source, env):
    files = source[0].read()
    rows = []
    for (language, pack), setups in files.iteritems():
        for setup, (asr_fname, kws_fname) in setups.iteritems():
            with meta_open(asr_fname) as asr_fd, meta_open(kws_fname) as kws_fd:
                asr = ASRResults(asr_fd)
                kws = KWSResults(kws_fd)
                rows.append([language, setup] + [asr.get(x) for x in ["error", "substitutions", "deletions", "insertions"]] + [kws.get(x) for x in ["pmiss", "mtwv"]])
    with meta_open(target[0].rstr(), "w") as ofd:
        body = "\n".join([r" & ".join([str(x) for x in row]) + r" \\" for row in rows])
        ofd.write(r"""
\begin{tabular}{|*{2}{l|}*{6}{r|}}
  \hline
  Language & Augmentation & \multicolumn{4}{|c|}{ASR} & \multicolumn{2}{|c|}{KWS} \\
  & & Errors & Subs & Dels & Ins & PMiss & MTWV \\
  \hline
%s
  \hline
\end{tabular}
""" % (body))
    return None

def build_extrinsic_tables_emitter(target, source, env):
    files = source[0].read()
    new_sources = [files_to_strings(files)] + leaves(files)
    return target, new_sources

def build_property_tables(target, source, env):
    properties = source[0].read() #[x.read() for x in source[0:2]]
    languages = set([x[0] for x in properties.keys()])
    lookup = {"PRE" : "Prefixes",
              "STM" : "Stems",
              "SUF" : "Suffixes",
              }
    packs = ["Limited"]
    language_table, morfessor_table, babelgum_table = {}, {}, {}


    for language in languages:
        language_properties = properties[(language, "Limited")]
        with meta_open(language_properties["prefixes"]) as prefix_fd, meta_open(language_properties["stems"]) as stem_fd, meta_open(language_properties["suffixes"]) as suffix_fd:
            pre, stm, suf = [
                [l.strip().split()[0] for l in prefix_fd if "<epsilon>" not in l],
                [l.strip().split()[0] for l in stem_fd if "<epsilon>" not in l],
                [l.strip().split()[0] for l in suffix_fd if "<epsilon>" not in l],
                ]
            babelgum_table[language] = [len(pre), "%.2f" % (sum(map(len, pre)) / max(1.0, float(len(pre)))), 
                                         len(stm), "%.2f" % (sum(map(len, stm)) / float(len(stm))),
                                         len(suf), "%.2f" % (sum(map(len, suf)) / float(len(suf))),
                                         ]            

        with meta_open(language_properties["limited_vocabulary"]) as lim_fd, meta_open(language_properties["dev_vocabulary"]) as dev_fd:
            lim_vocab = set(FrequencyList(lim_fd).make_conservative().keys())
            dev_vocab = set(FrequencyList(dev_fd).make_conservative().keys())
            lim_vocab_size = len(lim_vocab)
            dev_vocab_size = len(dev_vocab)
            both_vocabs = len(lim_vocab.union(dev_vocab))
            avg_len_lim_vocab = sum(map(len, lim_vocab)) / float(len(lim_vocab))
            avg_len_dev_vocab = sum(map(len, dev_vocab)) / float(len(dev_vocab))
            language_table[language] = [lim_vocab_size, "%.2f" % (avg_len_lim_vocab), 
                                        dev_vocab_size, "%.2f" % (avg_len_dev_vocab), 
                                        len([x for x in dev_vocab if x not in lim_vocab])]
        with meta_open(language_properties["morfessor_input"]) as input_fd, meta_open(language_properties["morfessor_output"]) as output_fd:
            input_vocab = FrequencyList({w : int(c) for c, w in [x.strip().split() for x in input_fd]})
            morf_output = MorfessorOutput(output_fd)
            pre = morf_output.morphs["PRE"]
            stm = morf_output.morphs["STM"]
            suf = morf_output.morphs["SUF"]
            morfessor_table[language] = [len(pre), "%.2f" % (sum(map(len, pre)) / max(1.0, float(len(pre)))), 
                                         len(stm), "%.2f" % (sum(map(len, stm)) / float(len(stm))),
                                         len(suf), "%.2f" % (sum(map(len, suf)) / float(len(suf))),
                                         ]
    # language, morfessor, babelgum
    with meta_open(target[0].rstr(), "w") as ofd:
        body = "\n".join([r"  %s & %s \\" % (l.title(), " & ".join(map(str, v))) for l, v in sorted(language_table.iteritems())])
        ofd.write(r"""
%%language properties
\begin{tabular}{|l|r|r|r|r|r|}
  \hline
  Language & \multicolumn{2}{|c|}{Training} & \multicolumn{2}{|c|}{Development} & OOV \\
  & Count & Avg. length & Count & Avg. length & \\
  \hline
%s
  \hline
\end{tabular}
""" % body)

    with meta_open(target[1].rstr(), "w") as ofd:
        body = "\n".join([r"  %s & %s \\" % (l.title(), " & ".join(map(str, v))) for l, v in sorted(morfessor_table.iteritems())])
        ofd.write(r"""
%%morfessor properties
\begin{tabular}{|l|r|r|r|r|r|r|}
  \hline
  Language & \multicolumn{2}{|c|}{Prefixes} & \multicolumn{2}{|c|}{Stems} & \multicolumn{2}{|c|}{Suffixes} \\
  & Count & Avg. length & Count & Avg. length & Count & Avg. length \\
  \hline
%s
  \hline
\end{tabular}
""" % body)

    with meta_open(target[2].rstr(), "w") as ofd:
        body = "\n".join([r"  %s & %s \\" % (l.title(), " & ".join(map(str, v))) for l, v in sorted(babelgum_table.iteritems())])
        ofd.write(r"""
%%babelgum properties
\begin{tabular}{|l|r|r|r|r|r|r|}
  \hline
  Language & \multicolumn{2}{|c|}{Prefixes} & \multicolumn{2}{|c|}{Stems} & \multicolumn{2}{|c|}{Suffixes} \\
  & Count & Avg. length & Count & Avg. length & Count & Avg. length \\
  \hline
%s
  \hline
\end{tabular}
""" % body)

    return None

def build_property_tables_emitter(target, source, env):
    properties = source[0].read()
    new_sources = [env.Value(files_to_strings(properties)),
                   ] + sum(map(leaves, [properties]), [])
    return target, new_sources


def build_site(target, source, env):
    properties, figures, results = [x.read() for x in source[0:3]]
    languages = set([x[0] for x in figures.keys()])
    lookup = {"PRE" : "Prefixes",
              "STM" : "Stems",
              "SUF" : "Suffixes",
              }
    packs = ["Limited"]
    base_path = os.path.dirname(target[0].rstr())
    try:
        os.makedirs(pjoin(base_path, "images"))
    except:
        pass
    with meta_open(pjoin(base_path, "theme.css"), "w") as ofd:
        ofd.write("body {text-align : center; vertical-align : top;}\n")
        ofd.write("table {text-align : center; vertical-align : top;}\n")
        ofd.write("tr {text-align : center; vertical-align : top;}\n")
        ofd.write("td {text-align : center; vertical-align : top;}\n")
    with meta_open(target[0].rstr(), "w") as ofd:
        xml = et.TreeBuilder()
        xml.start("html", {})
        xml.start("head", {}), xml.start("link", {"rel" : "stylesheet", "type" : "text/css", "href" : "theme.css"}), xml.end("link"), xml.end("head")
        xml.start("body", {}), xml.start("table", {})
        for language in languages:
            language_properties = properties[(language, "Limited")]
            with meta_open(language_properties["prefixes"]) as prefix_fd, meta_open(language_properties["stems"]) as stem_fd, meta_open(language_properties["suffixes"]) as suffix_fd:
                babel_output = {"PRE" : [l.strip().split()[0] for l in prefix_fd if "<epsilon>" not in l],
                                "STM" : [l.strip().split()[0] for l in stem_fd if "<epsilon>" not in l],
                                "SUF" : [l.strip().split()[0] for l in suffix_fd if "<epsilon>" not in l],
                                }
            with meta_open(language_properties["limited_vocabulary"]) as lim_fd, meta_open(language_properties["dev_vocabulary"]) as dev_fd:
                lim_vocab = set(FrequencyList(lim_fd).make_conservative().keys())
                dev_vocab = set(FrequencyList(dev_fd).make_conservative().keys())
                lim_vocab_size = len(lim_vocab)
                dev_vocab_size = len(dev_vocab)
                both_vocabs = len(lim_vocab.union(dev_vocab))
                avg_len_lim_vocab = sum(map(len, lim_vocab)) / float(len(lim_vocab))
                avg_len_dev_vocab = sum(map(len, dev_vocab)) / float(len(dev_vocab))
            with meta_open(language_properties["morfessor_input"]) as input_fd, meta_open(language_properties["morfessor_output"]) as output_fd:
                input_vocab = FrequencyList({w : int(c) for c, w in [x.strip().split() for x in input_fd]})
                morf_output = MorfessorOutput(output_fd)
                morf_analysis_counts = len(morf_output)
                morf_morph_counts = {k : len(v) for k, v in morf_output.morphs.iteritems()}
                morf_morph_lengths = {k : sum(map(len, v)) / float(len(v)) for k, v in morf_output.morphs.iteritems() if len(v) > 0}
                

            xml.start("tr", {}), xml.start("td", {}), xml.start("h1", {}), xml.data(language.title()), xml.end("h1"), xml.end("td"), xml.end("tr")

            xml.start("table", {})
            xml.start("tr", {}), [(xml.start("td", {}), xml.start("h3", {}), xml.data("%s information" % (x)), xml.end("h3"), xml.end("td")) for x in ["Language", "Morfessor", "BabelGUM"]], xml.end("tr")
            
            xml.start("tr", {})

            # Language information
            xml.start("td", {})
            xml.start("table", {"border" : "1"})
            #xml.start("tr", {}), [(xml.start("td", {}), xml.data(x), xml.end("td")) for x in ["Pack", "Vocabulary size"]], xml.end("tr")
            xml.start("tr", {}), xml.start("td", {}), xml.end("td"), xml.start("td", {}), xml.data("Count"), xml.end("td"), xml.start("td", {}), xml.data("Average length"), xml.end("td"), xml.end("tr")
            xml.start("tr", {}), xml.start("td", {}), xml.data("Limited vocab"), xml.end("td"), xml.start("td", {}), xml.data("%d" % (lim_vocab_size)), xml.end("td"), xml.start("td", {}), xml.data("%.2f" % (avg_len_lim_vocab)), xml.end("td"), xml.end("tr")
            xml.start("tr", {}), xml.start("td", {}), xml.data("Dev vocab"), xml.end("td"), xml.start("td", {}), xml.data("%d" % (dev_vocab_size)), xml.end("td"), xml.start("td", {}), xml.data("%.2f" % (avg_len_dev_vocab)), xml.end("td"), xml.end("tr")
            
            #for name, values in [(lookup[x[0]], x[1]) for x in sorted(morf_output.morphs.iteritems())]:
            #    xml.start("tr", {})
                #xml.start("td", {}), xml.data(name), xml.end("td")
                #xml.start("td", {}), xml.data(str(len(values))), xml.end("td")
                #xml.start("td", {}), xml.data(avg_len), xml.end("td")
            #    xml.end("tr")
            xml.end("table")
            xml.end("td")
            
            # Morfessor information
            xml.start("td", {})
            xml.start("table", {"border" : "1"})
            xml.start("tr", {}), [(xml.start("td", {}), xml.data(x), xml.end("td")) for x in ["Type", "Count", "Average length"]], xml.end("tr")
            for name, values in [(lookup[x[0]], x[1]) for x in sorted(morf_output.morphs.iteritems())]:
                if len(values) > 0:
                    avg_len = "%.2f" % (sum(map(len, values)) / float(len(values)))
                else:
                    avg_len = ""
                xml.start("tr", {})
                xml.start("td", {}), xml.data(name), xml.end("td")
                xml.start("td", {}), xml.data(str(len(values))), xml.end("td")
                xml.start("td", {}), xml.data(avg_len), xml.end("td")
                xml.end("tr")
            xml.end("table")
            xml.end("td")
            
            # BabelGUM information
            xml.start("td", {})
            xml.start("table", {"border" : "1"})
            xml.start("tr", {}), [(xml.start("td", {}), xml.data(x), xml.end("td")) for x in ["Type", "Count", "Average length"]], xml.end("tr")
            for name, values in [(lookup[x[0]], x[1]) for x in sorted(babel_output.iteritems())]:
                if len(values) > 0:
                    avg_len = "%.2f" % (sum(map(len, values)) / float(len(values)))
                else:
                    avg_len = ""
                xml.start("tr", {})
                xml.start("td", {}), xml.data(name), xml.end("td")
                xml.start("td", {}), xml.data(str(len(values))), xml.end("td")
                xml.start("td", {}), xml.data(avg_len), xml.end("td")
                xml.end("tr")
            xml.end("table")
            xml.end("td")
            xml.end("tr")
            
            xml.end("table")
            
            # graphs of IV increase and OOV reduction, type-based and token-based
            xml.start("tr", {}), xml.start("td", {}), xml.start("h3", {}), xml.data("Intrinsic performance evaluation"), xml.end("h3"), xml.end("td"), xml.end("tr")
            xml.start("tr", {}), xml.start("td", {})
            xml.start("table", {})
            for pack in packs:
                image_file = "%s_%s.png" % (language, pack)
                shutil.copy(figures[(language, pack)], pjoin(base_path, "images", image_file))
                xml.start("tr", {}), xml.start("td", {}), xml.start("img", {"src" : pjoin("images", image_file)}), xml.end("img"), xml.end("td"), xml.end("tr")
            xml.end("table")
            xml.end("td"), xml.end("tr")
            
            # word error rate for ASR and maximum term-weighted value for KWS
            xml.start("tr", {}), xml.start("td", {}), xml.start("h3", {}), xml.data("Extrinsic performance evaluation"), xml.end("h3"), xml.end("td"), xml.end("tr")
            xml.start("tr", {}), xml.start("td", {}), xml.start("table", {"border" : "1"})
            xml.start("tr", {}), [(xml.start("td", {}), xml.data(x), xml.end("td")) for x in ["Augmentation", "Error", "Substitutions", "Deletions", "Insertions", "PMiss", "MTWV"]], xml.end("tr")
            for name, values in sorted(results[(language, "Limited")].iteritems()):
                with meta_open(values["ASR"]) as asr_fd, meta_open(values["KWS"]) as kws_fd:
                    asr = ASRResults(asr_fd)
                    kws = KWSResults(kws_fd)
                    xml.start("tr", {})
                    xml.start("td", {}), xml.data(name), xml.end("td")
                    [(xml.start("td", {}), xml.data("%.3f" % (asr.get(x))), xml.end("td")) for x in ["error", "substitutions", "deletions", "insertions"]]
                    [(xml.start("td", {}), xml.data("%.3f" % (kws.get(x))), xml.end("td")) for x in ["pmiss", "mtwv"]]
                    xml.end("tr")
            xml.end("table")
            xml.end("td"), xml.end("tr")
            
        xml.end("table"), xml.end("body")
        xml.end("html")
        ofd.write(et.tostring(xml.close()))
    return None

def files_to_strings(data):
    if isinstance(data, dict):
        return {k : files_to_strings(v) for k, v in data.iteritems()}
    elif isinstance(data, NodeList):
        return data[0].rstr()
    elif isinstance(data, Node):
        return data.rstr()
    else:        
        raise Exception(type(data))

def strings_to_files(data):
    if isinstance(data, dict):
        return {k : files_to_strings(v) for k, v in data.iteritems()}
    elif not isinstance(data, basestring):
        return File(data)
    
def leaves(data):
    if isinstance(data, dict):
        return sum(map(leaves, data.values()), [])
    else:
        return [data]

def build_site_emitter(target, source, env):
    properties, figures, results = [x.read() for x in source[0:3]]
    new_targets = pjoin(env["BASE_PATH"], "index.html")
    new_sources = [env.Value(files_to_strings(properties)),
                   env.Value(files_to_strings(figures)),
                   env.Value(files_to_strings(results)),
                   ] + sum(map(leaves, [properties, figures, results]), [])    #[v[0].rstr() for v in sum([x.values() for x in properties.values()], []) + figures.values() + results.values()]
    return new_targets, new_sources

def run_new_kws(env, name, asr_output, *args, **kw):
    pass

def segment_transcripts(target, source, env):
    mapping = {}
    with meta_open(source[1].rstr()) as morph_fd:
        for l in morph_fd:
            toks = l.strip().split()
            mapping["".join([x.strip("+") for x in toks])] = toks
    sentences = []
    with meta_open(source[0].rstr()) as text_fd:
        for l in text_fd:
            sentences.append(sum([mapping.get(w, [w]) for w in l.split()], []))
        # morph = {}
        # for a in DataSet.from_stream(morph_fd)[0].indexToAnalysis.values():
        #     w = "".join(a)
        #     if len(a) == 1:
        #         ms = a
        #     else:
        #         ms = ["%s+" % (a[0])] + ["+%s+" % (x) for x in a[1:-1]] + ["+%s" % (a[-1])]
        #     morph[w] = morph.get(w, []) + [ms]
        # sentences = [[" ".join(morph.get(w, [[w]])[0]) for w in l.split()] for l in text_fd]        
    with meta_open(target[0].rstr(), "w", enc="utf-8") as ofd:
        ofd.write("\n".join([" ".join(s) for s in sentences]))
    return None

def vocabulary_comparison(target, source, env):
    data = {}
    for fname in [x.rstr() for x in source]:
        toks = os.path.basename(fname).split(".")[0].split("_")
        language = "_".join(toks[:-1])
        pack = toks[-1]
        data[language] = data.get(language, {})
        data[language][pack] = {}
        with meta_open(fname) as ifd:
            for line in ifd:
                for word in line.lower().split():
                    data[language][pack][word] = data[language][pack].get(word, 0) + 1
    with meta_open(target[0].rstr(), "w") as ofd:
        for language, packs in data.iteritems():
            vllp = set(packs.get("VLLP", packs.get("LLP", {})).keys())
            flp = set(packs["FLP"].keys())
            just_vllp = len([x for x in vllp if x not in flp])
            just_flp = len([x for x in flp if x not in vllp])
            both = len(flp) - just_flp
            #ofd.write("%s\t%d\t%d\t%d\t%d\t%d\n" % (language, len(vllp), len(flp), just_vllp, just_flp, both))
            ofd.write("%s\t%f\t%f\n" % (language, float(just_flp) / len(flp), sum([packs["FLP"][x] for x in flp if x not in vllp]) / float(sum(packs["FLP"].values()))))
    return None

def keywords_to_list(target, source, env):
    words = set()
    with meta_open(source[0].rstr(), enc=None) as ifd:
        for t in et.parse(ifd).getiterator("kw"):
            text = list(t.getiterator("kwtext"))[0].text
            for word in text.strip().split():
                if env.get("LOWER_CASE", False):
                    words.add(word.lower())
                else:
                    words.add(word)                    
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(sorted(words)) + "\n")
    return None

def prepare_segmentations_for_release(target, source, env):
    nag = env.get("NON_ACOUSTIC_GRAPHEMES")
    rx_str = "^(%s)+$" % ("|".join([unichr(int(x, base=16)) for x in env.get("NON_ACOUSTIC_GRAPHEMES")]))
    rx = re.compile(rx_str)
    for (seg_file, word_file), out in zip(pairs(source, 2), target):
        with meta_open(seg_file.rstr()) as ifd:
            data = [line.strip().split() for line in ifd]
            morphs = {"".join([x.strip("+") for x in ms]) : ms for ms in data}
        with meta_open(word_file.rstr()) as ifd:
            lines = [l.strip().split() for l in ifd if "_" not in l]
            
        for words in lines:
            for word in sum([x.split("-") for x in words], []):
                if word != "" and word not in morphs and "_" not in word and "<" not in word and not re.match(r"^\d+$", word):
                    return "%s, %s, %s" % (seg_file, word_file, word)
                    
        with meta_open(out.rstr(), "w") as ofd:
            for morph, seg in sorted(morphs.iteritems()):
                ofd.write("%s\t%s\n" % (morph, " ".join(seg)))
            
            # with meta_open(word_file.rstr()) as ifd:
            #     for line in ifd:
            #         line = line.strip()
            #         if line.startswith("-") or line.endswith("-") or "<" in line or "_" in line:
            #             pass
            #         #    ofd.write("%s\t%s\n" % (line, line))
            #         else:
            #             words = sum([x.split("-") for x in line.split()], [])                    
            #             ofd.write("%s\t%s\n" % (line, " ".join(sum([morphs.get(w, w) for w in words], []))))
            #         pass
                #ofd.write("\n".join(["%s\t%s" % (w, " ".join(ms)) for w, ms in morphs.iteritems()]) + "\n")

                # ms = []

                #     for morph in morphs.get(word, [word]):
                #         if rx.match(morph):
                #             return "Error: %s, %s" % (morph, word)
                #         ms.append(morph)
                # if len(ms) > 1:
                #     ms = ["%s+" % (ms[0])] + ["+%s+" % (x) for x in ms[1:-1]] + ["+%s" % (ms[-1])]
                # ofd.write("%s\t%s\n" % (" ".join(words), " ".join(ms)))
    return None

def morfessor_babel_experiment(env, target_base, non_acoustic, training_name, word_lists=[]):
    word_lists = Flatten(word_lists)
    if env.get("RUN_SEGMENTATION", True):
        env.Replace(TRAINING_NAME=word_lists[0])
        training_segmentations, model = env.TrainMorfessor(["work/morfessor/${SOURCE.filebase}_segmented.txt",
                                                            "work/morfessor/${SOURCE.filebase}.model"], word_lists[0], NON_ACOUSTIC_GRAPHEMES=non_acoustic)

        env.PrepareSegmentationsForRelease("work/segmentations/${SOURCES[1].filebase}+${SOURCES[1].filebase}+morfessor.txt",
                                           [training_segmentations, word_lists[0]],
                                           NON_ACOUSTIC_GRAPHEMES=non_acoustic)
        for word_list in word_lists[1:]:
            segmentations = env.ApplyMorfessor("work/morfessor/%s+${SOURCES[1].filebase}+morfessor.txt" % (os.path.splitext(os.path.basename(word_lists[0].rstr()))[0]),
                                               [model, word_list])
            env.PrepareSegmentationsForRelease("work/segmentations/%s+${SOURCES[1].filebase}+morfessor.txt" % (os.path.splitext(os.path.basename(word_lists[0].rstr()))[0]),
                                               [segmentations, word_list],
                                               NON_ACOUSTIC_GRAPHEMES=non_acoustic)
    return None

def adaptor_grammar_babel_experiment(env, target_path, model_name, non_acoustic, word_lists=[]):
    segmented = []
    env.Replace(MODEL_NAME=model_name)
    env.Replace(TARGET_PATH=target_path)
    #target_path = env.subst(target_path)
    #target_base = os.path.splitext(target)[0]
    #target_path = os.path.dirname(target)
    #print target, target_base, target_path

    #env.Replace(TARGET_BASE=target_base)

    training_words = word_lists[0]
    characters = env.CharacterProductions("${TARGET_PATH}/characters.txt", word_lists,
                                          NON_ACOUSTIC_GRAPHEMES=non_acoustic)
    pycfg_data = env.MorphologyData("${TARGET_PATH}/data.txt", training_words, NON_ACOUSTIC_GRAPHEMES=non_acoustic)
    cfg = env.ComposeGrammars("${TARGET_PATH}/cfg.txt", ["data/grammar_templates/simple_${MODEL_NAME}.txt", characters])
    segmentations, grammar, trace = env.RunPYCFG(["${TARGET_PATH}/output.txt",
                                                  "${TARGET_PATH}/grammar.txt",
                                                  "${TARGET_PATH}/trace.txt",
                                              ],
                                                 [cfg, pycfg_data])
    training_segmentations = env.NormalizePYCFGOutput("${TARGET_PATH}/output_normalized.txt", segmentations)
    segmented = [training_segmentations]
    #for keyword_list in Flatten(word_lists[1:]):
    #    kw_data = env.MorphologyData("${SOURCE.base}_${TRAINING_NAME}_${MODEL_NAME}.txt", keyword_list, NON_ACOUSTIC_GRAPHEMES=non_acoustic)
    #    segmented.append(env.ApplyAdaptorGrammar("${SOURCES[1].base}_${TRAINING_NAME}_${MODEL_NAME}_segmented.txt", [grammar, kw_data[0]]))
            
    return segmented

def clean_words(target, source, env):
    words = {}
    with meta_open(source[0].rstr()) as ifd:
        for line in [x.strip() for x in ifd]:
            toks = line.split()
            if len(toks) == 2:
                count = int(toks[1])
            else:
                count = None
            line = toks[0]
            if not line.startswith("-") or line.endswith("-") and not any([c in line for c in ["<"]]):                
                if "_" in line:
                    twords = [x for x in line.split("_") if len(x) > 1]
                else:
                    twords = [line]
                for word in sum([w.split("-") for w in twords], []):
                    if env.get("LOWER_CASE", False):
                        word = word.lower()
                    if count == None:
                        words[word] = 1
                    else:
                        words[word] = words.get(word, 0) + count
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(["%s %d" % (w, c) for w, c in words.iteritems()]) + "\n")
    return None

def collect_text(target, source, env):
    pattern = re.compile(source[1].read())
    #discard = re.compile(r"^(\[.*\]|\<.*\>|\(.*\)|\*.*\*)\s*$")
    discard = re.compile(r"^(\[.*\]|\<.*\>|\(.*\))\s*$")
    keep = re.compile(r".*\w+.*", re.UNICODE)
    with tarfile.open(source[0].rstr()) as ifd, meta_open(target[0].rstr(), "w") as ofd:
        for name in ifd.getnames():
            if pattern.match(name):
                if name.endswith("gz"):
                    stream = gzip.GzipFile(fileobj=ifd.extractfile(name))
                else:
                    stream = ifd.extractfile(name)
                for line in codecs.decode(stream.read(), "utf-8").split("\n"):
                    if not discard.match(line):
                        words = [word.strip("*") for word in line.strip().split() if not discard.match(word) and keep.match(word)]
                        words = set([word for word in words if word != ""])
                    #words = [word for word in line.strip().split()]
                        if len(words) > 0:
                            ofd.write("%s\n" % (" ".join(words)))
    return None
    # words = set()
    # with meta_open(target[0].rstr(), "w") as ofd:
    #     for dname in source:
    #         for fname in glob(os.path.join(dname.rstr(), "*.txt")) + glob(os.path.join(dname.rstr(), "*.txt.gz")):
    #             with meta_open(fname) as ifd:
    #                 for line in ifd:
    #                     if not line.startswith("["):
    #                         toks = []
    #                         for x in line.lower().split():
    #                             if x == "<hes>":
    #                                 toks.append("<HES>")
    #                             elif not x.startswith("<"):
    #                                 toks.append(x)
    #                         for t in toks:
    #                             words.add(t)
    #                         if len(toks) > 0:
    #                             ofd.write("%s </s>\n" % (" ".join(toks)))
    # with meta_open(target[1].rstr(), "w") as ofd:
    #     ofd.write("# BOS: <s>\n# EOS: </s>\n# UNK: <UNK>\n<s>\n</s>\n<UNK>\n")
    #     ofd.write("\n".join(sorted(words)) + "\n")                                      
    # return None
def collect_text_emitter(target, source, env):
    if not tarfile.is_tarfile(source[0].rstr()):
        env.Exit()
    else:
        return target, source

def resample(target, source, env):


    typeL = [ 'conversational', 'scripted' ]
    setL  = [ 'dev', 'training', 'untranscribed-training' ]

    with meta_open(target[0].rstr(), "w") as ofd:
        with meta_open(source[0].rstr(), "r") as ifd:
            for member in ifd.getmembers():
                print member.name
                if member.name.endswith("wav"):
                    with temp_file() as src, temp_file() as dst:
                        with meta_open(src, "w", enc=None) as fd:
                            fd.write(ifd.extractfile(member).read())
                        
                        #a48 = Audio()
                        #a8  = Audio()
                        #dec = Decimator()
                        #dec.factor = 6
                        #dec.readFilter(source[1].rstr())
                        #a48.readWAV(src)
                        #dec.compute(a48)
                        #a8.copy(dec)
                        #a8.writeWAV(dst, 8000)
                        with meta_open(dst, "w", enc=None) as fd:
                            fd.write(meta_open(src, enc=None).read())
                        ofd.add(dst, arcname=member.name)
                elif member.isfile():
                    ofd.addfile(member, ifd.extractfile(member))    
    return None

def _resample(target, source, env):

    typeL = [ 'conversational', 'scripted' ]
    setL  = [ 'dev', 'training', 'untranscribed-training' ]
    #with meta_open(target[0].rstr(), "w") as ofd:
    with temp_dir() as tdir:
        lookup = {}
        with meta_open(source[0].rstr(), "r") as ifd:
            for member in ifd.getmembers():
                new_name = os.path.join(tdir, member.name)
                try:
                    os.makedirs(os.path.dirname(new_name))
                except:
                    pass
                lookup[new_name] = member.name
                #print member.name
                if member.name.endswith("wav"):
                    old_name = os.path.join(tdir, "old", member.name)
                    try:
                        os.makedirs(os.path.dirname(old_name))
                    except:
                        pass

                    with meta_open(old_name, "w", enc=None) as ofd:
                        ofd.write(ifd.extractfile(member).read())
                    #a48 = Audio()
                    #a8  = Audio()
                    #dec = Decimator()
                    #dec.factor = 6
                    #dec.readFilter(source[1].rstr())
                    #a48.readWAV(old_name)
                    #dec.compute(a48)
                    #a8.copy(dec)
                    #a8.writeWAV(new_name, 8000)
                    
                elif member.isfile():
                    with meta_open(new_name, "w", enc=None) as ofd:
                        u = codecs.getreader("utf-8")
                        ofd.write(ifd.extractfile(member).read())
            with meta_open(target[0].rstr(), "w") as ofd:
                for f, a in lookup.iteritems():
                    ofd.add(f, arcname=a)
    return None

def word_list(target, source, env):
    words = {}
    with meta_open(source[0].rstr()) as ifd:
        for line in ifd:
            for word in line.strip().split():
                words[word] = words.get(word, 0) + 1
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(["%s %d" % (k, v) for k, v in words.iteritems()]) + "\n")            
    return None

def TOOLS_ADD(env):
    env.Append(BUILDERS = {
        "Resample" : Builder(action=resample),
        "CollectText" : Builder(action=collect_text),
        "CleanWords" : Builder(action=clean_words),
        "PrepareSegmentationsForRelease" : Builder(action=Action(prepare_segmentations_for_release, varlist=["NON_ACOUSTIC_GRAPHEMES"])),
        "KeywordsToList" : Builder(action=keywords_to_list),
        "PennToTranscripts" : Builder(action=penn_to_transcripts),
        "GenerateDataSubset" : Builder(action=generate_data_subset),
        "StmToData" : Builder(action=stm_to_data),
        "ExtractTranscripts" : Builder(action=extract_transcripts),
        "TranscriptsToData" : Builder(action=transcripts_to_data),
        "CONLLishToXML" : Builder(action=conllish_to_xml),
        #"CollateResults" : Builder(action=collate_results),
        #"TopWordsByTag" : Builder(action=top_words_by_tag),
        "RtmToData" : Builder(action=rtm_to_data),
        "BuildSite" : Builder(action=build_site, emitter=build_site_emitter),
        "BuildPropertyTables" : Builder(action=build_property_tables, emitter=build_property_tables_emitter),
        "BuildExtrinsicTables" : Builder(action=build_extrinsic_tables, emitter=build_extrinsic_tables_emitter),
        "SegmentTranscripts" : Builder(action=segment_transcripts),
        "VocabularyComparison" : Builder(action=vocabulary_comparison),
        "WordList" : Builder(action=word_list),
    })
    env.AddMethod(morfessor_babel_experiment, "MorfessorBabelExperiment")
    env.AddMethod(adaptor_grammar_babel_experiment, "AdaptorGrammarBabelExperiment")
    #env.AddMethod(run_asr, "RunASR")
    #env.AddMethod(run_kws, "RunKWS")
