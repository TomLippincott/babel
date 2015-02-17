from SCons.Builder import Builder
from SCons.Action import Action
import re
from glob import glob
from functools import partial
import logging
import os.path
from os.path import join as pjoin
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
from common_tools import meta_open, temp_dir
from babel import ProbabilityList, Arpabo, Pronunciations, Vocabulary


class CFG():
    dbFile = '${DATABASE_FILE}'
    #dictFile = '${PRONUNCIATIONS_FILE}'
    #vocab = '${VOCABULARY_FILE}'
    ctmDir = '${CTM_PATH}'
    latDir = '${LATTICE_PATH}'
    ecf_file = '${ECF_FILE}'
    keyword_file = '${KEYWORD_FILE}'    
    rttm_file = '${RTTM_FILE}'
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


def query_files(target, source, env):
    # OUTPUT:
    # iv oov map w2w <- 
    # INPUT: kw, iv, id
    # pad id to 4
    remove_vocab = source[-1].read()
    with meta_open(source[0].rstr()) as kw_fd, meta_open(source[1].rstr()) as iv_fd:
        keyword_xml = et.parse(kw_fd)
        keywords = set([(x.get("kwid"), x.find("kwtext").text.lower()) for x in keyword_xml.getiterator("kw")])
        vocab = [x.decode("utf-8") for x in Pronunciations(iv_fd).get_words()]
        if remove_vocab:
            remove_vocab = Vocabulary(meta_open(remove_vocab)).get_words()
        else:
            remove_vocab = []
        iv_keywords = sorted([(int(tag.split("-")[-1]), tag, term) for tag, term in keywords if all([y in vocab for y in term.split()]) and term not in remove_vocab])
        oov_keywords = sorted([(int(tag.split("-")[-1]), tag, term) for tag, term in keywords if any([y not in vocab for y in term.split()])])
        language_id = source[-2].read()
        with meta_open(target[0].rstr(), "w") as iv_ofd, meta_open(target[1].rstr(), "w") as oov_ofd, meta_open(target[2].rstr(), "w") as map_ofd, meta_open(target[3].rstr(), "w") as w2w_ofd, meta_open(target[4].rstr(), "w") as kw_ofd:
            iv_ofd.write("\n".join([x[2].encode("utf-8") for x in iv_keywords]))
            oov_ofd.write("\n".join([x[2].encode("utf-8") for x in oov_keywords]))
            map_ofd.write("\n".join(["%s %.5d %.5d" % x for x in 
                                     sorted([("iv", gi, li) for li, (gi, tag, term) in enumerate(iv_keywords, 1)] + 
                                            [("oov", gi, li) for li, (gi, tag, term) in enumerate(oov_keywords, 1)], lambda x, y : cmp(x[1], y[1]))]))
            w2w_ofd.write("\n".join([("0 0 %s %s 0" % (x.encode("utf-8"), x.encode("utf-8"))) for x in vocab if x != "VOCAB_NIL_WORD"] + ["0"]))
            for x in keyword_xml.getiterator("kw"):
                x.set("kwid", "KW%s-%s" % (language_id, x.get("kwid").split("-")[-1]))
            keyword_xml.write(kw_ofd) #.write(et.tostring(keyword_xml.))
    return None

def ecf_file(target, source, env):
    with open(source[0].rstr()) as fd:
        files = [(fname, int(chan), float(begin), float(end)) for fname, num, sph, chan, begin, end in [line.split() for line in fd]]
        data = {}
        for fname in set([x[0] for x in files]):
            relevant = [x for x in files if x[0] == fname]
            start = sorted(relevant, lambda x, y : cmp(x[2], y[2]))[0][2]
            end = sorted(relevant, lambda x, y : cmp(x[3], y[3]))[-1][3]
            channels = relevant[0][1]
            data[fname] = (channels, start, end)
        total = sum([x[2] - x[1] for x in data.values()])
        tb = et.TreeBuilder()
        tb.start("ecf", {"source_signal_duration" : "%f" % total, "language" : "", "version" : ""})
        for k, (channel, start, end) in data.iteritems():
            tb.start("excerpt", {"audio_filename" : k, "channel" : str(channel), "tbeg" : str(start), "dur" : str(end-start), "source_type" : "splitcts"})
            tb.end("excerpt")
        tb.end("ecf")        
        with open(target[0].rstr(), "w") as ofd:
            ofd.write(et.tostring(tb.close()))
    return None

def word_pronounce_sym_table(target, source, env):
    """
    convert dictionary format:
      
      WORD(NUM) WORD

    to numbered format:

      WORD(NUM) NUM

    with <EPSILON> as 0
    """
    ofd = meta_open(target[0].rstr(), "w")
    ofd.write("<EPSILON>\t0\n")
    for i, line in enumerate(meta_open(source[0].rstr())):
        ofd.write("%s\t%d\n" % (line.split()[0], i + 1))
    ofd.close()
    return None

def clean_pronounce_sym_table(target, source, env):
    """
    Deduplicates lexicon, removes "(NUM)" suffixes, and adds <query> entry.
    """
    with meta_open(source[0].rstr()) as ifd:
        words = set([re.match(r"^(.*)\(\d+\)\s+.*$", l).groups()[0] for l in ifd if not re.match(r"^(\<|\~).*", l)])
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(["%s\t%d" % (w, i) for i, w in enumerate(["<EPSILON>", "</s>", "<HES>", "<s>", "~SIL"] + sorted(words) + ["<query>"])]) + "\n")
    return None

def munge_dbfile(target, source, env):
    """
    NEEDS WORK!
    Converts a database file.
    """
    with meta_open(source[1].rstr()) as fd:
        lattice_files = set([os.path.basename(x.strip()) for x in fd])
    ofd = meta_open(target[0].rstr(), "w")
    for line in meta_open(source[0].rstr()):
        toks = line.split()
        fname = "%s.fsm.gz" % ("#".join(toks[0:2]))
        if fname in lattice_files:
            ofd.write(" ".join(toks[0:4] + ["0.0", toks[5]]) + "\n")
    ofd.close()
    return None

def keyword_xml_to_text(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        keywords = [k.text for k in et.parse(ifd).getiterator("kwtext")]
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(("\n".join(keywords)).encode("utf-8"))
    return None

def create_data_list(target, source, env):
    """
    NEEDS WORK!
    Creates the master list of lattice transformations.
    """
    args = source[-1].read()
    data = {}
    for line in meta_open(source[0].rstr()):
        toks = line.split()
        bn = os.path.basename(toks[2])
        data[toks[0]] = data.get(toks[0], {})
        data[toks[0]][toks[1]] = (bn, toks[4], toks[5])
    ofd = meta_open(target[0].rstr(), "w")
    for lattice_file in glob(os.path.join(args["LATTICE_DIR"], "*")):
        bn = os.path.basename(lattice_file)
        path = os.path.join(env["BASE_PATH"], "lattices")
        uttname, delim, uttnum = re.match(r"(.*)([^\w])(\d+)\.%s$" % (args["oldext"]), bn).groups()
        try:
            name, time, timeend = data[uttname][uttnum]
            newname = os.path.abspath(os.path.join(path, "%s%s%s.%s" % (uttname, delim, uttnum, args["ext"])))
            #ofd.write(env.subst("%s %s %s %s %s.osym %s" % (os.path.splitext(name)[0], time, timeend, newname, newname, os.path.abspath(lattice_file))) + "\n")
            ofd.write(env.subst("%s %s %s %s %s" % (os.path.abspath(lattice_file), os.path.splitext(name)[0], time, timeend, newname)) + "\n")
            #os.path.splitext(name)[0], time, timeend, newname, newname, os.path.abspath(lattice_file))) + "\n")
        except:
            continue
            return "lattice file not found in database: %s (are you sure your database file matches your lattice directory?)" % bn
    ofd.close()
    return None

def get_file_list(target, source, env):
    """
    """
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join([os.path.abspath(x.rstr()) for x in source]) + "\n")
    return None

def run_kws(env, experiment_path, asr_output, vocabulary, pronunciations, *args, **kw):
    cfg = CFG(env)
    env.Replace(EPSILON_SYMBOLS="'<s>,</s>,~SIL,<HES>'")
    #cfg.dictFile -> pronunciations,
    #cfg.vocab -> vocabulary

    devinfo = env.File("${MODEL_PATH}/devinfo")
    
    iv_query_terms, oov_query_terms, term_map, word_to_word_fst, kw_file = env.QueryFiles([pjoin(experiment_path, x) for x in ["iv_queries.txt", 
                                                                                                                               "oov_queries.txt",
                                                                                                                               "term_map.txt",
                                                                                                                               "word_to_word.fst",
                                                                                                                               "kwfile.xml"]], 
                                                                                          [cfg.keyword_file, pronunciations, env.Value(str(env["BABEL_ID"])), env.Value("")])

    wordpron = env.WordPronounceSymTable(pjoin(experiment_path, "in_vocabulary_symbol_table.txt"),
                                         pronunciations)

    vocabulary_symbols = env.CleanPronounceSymTable(pjoin(experiment_path, "cleaned_in_vocabulary_symbol_table.txt"),
                                                    wordpron)

    padfst = env.BuildPadFST(pjoin(experiment_path, "pad_fst.txt"),
                             wordpron)

    p2p_fst = env.FSTCompile(pjoin(experiment_path, "p2p_fst.txt"),
                             [vocabulary_symbols, word_to_word_fst])

    iv_queries = env.QueryToPhoneFST(pjoin(experiment_path, "iv_queries", "iv_query.fst"), 
                                     [p2p_fst, vocabulary_symbols, vocabulary, iv_query_terms, env.Value({"n" : 1, "I" : 1, "OUTDIR" : pjoin(experiment_path, "iv_queries")})])

    env.Replace(n=1, I=1, OUTDIR=pjoin(experiment_path, "oov_queries"))
    oov_queries = env.QueryToPhoneFST(pjoin(experiment_path, "oov_queries", "oov_query.fst"), 
                                      [p2p_fst, vocabulary_symbols, vocabulary, oov_query_terms, env.Value({"n" : 1, "I" : 1, "OUTDIR" : pjoin(experiment_path, "oov_queries")})])

    all_lattices = env.Textfile(os.path.join(experiment_path, "all_lattices.txt"), [x[1] for x in asr_output])
    full_mdb = env.MungeDatabase(pjoin(experiment_path, "full_munged_database.txt"),
                            [cfg.dbFile, all_lattices])
    full_ecf_file = env.ECFFile(pjoin(experiment_path, "full_ecf.xml"), full_mdb)    

    xml_template = env.File("${LORELEI_SVN}/${BABEL_ID}/LimitedLP/kws-resources/kws-resources-IndusDB.20140206.NWAY.dev/template.word.xml")
    
    iv_searches = []
    oov_searches = []
    asr_lattice_path = env.Dir(os.path.join(os.path.split(os.path.dirname(asr_output[0][0].rstr()))[0], "lat")).rstr()
    for ctm, lattice_list in asr_output:
        i = int(re.match(r".*?(\d+).ctm$", ctm.rstr()).group(1))
        mdb = env.MungeDatabase(pjoin(experiment_path, "munged_database-%d.txt" % (i)),
                                [cfg.dbFile, lattice_list])
        
        data_list = env.CreateDataList(pjoin(experiment_path, "data_list-%d.txt" % (i)),
                                       [mdb] + [env.Value({"oldext" : "fsm.gz", 
                                                           "ext" : "fst",
                                                           "subdir_style" : "hub4",
                                                           "LATTICE_DIR" : asr_lattice_path,
                                                       })], BASE_PATH=experiment_path)
        ecf_file = env.ECFFile(pjoin(experiment_path, "ecf-%d.xml" % (i)), mdb)
        
        idx, isym, osym = env.LatticeToIndex([pjoin(experiment_path, "lattices", "index-%d" % (i)),
                                              pjoin(experiment_path, "lattices", "isym-%d" % (i)),
                                              pjoin(experiment_path, "lattices", "osym-%d" % (i))],
                                             [data_list, wordpron, pronunciations])

        iv_searches.append(env.StandardSearch(pjoin(experiment_path, "iv_search_output-%d.txt" % (i)),
                                              [xml_template, data_list, idx, isym, osym, iv_queries]))
        oov_searches.append(env.StandardSearch(pjoin(experiment_path, "oov_search_output-%d.txt" % (i)),
                                               [xml_template, data_list, idx, isym, osym, oov_queries]))

    iv_search_outputs = env.GetFileList(pjoin(experiment_path, "iv_search_outputs.txt"), iv_searches)
    iv_merged = env.MergeSearchFromParIndex(pjoin(experiment_path, "iv_merged.txt"), iv_search_outputs)
    iv_sto_norm = env.SumToOneNormalize(pjoin(experiment_path, "iv_sto_norm.txt"), iv_merged)

    oov_search_outputs = env.GetFileList(pjoin(experiment_path, "oov_search_outputs.txt"), oov_searches)
    oov_merged = env.MergeSearchFromParIndex(pjoin(experiment_path, "oov_merged.txt"), oov_search_outputs)
    oov_sto_norm = env.SumToOneNormalize(pjoin(experiment_path, "oov_sto_norm.txt"), oov_merged)

    merged = env.MergeIVOOVCascade(pjoin(experiment_path, "merged.txt"), [iv_sto_norm, oov_sto_norm])
    # iterate here
    # merged = env.MergeIVOOVCascade(pjoin(experiment_path, "merged.txt"), [merged, excluded_xml])

    dt = env.ApplyRescaledDTPipe(pjoin(experiment_path, "dt.kwslist.xml"), [devinfo, cfg.dbFile, full_ecf_file, merged])
    kws_score = env.BabelScorer([pjoin(experiment_path, "score.%s" % x) for x in ["alignment.csv", "bsum.txt", "sum.txt"]],
                                [cfg.ecf_file, cfg.rttm_file, cfg.keyword_file, dt])
    return None

def TOOLS_ADD(env):
    BUILDERS = {
        "ECFFile" : Builder(action=ecf_file),
        "WordPronounceSymTable" : Builder(action=word_pronounce_sym_table), 
        "CleanPronounceSymTable" : Builder(action=clean_pronounce_sym_table), 
        "MungeDatabase" : Builder(action=munge_dbfile), 
        "CreateDataList" : Builder(action=create_data_list),
        "GetFileList" : Builder(action=get_file_list), 
        "QueryFiles" : Builder(action=query_files),
        "BuildPadFST" : Builder(action="${BUILDPADFST} ${SOURCE} ${TARGET}"),
        "FSTCompile" : Builder(action="${FSTCOMPILE} --isymbols=${SOURCES[0]} --osymbols=${SOURCES[0]} ${SOURCES[1]} > ${TARGETS[0]}"), 
        "QueryToPhoneFST" : Builder(action="${QUERY2PHONEFST} -p ${SOURCES[0]} -s ${SOURCES[1]} -d ${SOURCES[2]} -l ${TARGETS[0]} -n ${n} -I ${I} ${OUTDIR} ${SOURCES[3]} 2> /dev/null"),
        "MergeIVOOVCascade" : Builder(action="perl ${MERGEIVOOVCASCADE} ${SOURCES[0]} ${SOURCES[1]} ${TARGETS[0]}"),
        "LatticeToIndex" : Builder(action="${LAT2IDX} -D ${SOURCES[0]} -s ${SOURCES[1]} -d ${SOURCES[2]} -S ${EPSILON_SYMBOLS} -I ${TARGETS[0]} -i ${TARGETS[1]} -o ${TARGETS[2]} 2> /dev/null"),
        "StandardSearch" : Builder(action="${STDSEARCH} -F ${TARGETS[0]} -a IARPA-babel206b-v0.1e_conv-dev.kwlist.xml -d ${SOURCES[1]} -i ${SOURCES[2]} -s ${SOURCES[3]} -b KW${BABEL_ID}-0 -o ${SOURCES[4]} ${SOURCES[5]} 2> /dev/null"),
        "MergeSearchFromParIndex" : Builder(action="${MERGESEARCHFROMPARINDEXPRL} ${SOURCE} > ${TARGET}"),
        "SumToOneNormalize" : Builder(action="${SUMTOONENORMALIZE} < ${SOURCE} > ${TARGET}"),
        "ApplyRescaledDTPipe" : Builder(action="python ${APPLYRESCALEDDTPIPE} ${SOURCES[0]} ${SOURCES[1]} ${SOURCES[2]} < ${SOURCES[3]} > ${TARGETS[0]} 2> /dev/null"),
        "BabelScorer" : Builder(action="perl -X ${BABELSCORER} -e ${SOURCES[0]} -r ${SOURCES[1]} -t ${SOURCES[2]} -s ${SOURCES[3]} -c -o -b -d -a --ExcludePNGFileFromTxtTable -f ${'.'.join(TARGETS[0].rstr().split('.')[0:-2])} -y TXT"),
        "KeywordXMLToText" : Builder(action=keyword_xml_to_text),
    }
    
    env.AddMethod(run_kws, "RunKWS")
    env.Append(BUILDERS=BUILDERS)
