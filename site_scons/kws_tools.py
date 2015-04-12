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
import StringIO as stringio
from common_tools import meta_open, temp_dir, pairs
from babel import ProbabilityList, Arpabo, Pronunciations, Vocabulary
from openfst_tools import *
import codecs
from scons_tools import make_tar_builder

def query_files(target, source, env):
    with meta_open(source[0].rstr(), enc=None) as kw_fd, meta_open(source[1].rstr()) as iv_fd:
        keyword_xml = et.parse(kw_fd)
        keywords = list(set([(x.get("kwid"), x.find("kwtext").text.lower()) for x in keyword_xml.getiterator("kw")]))
        vocab = [x for x in Pronunciations(iv_fd).get_words()]
        iv_keywords = sorted([(int(tag.split("-")[-1]), tag, term) for tag, term in keywords if all([y in vocab for y in term.split()])])
        oov_keywords = sorted([(int(tag.split("-")[-1]), tag, term) for tag, term in keywords if any([y not in vocab for y in term.split()])])
    with meta_open(target[0].rstr(), "w") as iv_ofd, meta_open(target[1].rstr(), "w") as oov_ofd, meta_open(target[2].rstr(), "w") as w2w_ofd:
        iv_ofd.write("\n".join(["%s %s" % (x[2], x[1]) for x in iv_keywords]) + "\n")
        oov_ofd.write("\n".join(["%s %s" % (x[2], x[1]) for x in oov_keywords]) + "\n")
        w2w_ofd.write("\n".join([("0 0 %s %s 0" % (x, x)) for x in vocab if x != "VOCAB_NIL_WORD"] + ["0"]))
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

def keyword_xml_to_text(target, source, env):
    with meta_open(source[0].rstr(), enc=None) as ifd:
        words = sum([k.text.split() for k in et.parse(ifd).getiterator("kwtext")], [])
        words = set(sum([w.strip("-").split("-") for w in words if "_" not in w], []))
    with meta_open(target[0].rstr(), "w") as ofd:
        #x = list(keywords)[0]
        #print dir(x)
        ofd.write(("\n".join([" ".join(["^^^"] + [c for c in x] + ["$$$"]) for x in words])))
    return None

def oov_pronunciations(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        known_words = set([re.match(r"^(.*)\(\d+\)$", l.split()[0]).group(1) for l in ifd])
    with meta_open(source[1].rstr()) as ifd:
        other_words = set(sum([[w for w in l.strip().split()[0:-1]] for l in ifd], []))
        unknown_words = [w for w in other_words if w not in known_words]
    with meta_open(target[0].rstr(), "w") as ofd:
        for w in sorted(unknown_words):            
            pronunciation = ["u%.4x" % (ord(c)) for c in w if c != "_"]
            ofd.write("%s(01) %s\n" % (w, " ".join(pronunciation)))
    return None

def fix_ids(target, source, env):
    term_map = {}
    with meta_open(source[1].rstr()) as ifd:
        for line in ifd:
            n, t, f = line.strip().split()
            if n == source[2].read():
                term_map[f] = t
    with meta_open(source[0].rstr(), enc=None) as ifd:
        xml = et.parse(ifd)
        for k in xml.getiterator("detected_kwlist"):
            a, b = k.get("kwid").split("-")
            k.set("kwid", "-".join([a, term_map[b]]))
    with meta_open(target[0].rstr(), "w", enc=None) as ofd:
        xml.write(ofd)        
    return None

def make_index(target, source, env):
    transparent = env.get("TRANSPARENT").strip("'").split(",")
    thresh_words = float(env.get("PRINT_WORDS_THRESH"))
    thresh_eps = float(env.get("PRINT_EPS_THRESH"))
    lines = []
    maxi = 0
    count = 2
    new_start_node = 0
    previous_final_node = 0
    snodes = {}
    fnodes = set()
    with tarfile.open(source[0].rstr()) as ifd:
        for member in ifd.getnames():
            nodes = set()
            name = os.path.basename(member).split(".")[0]
            if count == 2:
                lines.append("0 1 <epsilon> %s 0.0" % (name))
            count = previous_final_node + 1
            lines.append("%d %d <epsilon> <epsilon> 0.0" % (count, count + 1))
            if count > 1:
                snodes[count] = name
            startnode = count
            count += 1
            reader = codecs.getreader("utf-8")
            for line in reader(ifd.extractfile(member)):
                toks = line.strip().split()
                start_time = toks[0]
                duration = toks[1]
                words = [(word.split("(")[0], float(weight)) for word, weight in pairs(toks[2:], 2) if word.split("(")[0] not in transparent]
                total_score = sum([x[1] for x in words])
                if total_score > thresh_words:
                    end_time = float(toks[0]) + float(toks[1])
                    label = "%s-%s" % (start_time, end_time)
                    for word, weight in words:
                        if weight == 0:
                            weight = 1.0e-8
                        elif weight > 1:
                            weight = 1.0
                        score = -math.log(weight)
                        lines.append("%d %d %s %s %.18f" % (count, count + 1, word, label, score))
                    remaining_score = 1.0 - score
                    if remaining_score >= thresh_eps:
                        remaining_score = -math.log(remaining_score)
                        lines.append("%d %d %s %s %.18f" % (count, count + 1, "<epsilon>", label, remaining_score))
                    count += 1
                    nodes.add(count)
            final_node = count
            for node in nodes:
                if node != final_node:
                    lines.append("%d %d <epsilon> <epsilon> 0" % (startnode, node))
                lines.append("%d %d <epsilon> <epsilon> 0" % (node, final_node + 1))
            count += 1
            fnodes.add(count)
            if count > maxi:
                maxi = count
            previous_final_node = count
    for k, v in snodes.iteritems():
        lines.append("%d %d <epsilon> %s 0" % (new_start_node, k, v))
    new_end_node = maxi + 1
    for node in fnodes:
        lines.append("%d %d <epsilon> <epsilon> 0" % (k, new_end_node))
    lines.append("%d" % (new_end_node))
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(lines))  
    return None

def index_to_symbol_tables(target, source, env):
    osyms = set()
    bsyms = set()
    with meta_open(source[0].rstr()) as ifd:
        for line in ifd:
            toks = line.strip().split()
            if len(toks) == 1:
                continue
            isym = toks[2]
            osym = toks[3]
            osyms.add(osym)
            if not (isym == "<epsilon>" and osym == "<epsilon>"):
                bsym = "%s:%s" % (isym, osym)
                bsym.replace("<epsilon>:", "").replace(":<epsilon>", "")
                bsyms.add(bsym)

    with meta_open(target[0].rstr(), "w") as osym_ofd:
        osym_ofd.write("<epsilon> 0\n")
        for i, osym in enumerate(osyms):
            osym_ofd.write("%s %d\n" % (osym, i + 1))
    with meta_open(target[1].rstr(), "w") as bsym_ofd:
        bsym_ofd.write("<epsilon> 0\n")
        for i, bsym in enumerate(bsyms):
            bsym_ofd.write("%s %d\n" % (bsym, i + 1))
    return None

def combine_symbol_tables(target, source, env):
    symbols = set()
    for fname in source:
        with meta_open(fname.rstr()) as ifd:
            for line in ifd:
                toks = line.strip().split()
                symbol = toks[0]
                symbols.add(symbol)
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("<epsilon> 0\n")
        ofd.write("\n".join(["%s %d" % (s, i + 1) for i, s in enumerate(symbols)]) + "\n")
    return None

def add_word_breaks(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        lines = [x.strip().split() for x in ifd]
    with meta_open(target[0].rstr(), "w") as ofd:
        for line in lines:
            if len(line) == 2:
                ofd.write("%s [ wb ]\n" % (" ".join(line)))
            else:
                ofd.write("%s %s [ wb ] %s [ wb ]\n" % (line[0], line[1], " ".join(line[2:])))
    return None

def add_phone(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        data = {int(d) : s for s, d in [l.strip().split() for l in ifd]}
    m = max(data.keys())
    for i, s in enumerate(source[1].read()):
        data[m + i + 1] = s
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(["%s %d" % (s, d) for d, s in sorted(data.iteritems())]) + "\n")
    return None

def create_p2p(target, source, env):
    add_delete = env.get("ADD_DELETE")
    add_insert = env.get("ADD_INSERT")
    threshold = 100
    with meta_open(source[0].rstr()) as ifd, meta_open(target[0].rstr(), "w") as ofd:
        for line in ifd:
            if float(line.split()[-1]) <= threshold:
                ofd.write(re.sub(r"<EPSILON>", "<epsilon>", line))
        with meta_open(source[1].rstr()) as pifd:        
            for line in pifd:
                if not re.match(r".*epsilon.*", line, re.I):
                    s = line.split()[0]
                    if add_delete > 0:
                        ofd.write("0 0 %s <epsilon> %f\n" % (s, add_delete))
                    if add_insert > 0:
                        ofd.write("0 0 <epsilon> %s %f\n" % (s, add_insert))                        
    return None

def run_kws(env, experiment_path, asr_output, vocabulary, pronunciations, keyword_file, *args, **kw):
    env.Replace(EPSILON_SYMBOLS="'<s>,</s>,~SIL,<HES>'")

    if not env["RUN_KWS"]:
        return None

    pronunciations = env.AddWordBreaks(pjoin(experiment_path, "pronunciations.txt"), pronunciations)
    
    database_file = env.Glob('${DATABASE_FILE}')[0]
    ecf_file = env.Glob("${ECF_FILE}")[0]
    rttm_file = env.Glob("${RTTM_FILE}")[0]
    devinfo = env.File("${MODEL_PATH}/devinfo")
    p2p_file = env.File("${P2P_FILE}")
    expid = os.path.basename(ecf_file.rstr()).split("_")[0]

    iv_query_terms, oov_query_terms, word_to_word_fst = env.QueryFiles([pjoin(experiment_path, x) for x in ["iv_queries.txt", 
                                                                                                            "oov_queries.txt",
                                                                                                            "word_to_word.fst"]],
                                                                       [keyword_file, pronunciations])

    keyword_symbols = env.KeywordSymbols(pjoin(experiment_path, "keywords.sym"), [iv_query_terms, oov_query_terms])

    oov_pronunciations_nobreak = env.OOVPronunciations(pjoin(experiment_path, "oov_pronunciations_nobreak.txt"), [pronunciations, oov_query_terms])
    oov_pronunciations = env.AddWordBreaks(pjoin(experiment_path, "oov_pronunciations.txt"), oov_pronunciations_nobreak)
    
    fst_header, phone_symbols, word_symbols, p2w_fsm, p2w_fst, w2p_fsm, w2p_fst = env.WordsToPhones(
        [pjoin(experiment_path, x) for x in ["fst_header", "phones.sym", "words.sym", "phones2words.fsm", "phones2words.fst", "words2phones.fsm", "words2phones.fst"]],
        [pronunciations, oov_pronunciations]
    )
    phone_symbols = env.AddPhone(pjoin(experiment_path, "p2p_workaround", "phones.sym"), [phone_symbols, Value(["u0071", "HES01", "HES02"])])
    #p2p_fst = env.PhonesToPhones(pjoin(experiment_path, "p2p_workaround", "P2P.fst"), p2p_file)
    p2p_fsm = env.PhonesToPhones(pjoin(experiment_path, "P2P.fsm"), [p2p_file, phone_symbols])
    p2p_unsorted = env.FSTCompile(pjoin(experiment_path, "P2P_unsorted.fst"), [phone_symbols, phone_symbols, p2p_fsm])
    p2p_fst = env.FSTArcSort(pjoin(experiment_path, "P2P.fst"), [p2p_unsorted, Value("ilabel")])

    p2w_fst = env.File("tokpisin_cn_kws/data/OFST/phones2words.fst")
    w2p_fst = env.File("tokpisin_cn_kws/data/OFST/words2phones.fst")
    p2p_fst = env.File("tokpisin_cn_kws/data/OFST/P2P.fst")
    word_symbols = env.File("tokpisin_cn_kws/data/OFST/words.sym")
    
    #p2p_fst = env.PhonesToPhones(pjoin(experiment_path, "P2P.fst"), phone_symbols)
    #return None
    wordpron = env.WordPronounceSymTable(pjoin(experiment_path, "in_vocabulary_symbol_table.txt"),
                                         pronunciations)

    vocabulary_symbols = env.CleanPronounceSymTable(pjoin(experiment_path, "cleaned_in_vocabulary_symbol_table.txt"),
                                                    wordpron)

    #p2p_fst = env.FSTCompile(pjoin(experiment_path, "p2p_fst.txt"),
    #                         [vocabulary_symbols, vocabulary_symbols, word_to_word_fst])

    #iv_queries = env.QueryToPhoneFST(pjoin(experiment_path, "iv_queries", "list.txt"), 
    #                                 [p2p_fst, vocabulary_symbols, vocabulary, iv_query_terms], I=1, n=1)
    iv_queries = []
    for i, terms in enumerate(env.SplitList([pjoin(experiment_path, "iv_queries_%d.txt" % i) for i in range(10)], iv_query_terms)):
        iv_queries.append(env.CreateQueries(pjoin(experiment_path, "iv_query_fsts_%d.tgz" % i), [terms, word_symbols, p2p_fst, w2p_fst, p2w_fst], NBESTP2P=2000))

    oov_queries = env.CreateQueries(pjoin(experiment_path, "oov_queries.tgz"), [oov_query_terms, word_symbols, p2p_fst, w2p_fst, p2w_fst], NBESTP2P=2)


    #iv_queries = env.CreateQueryFSTs(pjoin(experiment_path, "iv_queries", "list.txt"), 
    #                                 [iv_query_terms, p2p_fst, vocabulary_symbols, vocabulary], I=1, n=1)

    #oov_queries = env.CreateQueryFSTs(pjoin(experiment_path, "oov_queries", "list.txt"), 
    #                                  [oov_query_terms, p2p_fst, vocabulary_symbols, vocabulary], I=1, n=1, NBESTP2P=2)

    #iv_queries = env.FileList(pjoin(experiment_path, "iv_queries", "list2.txt"), iv_queries)
    #oov_queries = env.FileList(pjoin(experiment_path, "oov_queries", "list2.txt"), oov_queries)
    
    #oov_queries = env.QueryToPhoneFST(pjoin(experiment_path, "oov_queries", "list.txt"), 
    #                                  [p2p_fst, vocabulary_symbols, vocabulary, oov_query_terms], I=1, n=1)
    #return None
    #iv_queries = env.CreateQueryFSTs(pjoin(experiment_path, "iv_queries", "list.txt"), 
    #                                 [p2p_fst, vocabulary_symbols, vocabulary, iv_query_terms], I=1, n=1)

    #oov_queries = env.CreateQueryFSTs(pjoin(experiment_path, "oov_queries", "list.txt"), 
    #                                  [p2p_fst, vocabulary_symbols, vocabulary, oov_query_terms], I=1, n=1)
    #return None
    ivs = []
    oovs = []

    for lattices in asr_output:
        i = int(re.match(r".*?(\d+)_of_(\d+).*$", lattices.rstr()).group(1))
        index = env.MakeIndex(pjoin(experiment_path, "index-%d.fsm" % i), lattices)
        sym, bsym = env.IndexToSymbolTables([pjoin(experiment_path, "index-%d-sorted.%s" % (i, x)) for x in ["sym", "bsym"]],
                                            [index, keyword_symbols])

        kwd_sym = env.CombineSymbolTables(pjoin(experiment_path, "index-kwd-%d.sym" % (i)), [sym, word_symbols])
        index_fst = env.FSTCompile(pjoin(experiment_path, "index-%d.fst" % (i)), [word_symbols, sym, index])
        sorted_index_fst = env.FSTArcSort(pjoin(experiment_path, "index-%d-sorted.fst" % (i)), [index_fst[0], env.Value("ilabel")])
        
        ivs.append(env.PerformSearch(pjoin(experiment_path, "IV_results", "result.%d.txt" % i), [sorted_index_fst, sym, bsym, word_symbols, iv_queries]))
        continue
        oovs.append(env.PerformSearch(pjoin(experiment_path, "OOV_results", "result.%d.txt" % i), [sorted_index_fst, sym, bsym, word_symbols, oov_queries]))

    return None
        
    iv_xml = env.Glob("${IBM_MODELS}/${BABEL_ID}/${PACK}/KWS-resources/kws-resources-IndusDB.20141020/template.iv.xml")[0]
    oov_xml = env.Glob("${IBM_MODELS}/${BABEL_ID}/${PACK}/KWS-resources/kws-resources-IndusDB.20141020/template.oov.xml")[0]

    iv_comb = env.CombineCN([pjoin(experiment_path, x) for x in ["iv_temp_1.txt", "iv_temp_2.txt", "iv.xml"]],
                            [iv_xml] + ivs, NIST_EXPID_CORPUS=expid, LANGUAGE_TEXT=env.subst("${LANGUAGE_NAME}").replace("_", ""))
    oov_comb = env.CombineCN([pjoin(experiment_path, x) for x in ["oov_temp_1.txt", "oov_temp_2.txt", "oov.xml"]],
                             [oov_xml] + oovs, NIST_EXPID_CORPUS=expid, LANGUAGE_TEXT=env.subst("${LANGUAGE_NAME}").replace("_", ""))
    
    iv_sto_norm = env.SumToOneNormalize(pjoin(experiment_path, "iv_norm.xml"), iv_comb[-1])
    oov_sto_norm = env.SumToOneNormalize(pjoin(experiment_path, "oov_norm.xml"), oov_comb[-1])
    merged = env.MergeIVOOVCascade(pjoin(experiment_path, "merged.xml"), [iv_sto_norm, oov_sto_norm])
    
    dt = env.ApplyRescaledDTPipe(pjoin(experiment_path, "dt.kwslist.xml"), [devinfo, database_file, ecf_file, merged])
    kws_score = env.BabelScorer([pjoin(experiment_path, "score.%s" % x) for x in ["alignment.csv", "bsum.txt", "sum.txt"]],
                                [ecf_file, rttm_file, keyword_file, dt])
    return None

def create_queries(target, source, env):
    query_terms, word_symbols, p2p_fst, w2p_fst, p2w_fst = source
    with meta_open(query_terms.rstr()) as ifd:
        terms = {toks[-1] : toks[0:-1] for toks in [line.strip().split() for line in ifd]}
    with meta_open(target[0].rstr(), "w") as ofd:
        for key, words in terms.iteritems():
            name = env.subst("%s.p2p2w.${NBESTP2P}.fst" % (key))
            keyword = "".join(words)
            text = "\n".join(["%d %d %s" % (i, i + 1, w) for i, w in enumerate(words)] + ["%d" % (len(words))]) + "\n"

            pid = subprocess.Popen(shlex.split(env.subst("${OPENFST_BINARIES}/fstcompile --acceptor -isymbols=${SOURCE}", source=word_symbols)),
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            text, error = pid.communicate(text)
            if len(keyword) > env.get("MINPHLENGTH") and env.get("NBESTP2P") > 0:
                pid = subprocess.Popen(shlex.split(env.subst("${OPENFST_BINARIES}/fstcompose - ${SOURCE}", source=w2p_fst)), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                text, error = pid.communicate(text)
                pid = subprocess.Popen(shlex.split(env.subst("${OPENFST_BINARIES}/fstcompose - ${SOURCE}", source=p2p_fst)), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                text, error = pid.communicate(text)
                pid = subprocess.Popen(shlex.split(env.subst("${OPENFST_BINARIES}/fstshortestpath --nshortest=${NBESTP2P} -")), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                text, error = pid.communicate(text)
                pid = subprocess.Popen(shlex.split(env.subst("${OPENFST_BINARIES}/fstrmepsilon")), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                text, error = pid.communicate(text)
                pid = subprocess.Popen(shlex.split(env.subst("${OPENFST_BINARIES}/fstcompose - ${SOURCE}", source=p2w_fst)), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                text, error = pid.communicate(text)
                pid = subprocess.Popen(shlex.split(env.subst("${OPENFST_BINARIES}/fstproject --project_output")), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                text, error = pid.communicate(text)
                pid = subprocess.Popen(shlex.split(env.subst("${OPENFST_BINARIES}/fstrmepsilon")), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                text, error = pid.communicate(text)
                pid = subprocess.Popen(shlex.split(env.subst("${OPENFST_BINARIES}/fstdeterminize")), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                text, error = pid.communicate(text)
                pid = subprocess.Popen(shlex.split(env.subst("${OPENFST_BINARIES}/fstminimize")), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                text, error = pid.communicate(text)                

            info = tarfile.TarInfo(name)
            info.size = len(text)
            ofd.addfile(info, stringio.StringIO(text))
    return None

# def compile_queries(target, source, env):
#     nbestp2p = env.get("NBESTP2P")
#     min_len = env.get("MINPHLENGTH")
#     with meta_open(source[0].rstr()) as ifd, meta_open(target[0].rstr(), "w") as ofd:
#         for name in ifd.getnames():
#             data = ifd.extractfile(name).read().split("\n")
#             if len(
#             "fstcompile --acceptor -isymbols=$wordsym  $name | fstcompose - $w2p | fstcompose - $p2p  |fstshortestpath --nshortest=$NbestP2P - | fstrmepsilon  | fstcompose - $p2w | fstproject --project_output | fstrmepsilon |  fstdeterminize  | fstminimize > $nameop2p2w"
#         pass
    
#     return None

def perform_search(target, source, env):
    index, index_symbols, index_bsymbols, word_symbols = source[0:4]
    terms = source[4:]
    prune = env.get("PRUNE")
    results = []
    for f in terms:
        with meta_open(f.rstr()) as ifd:
            for fname in ifd.getnames():
                key = fname.split(".")[0]
                text = ifd.extractfile(fname).read()
                cmd = env.subst("fstcompose - ${SOURCES[0]} | fstprune -weight=${PRUNE} - | fstrmepsilon | fstprint -isymbols=${SOURCES[3]}  -osymbols=${SOURCES[1]} -  | perl ${CN_KWS_SCRIPTS}/collapse_fields.pl | fstcompile --acceptor -isymbols=${SOURCES[2]} -  | fstshortestpath --nshortest=10000 - | ${CN_KWS_SCRIPTS}/fstprintpaths ${SOURCES[2]} -  |  perl ${CN_KWS_SCRIPTS}/process.all.words.pl - %s | sort -k 5 -gr |perl  ${CN_KWS_SCRIPTS}/clean_result.words.pl -" % key, source=source, target=target)
                pid = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
                out, err = pid.communicate(text)
                results.append(out.strip())
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(results).strip() + "\n")            
    #"fstcompose ./data/IVfst/KW207-11981.p2p2w.2000.fst index_WORD/index.part10.fst | fstprune -weight=10 - | fstrmepsilon | fstprint -isymbols=./data/OFST/words.sym  -osymbols=index_WORD/index.part10.sym -  | perl /home/tom/projects/babel/tokpisin_cn_kws/scripts/collapse_fields.pl | fstcompile --acceptor -isymbols=index_WORD/index.part10.bsym -  | fstshortestpath --nshortest=10000 - | /home/tom/projects/babel/tokpisin_cn_kws/scripts/fstprintpaths index_WORD/index.part10.bsym -  |  perl /home/tom/projects/babel/tokpisin_cn_kws/scripts/process.all.words.pl - KW207-11981 | sort -k 5 -gr |perl  /home/tom/projects/babel/tokpisin_cn_kws/scripts/clean_result.words.pl - >> result_IV/result.clean.10"
    return None

def split_list(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        lines = [l.strip() for l in ifd]
    per = len(lines) / len(target)
    for i, f in enumerate(target):
        with meta_open(f.rstr(), "w") as ofd:
            if i == len(target) - 1:
                to = len(lines)
            else:
                to = (i + 1) * per
            ofd.write("\n".join(lines[i * per : to]) + "\n")
#create_words_to_phones
#merge_iv_oov_cascade
#create_keyword_symbols
#perform_search

def TOOLS_ADD(env):
    BUILDERS = {
        "SplitList" : Builder(action=split_list),
        "FileList" : Builder(action="find ${SOURCES[0].dir} -type f -name \\*fst > ${TARGET}"),
        "AddWordBreaks" : Builder(action=add_word_breaks),
        "AddPhone" : Builder(action=add_phone),
        #"FixIDs" : Builder(action=fix_ids),
        "ECFFile" : Builder(action=ecf_file),
        "WordPronounceSymTable" : Builder(action=word_pronounce_sym_table), 
        "CleanPronounceSymTable" : Builder(action=clean_pronounce_sym_table), 
        #"MungeDatabase" : Builder(action=munge_dbfile), 
        #"CreateDataList" : Builder(action=create_data_list),
        #"GetFileList" : Builder(action=get_file_list), 
        "QueryFiles" : Builder(action=query_files),
        #"CompileQueries" : Builder(action=compile_queries),
        #"BuildPadFST" : Builder(action="${BUILDPADFST} ${SOURCE} ${TARGET}"),

        #"QueryToPhoneFST" : Builder(action="${QUERY2PHONEFST} -p ${SOURCES[0]} -s ${SOURCES[1]} -d ${SOURCES[2]} -l ${TARGETS[0]} -n ${NBESTP2P} -I ${MINPHLENGTH} ${TARGETS[0].get_dir()} ${SOURCES[3]} 2> /dev/null"),
        "MergeIVOOVCascade" : Builder(action="perl ${MERGEIVOOVCASCADE} ${SOURCES[0]} ${SOURCES[1]} ${TARGETS[0]}"),
        #"LatticeToIndex" : Builder(action="${LAT2IDX} -D ${SOURCES[0]} -s ${SOURCES[1]} -d ${SOURCES[2]} -S ${EPSILON_SYMBOLS} -I ${TARGETS[0]} -i ${TARGETS[1]} -o ${TARGETS[2]} 2> /dev/null"),
        #"StandardSearch" : Builder(action="${STDSEARCH} -F ${TARGETS[0]} -a IARPA-babel${BABEL_ID}_conv-dev.kwlist.xml -d ${SOURCES[1]} -i ${SOURCES[2]} -s ${SOURCES[3]} -b KW${BABEL_ID}-0 -o ${SOURCES[4]} ${SOURCES[5]} 2> /dev/null"),
        #"MergeSearchFromParIndex" : Builder(action="${MERGESEARCHFROMPARINDEXPRL} ${SOURCE} > ${TARGET}"),
        "SumToOneNormalize" : Builder(action="${SUMTOONENORMALIZE} < ${SOURCE} > ${TARGET}"),
        "ApplyRescaledDTPipe" : Builder(action="python ${APPLYRESCALEDDTPIPE} ${SOURCES[0]} ${SOURCES[1]} ${SOURCES[2]} < ${SOURCES[3]} > ${TARGETS[0]} 2> /dev/null"),

        "KeywordXMLToText" : Builder(action=keyword_xml_to_text),
        
        #"CNQueryFile" : Builder(action="python ${KWS_SCRIPTS}/term.py ${SOURCES} > ${TARGET}"),
        "KeywordSymbols" : Builder(action="perl ${CN_KWS_SCRIPTS}/kwdsym.pl ${SOURCES} ${TARGET.get_dir()} 2> /dev/null"),
        "OOVPronunciations" : Builder(action=oov_pronunciations),
        "WordsToPhones" : Builder(action="perl ${CN_KWS_SCRIPTS}/create_wp.pl ${SOURCES[0]} ${SOURCES[1]} ${TARGET.get_dir()} ${TRANSPARENT} 2> /dev/null"),
        "PhonesToPhones" : Builder(action=create_p2p),
                                   #"perl ${CN_KWS_SCRIPTS}/create_p2p.pl ${SOURCES[0]} ${TARGET.get_dir()} ${ADD_DELETE} ${ADD_INSERT} 2> /dev/null"),
        #"CreateQueryFSTs" : Builder(action="perl ${CN_KWS_SCRIPTS}/create_query_fsts.pl ${SOURCES[0]} ${OUTDIR} ${SOURCES[1].get_dir()} P2P.fst ${NBESTP2P} ${MINPHLENGTH}"),
        "CreateQueries" : Builder(action=create_queries),
        "CreateQueryFSTs" : Builder(action=[
            "perl ${CN_KWS_SCRIPTS}/create_query_fsts.pl ${SOURCES[0]} ${TARGETS[0].get_dir()} ${SOURCES[1].get_dir()} ${SOURCES[1].file} ${NBESTP2P} ${MINPHLENGTH} 2> /dev/null",
            "find ${TARGETS[0].dir} -type f -name \\*fst > ${TARGETS[0]}"]),
        #"touch ${TARGETS[0]}"]),
        ## python ${KWS_SCRIPTS}/term.py ${IV_XML_TEMPLATE} ${IV_QUERY_FILE} > ${IV_QUERY}
        ## python ${KWS_SCRIPTS}/term.py ${OOV_XML_TEMPLATE} ${OOV_QUERY_FILE} > ${OOV_QUERY}
        ## perl ${KWS_SCRIPTS}/kwdsym.pl ${IV_QUERY} ${OOV_QUERY} ${DATA_FST}
        ## #create words2phones.fst,  phones2words.fst  words.isyms  phones.isyms
        ## perl ${KWS_SCRIPTS}/create_wp.pl ${DICT_IV} ${DICT_OOV} ${DATA_FST} ${TRANSPARENT}
        ## #create p2p fst
        ## perl ${KWS_SCRIPTS}/create_p2p.pl ${P2P} ${DATA_FST} ${addDEL} ${addINS}
        ## perl ${KWS_SCRIPTS}/split_list.pl  ${FILE} ${JOB} ${JOBN} | perl ${KWS_SCRIPTS}/create_query_fsts.pl - ${OUTDIR} ${DATA_FST} P2P.fst ${NBESTP2P} ${MINPHLENGTH}
        "MakeIndex" : Builder(action=make_index),
        "IndexToSymbolTables" : Builder(action=index_to_symbol_tables),
        "CombineSymbolTables" : Builder(action=combine_symbol_tables),
        #"IndexToBsymTable" : Builder(action=index_to_bsym_table),
        #"IndexToKeywordSymTable" : Builder(action=index_to_keyword_sym_table),
        #"CompileIndexToFST" : Builder(action=compile_index_to_fst),
        #"_MakeIndex" : Builder(action=["perl ${CN_KWS_SCRIPTS}/make_index.pl ${SOURCES[0]} ${TARGETS[0].base}.fsm ${PRINT_WORDS_THRESH} ${PRINT_EPS_THRESH} ${TRANSPARENT}",
        #                              "perl ${CN_KWS_SCRIPTS}/compile_fst_index.sym.pl ${TARGETS[0].base}.fsm ${TARGETS[0].get_dir()} ${CN_KWS_SCRIPTS}"]),
        ## perl  ${KWS_SCRIPTS}/split_list.pl ${LIST} ${JOB} ${JOBN} | perl ${KWS_SCRIPTS}/make_index.pl - ${INDEX}/index.part${JOB}.fsm ${PRINT_WORDS_THRESH} ${PRINT_EPS_THRESH} ${TRANSPARENT}
        ## perl ${KWS_SCRIPTS}/compile_fst_index.sym.pl $INDEX/index.part${JOB}.fsm ${DATA_FST} ${KWS_SCRIPTS}
        "SearchQueries" : Builder(action="perl ${CN_KWS_SCRIPTS}/search_queries.pl ${SOURCES[0]} ${SOURCES[1]} ${CN_KWS_SCRIPTS} ${SOURCES[1].get_dir()} ${JOB} ${PRUNE} ${TARGETS[0].get_dir()} 2> /dev/null"),
        "PerformSearch" : Builder(action=perform_search),
        ## ls ${QUERY_DIR}/KW*.p2p2w.${NBESTP2P}.fst  |  perl ${KWS_SCRIPTS}/search_queries.pl - ${INDEX}/index.part${JOB}.fst ${KWS_SCRIPTS} ${DATA_FST} ${JOB} ${PRUNE} ${RESULT}
        # Prepare Cascade
# ${KWSBase}/scripts/run_prepare_nway_cascade.sh word_cfg.sh seg_cfg.sh phone_cfg.sh phone_cfg.sh cascade_cfg.sh

# # Run Cascade Search
# for (( i=1; i<=$N; i++))
# do
#   ${KWSBase}/scripts/run_search_cascade.sh phone_cfg.sh cascade_cfg.sh $i
# done

# # Merged Cascaded Search
# ${KWSBase}/scripts/run_merge.sh cascade_cfg.sh

# # Get Cascaded Results
# ${KWSBase}/scripts/run_postprocess_nway_cascade.sh cascade_cfg.sh ${RESULT_DIR}

        "CombineCN" : Builder(action=["cat ${SOURCES[1:]} > ${TARGETS[0]}",
                                      "cat ${TARGETS[0]} > ${TARGETS[1]}",
                                      #"perl ${CN_KWS_SCRIPTS}/check_result.pl < ${TARGETS[0]} > ${TARGETS[1]}",
                                      "perl  ${CN_KWS_SCRIPTS}/convert_CN.pl ${TARGETS[1]} ${NIST_EXPID_CORPUS} ${LANGUAGE_TEXT} ${SOURCES[0]} > ${TARGETS[2]}"
                                  ]),
        ## cat ${RESULT}/result.clean.* | perl ${KWS_SCRIPTS}/check_result.pl |  perl  ${KWS_SCRIPTS}/convert_CN.pl - ${NIST_EXPID_CORPUS} ${LANGUAGE_TEXT} ${XML_TEMPLATE} > ${RESULT}/result.kwslist.xml
        ## perl ${KWS_SCRIPTS}/applySTONormalization.gamma.thresh.prl < ${RESULT}/result.kwslist.xml > ${RESULT}/result.STO.kwslist.xml
        ## perl ${KWS_SCRIPTS}/merge_iv_oov_cascade_v3.pl RESULT_IV/result.STO.kwslist.xml RESULT_OOV/result.STO.kwslist.xml result.xml
        # python ${KWS_SCRIPTS}/applyRescaledDTpipe.py ${DEVINFOFILE} ${DBFILE} ${DEV_ECF} < result.xml > result.2.xml
        "BabelScorer" : Builder(action="perl -X ${BABELSCORER} -e ${SOURCES[0]} -r ${SOURCES[1]} -t ${SOURCES[2]} -s ${SOURCES[3]} -c -o -b -d -a --ExcludePNGFileFromTxtTable -f ${'.'.join(TARGETS[0].rstr().split('.')[0:-2])} -y TXT 2> /dev/null 1> /dev/null"),
    }
    
    env.AddMethod(run_kws, "RunKWS")
    env.Append(BUILDERS=BUILDERS)
