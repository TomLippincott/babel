from SCons.Builder import Builder
from SCons.Action import Action
import re
from glob import glob
from functools import partial
import logging
import os.path
from os.path import join as pjoin
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
import StringIO as stringio
from common_tools import meta_open, temp_dir, pairs
from babel import ProbabilityList, Arpabo, Pronunciations, Vocabulary
import codecs
from scons_tools import make_tar_builder
import tarfile


def add_word_breaks(target, source, env):
    """Adds the "word break" symbols to a pronunciation file.

    No idea why, but some of IBM's scripts require this.

    Sources: pronunciation file (with no word breaks)
    Targets: pronunciation file (with word breaks)
    """
    with meta_open(source[0].rstr()) as ifd:
        lines = [x.strip().split() for x in ifd]
    with meta_open(target[0].rstr(), "w") as ofd:
        for line in lines:
            if len(line) == 2:
                ofd.write("%s [ wb ]\n" % (" ".join(line)))
            else:
                ofd.write("%s %s [ wb ] %s [ wb ]\n" % (line[0], line[1], " ".join(line[2:])))
    return None


def query_files(target, source, env):
    """Creates lists of in-vocabulary and out-of-vocabulary keywords, and a dummy word-to-word FST.
    
    Sources: keyword XML file, pronunciations file
    Targets: in-vocabulary search terms file, out-of-vocabulary search terms file, word-to-word FST file
    """
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


def keyword_symbols(target, source, env, for_signature):
    """Create OpenFST-format symbol file of two input files that have one symbol per line.

    Targets: input file 1, input file 2
    Sources: symbol file
    """
    return "perl ${CN_KWS_SCRIPTS}/kwdsym.pl ${SOURCES} ${TARGET.get_dir()} ${COMMAND_LINE_SUFFIX}"


def oov_pronunciations(target, source, env):
    """Creates pronunciations for out-of-vocabulary query terms using a G2P model.

    Sources: in-vocabulary pronunciation file, out-of-vocabulary query terms, g2p model
    Targets: out-of-vocabulary pronunciation file
    """
    return None


def graphemic_oov_pronunciations(target, source, env):
    """Creates grapheme-based pronunciations for out-of-vocabulary query terms.

    Sources: in-vocabulary pronunciation file, out-of-vocabulary query terms
    Targets: grapheme-based out-of-vocabulary pronunciation file
    """
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


def words_to_phones(target, source, env, for_signature):
    """Creates several OpenFST files necessary for later stages.

    Sources: in-vocabulary pronunciations, out-of-vocabulary pronunciations, out-of-vocabulary dictionary
    Targets: fst header file, phone symbol file, word symbol file, word-to-phone FSM file, word-to-phone FST file
    """
    return "perl bin/create_wp.more.pl ${SOURCES[0]} ${SOURCES[1]} ${SOURCES[2]} ${TARGET.get_dir()} ${TRANSPARENT} ${COMMAND_LINE_SUFFIX}"


def phones_to_phones(target, source, env):
    """Creates a phone-to-phone FSM to model likely insertion, deletion, and replacement.

    Sources: P2P file, P2P symbol file
    Targets: P2P FSM file
    """
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


def add_phone(target, source, env):
    """Add phones to a baseline phone symbol file.

    Sources: phone symbol file, list of additional phones
    Targets: augmented phone symbol file
    """
    with meta_open(source[0].rstr()) as ifd:
        data = {int(d) : s for s, d in [l.strip().split() for l in ifd]}
    m = max(data.keys())
    for i, s in enumerate(source[1].read()):
        data[m + i + 1] = s
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(["%s %d" % (s, d) for d, s in sorted(data.iteritems())]) + "\n")
    return None


def word_pronounce_sym_table(target, source, env):
    """Convert a dictionary file to an OpenFST symbol file.
    
    Sources: dictionary file
    Targets: symbol file
    """
    ofd = meta_open(target[0].rstr(), "w")
    ofd.write("<EPSILON>\t0\n")
    for i, line in enumerate(meta_open(source[0].rstr())):
        ofd.write("%s\t%d\n" % (line.split()[0], i + 1))
    ofd.close()
    return None


def clean_pronounce_sym_table(target, source, env):
    """Deduplicates lexicon, removes "(NUM)" suffixes, and adds <query> entry.

    Sources: dictionary file
    Targets: symbol file
    """
    with meta_open(source[0].rstr()) as ifd:
        words = set([re.match(r"^(.*)\(\d+\)\s+.*$", l).groups()[0] for l in ifd if not re.match(r"^(\<|\~).*", l)])
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(["%s\t%d" % (w, i) for i, w in enumerate(["<EPSILON>", "</s>", "<HES>", "<s>", "~SIL"] + sorted(words) + ["<query>"])]) + "\n")
    return None


def split_list(target, source, env):
    """Splits a file with N lines into N / k lines.

    Sources: line file, split count
    Targets: split file 1, split file 2 ...
    """
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


def create_queries(target, source, env):
    """Expand and compile each query term into an FST that can be composed with an index FST to perform search.

    Sources: query term file, word symbol file, P2P FST, W2P FST
    Targets: archive of query FSTs
    """
    query_terms, word_symbols, p2p_fst, w2p_fst = source
    with meta_open(query_terms.rstr()) as ifd:
        terms = {toks[-1] : toks[0:-1] for toks in [line.strip().split() for line in ifd]}
    with meta_open(target[0].rstr(), "w") as ofd:
        for key, words in terms.iteritems():
            name = env.subst("%s.p2p2w.${NBESTP2P}.fst" % (key))
            keyword = "".join(words)
            text = "\n".join(["%d %d %s" % (i, i + 1, w) for i, w in enumerate(words)] + ["%d" % (len(words))]) + "\n"
            pid = subprocess.Popen(shlex.split(env.subst("${OPENFST_BINARIES}/fstcompile --acceptor -isymbols=${SOURCE}", source=word_symbols)),
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            text, error = pid.communicate(text.encode("utf-8"))
            pid = subprocess.Popen(shlex.split(env.subst("${OPENFST_BINARIES}/fstcompose - ${SOURCE}", source=w2p_fst)), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            text, error = pid.communicate(text)
            if len(keyword) > env.get("MINPHLENGTH") and env.get("NBESTP2P") > 0:
                pid = subprocess.Popen(shlex.split(env.subst("${OPENFST_BINARIES}/fstcompose - ${SOURCE}", source=p2p_fst)), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                text, error = pid.communicate(text)
                pid = subprocess.Popen(shlex.split(env.subst("${OPENFST_BINARIES}/fstshortestpath --nshortest=${NBESTP2P} -")), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                text, error = pid.communicate(text)
            pid = subprocess.Popen(shlex.split(env.subst("${OPENFST_BINARIES}/fstrmepsilon")), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
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


def make_index(target, source, env):
    """Take lots of small consensus networks and turn them into one huge consensus network FST to be composed with query FSTs.
    
    Sources: archive of consensus networks (from ASR)
    Targets: single FSM of all consensus networks with appropriate labels
    """
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
        for member in sorted(ifd.getnames()):
            nodes = set()
            name = os.path.basename(member)
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
                words = [(word.split("(")[0], float(weight)) for word, weight in pairs(toks[2:], 2)]
                total_score = sum([x[1] for x in words if x[0] not in transparent])
                if total_score > thresh_words:
                    end_time = float(toks[0]) + float(toks[1])
                    label = "%s-%s" % (start_time, end_time)
                    for word, weight in words:
                        if weight == 0:
                            weight = 1.0e-8
                        elif weight > 1:
                            weight = 1.0
                        if word not in transparent:
                            score = -math.log(weight)
                            lines.append("%d %d %s %s %.18f" % (count, count + 1, word, label, score))
                    remaining_score = 1.0 - total_score
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
        lines.append("%d %d <epsilon> <epsilon> 0" % (node, new_end_node))
    lines.append("%d" % (new_end_node))
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(lines) + "\n")  
    return None


def index_to_symbol_tables(target, source, env):
    """Create symbol and transducer symbol tables based on an index file.

    Sources: index file
    Targets: symbol file, bsymbol file
    """
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


def expand_phone_index(target, source, env):
    """Converts a word index into a phone index.

    Sources: word index file, pronunciation dictionary
    Targets: phone index file
    """
    fsm, pdict = source
    pronunciations = {}
    with meta_open(pdict.rstr()) as ifd:
        for line in ifd:
            word, pronunciation = re.match(r"^(\S+?)\(\d+\)\s+(.*)$", line.replace("[ wb ]", "").replace("[wb]", "")).groups()
            pronunciations[word] = pronunciations.get(word, []) + [pronunciation.strip().split()]
    arcs = []
    last_node = -1
    with meta_open(fsm.rstr()) as ifd:
        for line in ifd:
            toks = line.strip().split()
            arcs.append(toks)
            if len(toks) > 1:
                start = int(toks[0])
                end = int(toks[1])
                last_node = max([last_node, start, end])
    last_node += 1
    with meta_open(target[0].rstr(), "w") as ofd:
        for arc in arcs:
            new_arcs = []
            if len(arc) < 2:
                new_arcs.append(arc)
            elif arc[2] == "<epsilon>":
                arc[3] = re.sub(r"\.0$", "", arc[3])
                new_arcs.append(arc)
            else:
                arc[3] = re.sub(r"\.0$", "", arc[3])
                start, end, token, label, score = arc
                label = "%s&%s" % (token, label)
                for phones in sorted(pronunciations[arc[2]]):
                    if len(phones) == 1:
                        new_arcs.append([start, end, phones[0], label, score])
                    else:
                        last_node += 1
                        new_arcs.append([start, str(last_node), phones[0], label, score])
                        for phone in phones[1:-1]:
                            last_node += 1
                            new_arcs.append([str(last_node - 1), str(last_node), phone, "<epsilon>", "0"])
                        new_arcs.append([str(last_node), end, phones[-1], "<epsilon>", "0"])
            ofd.write("\n".join([" ".join(x) for x in new_arcs]) + "\n")            
    return None


def combine_symbol_tables(target, source, env):
    """Concatenates symbol files.

    Sources: symbol file 1, symbol file 2 ...
    Targets: combined symbol file
    """
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


def perform_search(target, source, env):
    """Searches for each query term in the index.

    Sources: index file, phone symbol file, index symbol file, fst header, keyword 1, keyword 2 ...
    Targets: search result file
    """
    index, phone_symbols, index_symbols, fst_header = source[0:4]
    terms = source[4:]
    prune = env.get("PRUNE")
    results = []
    for f in terms:
        with meta_open(f.rstr()) as ifd:
            for fname in ifd.getnames():
                key = fname.split(".")[0]
                e = codecs.getreader("utf-8")
                query = ifd.extractfile(fname).read()
                cmd = env.subst("fstcompose - ${SOURCES[0]} | fstprune -weight=${PRUNE} - | fstrmepsilon | fstprint -isymbols=${SOURCES[1]}  -osymbols=${SOURCES[2]} -  | cat ${SOURCES[3]} - | bin/FsmOp -out-cost - -n-best 50000 -gen | perl ${CN_KWS_SCRIPTS}/process.1.pl - 100 1e-40 %s | sort -k 5 -gr | perl ${CN_KWS_SCRIPTS}/clean_result.words.pl -" % key, source=source, target=target)
                pid = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                out, err = pid.communicate(query)
                if not re.match(r"^\s*$", out):
                    for line in out.strip().split("\n"):
                        toks = line.split()
                        toks[3] = key
                        results.append(" ".join(toks))
    with meta_open(target[0].rstr(), "w", None) as ofd:
        ofd.write("\n".join(results).strip() + "\n")            
    return None


def combine_results(target, source, env, for_signature):
    """Combines and formats a list of search results.

    Sources:
    Targets:
    """
    return ["cat ${SOURCES[1:]} > ${TARGETS[0]}",
            "cat ${TARGETS[0]} > ${TARGETS[1]}",
            "perl  ${CN_KWS_SCRIPTS}/convert_CN.pl ${TARGETS[1]} ${NIST_EXPID_CORPUS} ${LANGUAGE_TEXT} ${SOURCES[0]} > ${TARGETS[2]}",
            ]


def sum_to_one_normalize(target, source, env, for_signature):
    """Performs some normalization.

    Sources:
    Targets:
    """
    return "${SUMTOONENORMALIZE} < ${SOURCE} > ${TARGET}"


def merge_iv_oov_cascade(target, source, env, for_signature):
    """Combine the in-vocabulary and out-of-vocabulary results.

    Sources:
    Targets:
    """
    return "perl ${MERGEIVOOVCASCADE} ${SOURCES[0]} ${SOURCES[1]} ${TARGETS[0]}"


def apply_rescaled_dt_pipe(target, source, env, for_signature):
    """Performs some rescaling.

    Sources:
    Targets:
    """
    return "python ${APPLYRESCALEDDTPIPE} ${SOURCES[0]} ${SOURCES[1]} ${SOURCES[2]} < ${SOURCES[3]} > ${TARGETS[0]} ${COMMAND_LINE_SUFFIX}"


def babel_scorer(target, source, env, for_signature):
    """Runs the F4DE scoring engine to calculate term-weighted values.

    Sources:
    Targets:
    """
    return "perl -X ${BABELSCORER} -e ${SOURCES[0]} -r ${SOURCES[1]} -t ${SOURCES[2]} -s ${SOURCES[3]} -c -o -b -d -a --ExcludePNGFileFromTxtTable -f ${'.'.join(TARGETS[0].rstr().split('.')[0:-2])} -y TXT ${COMMAND_LINE_SUFFIX}"


def run_kws(env, experiment_path, asr_output, vocabulary, pronunciations, keyword_file, *args, **kw):
    """
    This is a wrapper around all the other KWS builders to produce a full pipeline from ASR output (consensus networks),
    vocabulary, pronunciation dictionary, and keyword list.    

    Inputs: experiment output path, asr_output, vocabulary file, pronunciation file, keyword file
    Outputs: KWS output
    """
    env.Replace(EPSILON_SYMBOLS="'<s>,</s>,~SIL,<HES>'")

    if not env["RUN_KWS"]:
        return None
    to = 1 if env["DEBUG"] else env["JOB_COUNT"]
    
    pronunciations = env.AddWordBreaks(pjoin(experiment_path, "pronunciations.txt"), pronunciations)
    
    database_file = env.Glob('${DATABASE_FILE}')[0]
    ecf_file = env.Glob("${ECF_FILE}")[0]
    rttm_file = env.Glob("${RTTM_FILE}")[0]
    devinfo = env.File("${MODEL_PATH}/devinfo")
    p2p_file = env.File("${P2P_FILE}")
    dict_oov = env.Glob("${IBM_MODELS}/${BABEL_ID}/${PACK}/kws-resources/*/dict.OOV.v2p")
    dict_test = env.Glob("${IBM_MODELS}/${BABEL_ID}/${PACK}/kws-resources/*/dict.test.gz")
    expid = os.path.basename(ecf_file.rstr()).split("_")[0]

    iv_query_terms, oov_query_terms, word_to_word_fst = env.QueryFiles([pjoin(experiment_path, x) for x in ["iv_queries.txt", 
                                                                                                            "oov_queries.txt",
                                                                                                            "word_to_word.fst"]],
                                                                       [keyword_file, pronunciations])

    keyword_symbols = env.KeywordSymbols(pjoin(experiment_path, "keywords.sym"), [iv_query_terms, oov_query_terms])

    if kw.get("GRAPHEMIC", False):
        oov_pronunciations_nobreak = env.GraphemicOOVPronunciations(pjoin(experiment_path, "oov_pronunciations_nobreak.txt"), [pronunciations, oov_query_terms])
    else:
        oov_pronunciations_nobreak = env.OOVPronunciations(pjoin(experiment_path, "oov_pronunciations_nobreak.txt"), [pronunciations, oov_query_terms])

    oov_pronunciations = env.AddWordBreaks(pjoin(experiment_path, "oov_pronunciations.txt"), oov_pronunciations_nobreak)

    fst_header, phone_symbols, word_symbols, w2p_fsm, w2p_fst = env.WordsToPhones(
        [pjoin(experiment_path, x) for x in ["fst_header", "phones.sym", "words.sym", "words2phones.fsm", "words2phones.fst"]],
        [pronunciations, oov_pronunciations, dict_oov]
    )

    phone_symbols = env.AddPhone(pjoin(experiment_path, "p2p_workaround", "phones.sym"), [phone_symbols, env.Value(["u0071", "HES01", "HES02"])])

    p2p_fsm = env.PhonesToPhones(pjoin(experiment_path, "P2P.fsm"), [p2p_file, phone_symbols])

    p2p_unsorted = env.FSTCompile(pjoin(experiment_path, "P2P_unsorted.fst"), [phone_symbols, phone_symbols, p2p_fsm])

    p2p_fst = env.FSTArcSort(pjoin(experiment_path, "P2P.fst"), [p2p_unsorted, env.Value("ilabel")])
    
    wordpron = env.WordPronounceSymTable(pjoin(experiment_path, "in_vocabulary_symbol_table.txt"),
                                         pronunciations)
    
    vocabulary_symbols = env.CleanPronounceSymTable(pjoin(experiment_path, "cleaned_in_vocabulary_symbol_table.txt"),
                                                    wordpron)
    
    iv_queries = []
    
    for i, terms in list(enumerate(env.SplitList([pjoin(experiment_path, "iv_queries_%d_of_${JOB_COUNT}.txt" % (i + 1)) for i in range(env["JOB_COUNT"])], iv_query_terms)))[0:to]:
        iv_queries.append(env.CreateQueries(pjoin(experiment_path, "iv_query_fsts_%d_of_${JOB_COUNT}.tgz" % (i + 1)),
                                            [terms, word_symbols, p2p_fst, w2p_fst], NBESTP2P=env["NBESTP2P_IV"]))
    
    oov_queries = env.CreateQueries(pjoin(experiment_path, "oov_query_fsts.tgz"), [oov_query_terms, word_symbols, p2p_fst, w2p_fst], NBESTP2P=env["NBESTP2P_OOV"])
    
    ivs = []
    oovs = []
    
    for lattices in asr_output[0:to]:
        
        i, j = [int(x) for x in re.match(r".*?(\d+)_of_(\d+).*$", lattices.rstr()).groups()]

        index = env.MakeIndex(pjoin(experiment_path, "index_%d_of_%d.fsm" % (i, j)), lattices)
        
        sym, bsym = env.IndexToSymbolTables([pjoin(experiment_path, "index_%d_of_%d_sorted.%s" % (i, j, x)) for x in ["sym", "bsym"]],
                                            [index, keyword_symbols])
        
        expanded_index = env.ExpandPhoneIndex(pjoin(experiment_path, "expanded_index_%d_of_%d.fsm" % (i, j)),
                                              [index, dict_test])

        expanded_index_symbols, ebsym = env.IndexToSymbolTables([pjoin(experiment_path, "expanded_index_%d_of_%d_sorted.%s" % (i, j, x)) for x in ["sym", "bsym"]],
                                            [expanded_index])

        expanded_index_fst = env.FSTCompile(pjoin(experiment_path, "expanded_index_%d_of_%d.fst" % (i, j)),
                                            [phone_symbols, expanded_index_symbols, expanded_index])
        
        kwd_sym = env.CombineSymbolTables(pjoin(experiment_path, "index_kwd_%d_of_%d.sym" % (i, j)),
                                          [sym, word_symbols])

        sorted_expanded_index_fst = env.FSTArcSort(pjoin(experiment_path, "expanded_index_%d_of_%d_sorted.fst" % (i, j)),
                                                   [expanded_index_fst[0], env.Value("ilabel")])

        ivs.append(env.PerformSearch(pjoin(experiment_path, "IV_results", "result.%d_of_%d.txt" % (i, j)),
                                     [sorted_expanded_index_fst, phone_symbols, expanded_index_symbols, fst_header, iv_queries]))
        
        oovs.append(env.PerformSearch(pjoin(experiment_path, "OOV_results", "result.%d_of_%d.txt" % (i, j)),
                                      [sorted_expanded_index_fst, phone_symbols, expanded_index_symbols, fst_header, oov_queries]))
        
    iv_xml = env.Glob("${IBM_MODELS}/${BABEL_ID}/${PACK}/*-resources/kws-resources-IndusDB.*/template.iv.xml")[0]
    oov_xml = env.Glob("${IBM_MODELS}/${BABEL_ID}/${PACK}/*-resources/kws-resources-IndusDB.*/template.oov.xml")[0]

    iv_comb = env.CombineResults([pjoin(experiment_path, x) for x in ["iv_temp_1.txt", "iv_temp_2.txt", "iv.xml"]],
                                 [iv_xml] + ivs, NIST_EXPID_CORPUS=expid, LANGUAGE_TEXT=env.subst("${LANGUAGE_NAME}").replace("_", ""))
    oov_comb = env.CombineResults([pjoin(experiment_path, x) for x in ["oov_temp_1.txt", "oov_temp_2.txt", "oov.xml"]],
                                  [oov_xml] + oovs, NIST_EXPID_CORPUS=expid, LANGUAGE_TEXT=env.subst("${LANGUAGE_NAME}").replace("_", ""))

    iv_sto_norm = env.SumToOneNormalize(pjoin(experiment_path, "iv_norm.xml"), iv_comb[-1])
    oov_sto_norm = env.SumToOneNormalize(pjoin(experiment_path, "oov_norm.xml"), oov_comb[-1])

    merged = env.MergeIVOOVCascade(pjoin(experiment_path, "merged.xml"), [iv_sto_norm, oov_sto_norm])

    dt = env.ApplyRescaledDTPipe(pjoin(experiment_path, "dt.kwslist.xml"), [devinfo, database_file, ecf_file, merged])

    kws_score = env.BabelScorer([pjoin(experiment_path, "score.%s" % x) for x in ["alignment.csv", "bsum.txt", "sum.txt"]],
                                [ecf_file, rttm_file, keyword_file, dt])

    return merged


def run_cascade(env, experiment_path, word_space, morph_space, **kw):
    """Here is the big unimplemented part of the pipeline: cascaded search using both word, morph, and phone space.
    
    Like run_asr and run_kws, this is a method, not a builder, that sets up the experiment.  You'll have to figure 
    out what to do here by sifting through /vega/ccls/projects/babel_data/kws-NWAY.dev.tgz.  Here's the basic 
    sequence of scripts I found when I looked there:

      run_prepare_nway_cascade.sh
      run_search_cascade.sh
      run_postprocess_nway_cascade.sh

    You'll need to determine what these scripts do, and create builders that have the same behavior (either with
    new Python code in the builder, or by invoking the appropriate scripts).  Then, I think you can use these two 
    builders at the very end, just like basic KWS:

      env.ApplyRescaledDTPipe
      env.BabelScorer
    
    Inputs: experiment output path, KWS output for word space, KWS output for morph space
    Outputs: KWS output for cascade
    """
    return None


def TOOLS_ADD(env):
    """Conventional way to add builders and methods to an SCons environment."""
    env.Append(BUILDERS = {
        "AddWordBreaks" : Builder(action=add_word_breaks),
        "QueryFiles" : Builder(action=query_files),
        "KeywordSymbols" : Builder(generator=keyword_symbols),
        "OOVPronunciations" : Builder(action=oov_pronunciations),
        "GraphemicOOVPronunciations" : Builder(action=graphemic_oov_pronunciations),
        "WordsToPhones" : Builder(generator=words_to_phones),
        "AddPhone" : Builder(action=add_phone),
        "PhonesToPhones" : Builder(action=phones_to_phones),
        "WordPronounceSymTable" : Builder(action=word_pronounce_sym_table), 
        "CleanPronounceSymTable" : Builder(action=clean_pronounce_sym_table),
        "SplitList" : Builder(action=split_list),
        "CreateQueries" : Builder(action=create_queries),
        "MakeIndex" : Builder(action=make_index),
        "IndexToSymbolTables" : Builder(action=index_to_symbol_tables),
        "ExpandPhoneIndex" : Builder(action=expand_phone_index),
        "CombineSymbolTables" : Builder(action=combine_symbol_tables),
        "PerformSearch" : Builder(action=perform_search),
        "CombineResults" : Builder(generator=combine_results),
        "SumToOneNormalize" : Builder(generator=sum_to_one_normalize),
        "MergeIVOOVCascade" : Builder(generator=merge_iv_oov_cascade),
        "ApplyRescaledDTPipe" : Builder(generator=apply_rescaled_dt_pipe),
        "BabelScorer" : Builder(generator=babel_scorer),
    })
    env.AddMethod(run_kws, "RunKWS")
    env.AddMethod(run_cascade, "RunCascade")
