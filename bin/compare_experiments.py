import argparse
import difflib
from pprint import pprint
from common_tools import meta_open
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--ibm_asr", dest="ibm_asr", help="path to IBM ASR experiment")
parser.add_argument("--ibm_kws", dest="ibm_kws", help="path to IBM KWS experiment")
parser.add_argument("--scons_asr", dest="scons_asr", help="path to SCons ASR experiment output")
parser.add_argument("--scons_kws", dest="scons_kws", help="path to SCons KWS experiment output")
parser.add_argument("-o", "--output", dest="output", help="output file")
options = parser.parse_args()

args = {
    "IBM_ASR" : options.ibm_asr,
    "IBM_KWS" : options.ibm_kws,
    "SCONS_ASR" : options.scons_asr,
    "SCONS_KWS" : options.scons_kws,
    }

to_compare = [
    ("%(IBM_KWS)s/data/queryterms.iv", "%(SCONS_KWS)s/iv_queries.txt"),
    ("%(IBM_KWS)s/data/queryterms.oov", "%(SCONS_KWS)s/oov_queries.txt"),
    ("%(IBM_KWS)s/data/OFST/fst_header", "%(SCONS_KWS)s/fst_header"),
    ("%(IBM_KWS)s/data/OFST/P2P.fsm", "%(SCONS_KWS)s/P2P.fsm"),
    ("%(IBM_KWS)s/data/OFST/keywords.sym", "%(SCONS_KWS)s/keywords.sym"),
    ("%(IBM_KWS)s/data/OFST/phones.sym", "%(SCONS_KWS)s/phones.sym"),
    ("%(IBM_KWS)s/data/OFST/words.sym", "%(SCONS_KWS)s/words.sym"),
    ("%(IBM_KWS)s/data/OFST/words2phones.fsm", "%(SCONS_KWS)s/words2phones.fsm"),
    ]

for comparison in to_compare:
    try:
        ibm, scons, comp = comparison
    except:
        ibm, scons = comparison
        comp = "diff"
    ibm = ibm % args
    scons = scons % args
    print "comparing %s to %s" % (ibm, scons)
    with meta_open(ibm) as ifdA, meta_open(scons) as ifdB:
        d = difflib.Differ()
        linesA = sorted(ifdA.read().splitlines(1))
        linesB = sorted(ifdB.read().splitlines(1))
        r = [l for l in d.compare(linesA, linesB) if not l.startswith(" ")]
        print len(r)
        #print r
