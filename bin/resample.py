from attila import *
import misc

from glob import glob
from os.path import basename, isdir, join
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="input")
parser.add_argument("-o", "--output", dest="output")
parser.add_argument("-a", "--antialias", dest="antialias")
options = parser.parse_args()

a48 = Audio()
a8  = Audio()
dec = Decimator()
dec.factor = 6
dec.readFilter(options.antialias)
a48.readWAV(options.input)
dec.compute(a48)
a8.copy(dec)
a8.writeWAV(options.output, 8000)
