#! /usr/bin/env python

import glob
import os

from fabricate import *


CC = 'gcc'
CCFLAGS = '-std=c99 -Wall -W -Wno-unused-parameter'
DEBUG_FLAGS   = '-O0 -g -DDEBUG'
RELEASE_FLAGS = '-O3'

sources = glob.glob('src/*.c')


# --- TARGETS ---

def build(mode='release'):
    compile(mode)
    make_library(mode)


def debug():
    build('debug')


def test():
    build('debug')
    compile_tests()
    execute_tests()


def clean():
    autoclean()


# --- RECIPES ---

def compile(mode='release'):
    extra_flags = RELEASE_FLAGS if mode == 'release' else \
                  DEBUG_FLAGS   if mode == 'debug'   else None
    fnames = [s[len('src/'):-len('.c')] for s in sources]
    for s in fnames:
        cmd = [CC]
        cmd.extend(CCFLAGS.split(' '))
        cmd.extend(extra_flags.split(' '))
        cmd.extend(['-c', 'src/' + s + '.c',
                    '-o', 'build/' + mode + '/' + s + '.o'])
        run(*cmd)


def compile_tests():
    for cfile in glob.glob('tests/test_*.c'):
        cmd = [CC]
        cmd.extend(CCFLAGS.split(' '))
        cmd.extend(DEBUG_FLAGS.split(' '))
        cmd.extend(['-Isrc', cfile, 'build/debug/librandomtrees.a',
                    '-o', strip_ext(cfile), '-lm'])
        run(*cmd)


def make_library(mode='release'):
    out_dir = 'build/' + mode + '/'
    obj = glob.glob(out_dir + '*.o')
    run('ar', 'rcs', out_dir + 'librandomtrees.a', *obj)


def execute_tests():
    print
    print "Running unit test:"
    for tmatch in glob.glob('tests/test_*'):
        test = os.path.basename(tmatch)
        if strip_ext(test) != test:
            continue
        shell('cd', 'tests', '&&', './runner.py', test, silent=False,
                                                        shell=True)


# --- utils ---

def strip_ext(s):
    return os.path.splitext(s)[0]


main()
