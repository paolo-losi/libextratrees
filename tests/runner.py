#! /usr/bin/env python

import os
import sys
import subprocess
import difflib


def main():
    test = sys.argv[1]
    sp = subprocess.Popen(['./%s' % test], shell=True,
                                           stderr=subprocess.PIPE)
    test_output = sp.stderr.read()

    reference_fname = '%s.out' % test

    if not os.path.exists(reference_fname):
        fout = open(reference_fname, 'w')
        fout.write(test_output)
        fout.close()
        print '%20s: NEW' % test
        return

    reference_output = open(reference_fname).read()

    reference_output = cleanup_output(reference_output)
    test_output      = cleanup_output(test_output)

    if reference_output != test_output:
        differ = difflib.Differ()
        diff_out = differ.compare(reference_output, test_output)

        print "%20s: FAILED" % test
        print
        print "----------- test reference diff -----------"
        sys.stdout.writelines(diff_out)
        print "-------------------------------------------"
        print
    else:
        print "%20s: ok" % test


def cleanup_output(output):
    ret = []
    for l in output.splitlines(1):
        if l.startswith('[') and l[23] == ':':
            l = l[:23] + l[27:]
        ret.append(l)
    return ret


if __name__ == '__main__':
    main()
