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

    if reference_output != test_output:
        differ = difflib.Differ()
        diff_out = differ.compare(reference_output.splitlines(1),
                                  test_output.splitlines(1))

        print "%20s: FAILED" % test
        print
        print "----------- test reference diff -----------"
        sys.stdout.writelines(diff_out)
        print "-------------------------------------------"
        print
    else:
        print "%20s: ok" % test


if __name__ == '__main__':
    main()
