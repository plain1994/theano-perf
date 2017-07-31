#!/usr/bin/env python
from __future__ import print_function
import os
import sys, timeit, time
import commands

import numpy

import theano, theano.tensor.signal.conv
from theano import tensor


def print_help(exit_status):
    if exit_status:
        print('command "%s" not recognized' % (' '.join(sys.argv)))
    print('Type "theano-perf" to print this help')  
    print('Type "theano-perf help" to print this help')


    print('Type "theano-perf gem" to test gem')
    print('Type "theano-perf mem" to test memory bandwidth')
    print('Type "theano-perf disk" to test disk write and read speed')
    print('Type "theano-perf op" to test benchmark of op')
    
    sys.exit(exit_status)

if len(sys.argv) == 1:
    print_help(exit_status=0)
    
elif len(sys.argv) == 2:
    if sys.argv[1] == 'help':
        print_help(exit_status=0)

    if sys.argv[1] == 'gem':
        (status, output) = commands.getstatusoutput("lscpu | grep -i 'Model name'")
        if (status == 0):
            print ("CPU model: %s" %(output.split(':')[1].strip()))
        (status, output) = commands.getstatusoutput("lscpu | grep -i 'CPU(s)'")
        if (status == 0):
            print ("CPU cores: %s" %(output.split('\n')[0].split(':')[1].strip()))
        (status, output) = commands.getstatusoutput("lscpu | grep -i 'CPU MHz'")
        if (status == 0):
            print ("CPU MHz: %s \n" %(output.split(':')[1].strip()))


        status = os.system('g++ dgemm_with_timing.c -o a.out -lmkl_rt')
        if (status == 0):
            print("Compiled gem test code.\n")
            print("This code measures performance of Intel(R) MKL function dgemm,",
            "computing real matrix C=alpha*A*B+beta*C, where A, B, and C",
            "are matrices and alpha and beta are double precision scalars.\n")
        status = os.system('bash run_gemm.sh')
        if (status == 0):
            print("Ran gem test code.\n")

    elif sys.argv[1] == 'mem':
        (status, output) = commands.getstatusoutput("icpc -mcmodel=medium -O3 -qopenmp bw.cpp gettime.cpp -o bw.out -liomp5 ")
        if (status == 0):
            print("Compiled memeroy test code.\n")
            status = os.system("./bw.out")
            if (status == 0):
                print("Ran memory test code.\n")


    elif sys.argv[1] == 'disk':
        (status, output) = commands.getstatusoutput('dd if=/dev/zero of=ddfile bs=8k count=250000')
        if (status == 0):
            #print output
            print("Disk write test done, write speed: %s\n" %(output.split('\n')[2].split(',')[2].strip()))
            (status2, output) = commands.getstatusoutput('dd if=ddfile of=/dev/null bs=8k count=250000')
            if (status2 == 0):
                print("Disk read test done, read speed: %s\n" %(output.split('\n')[2].split(',')[2].strip()))
            (status, output) = commands.getstatusoutput('rm -rf ./ddfile')

    elif sys.argv[1] == 'op':
        x = tensor.fmatrix('x')
        y = tensor.fmatrix('y')
        z = theano.tensor.signal.conv.conv2d(x, y)
        f = theano.function([x, y], z)
        a = numpy.random.rand(500, 500).astype(numpy.float32)
        b = numpy.random.rand(10, 10).astype(numpy.float32)
        start_time = timeit.default_timer()
        for i in range(500):
            o = f(a, b)
        end_time = timeit.default_timer()
        #print (o)
        print ('Conv image size(500, 500), filter size(10, 10) ran 500 epoches for %is' % ((end_time - start_time)))

    else:
        print_help(exit_status=1)

elif len(sys.argv) == 3 and sys.argv[1] == 'op':
    if sys.argv[2] == 'conv':
        x = tensor.fmatrix('x')
        y = tensor.fmatrix('y')
        z = theano.tensor.signal.conv.conv2d(x, y)
        f = theano.function([x, y], z)
        a = numpy.random.rand(500, 500).astype(numpy.float32)
        b = numpy.random.rand(10, 10).astype(numpy.float32)
        start_time = timeit.default_timer()
        for i in range(500):
            o = f(a, b)
        end_time = timeit.default_timer()
        #print (o)
        print ('Conv image size(500, 500), filter size(10, 10) ran 500 epoches for %is' % ((end_time - start_time)))

    elif sys.argv[2] == 'relu':
        print("TBD")
    else:
        print_help(exit_status=1)
else:
    print_help(exit_status=1)
