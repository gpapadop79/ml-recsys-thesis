# -*- coding: utf-8 -*-
"""
Created on Mon Nov 09 21:38:18 2015

@author: Paul McGuire
"""
# http://stackoverflow.com/questions/1557571/how-to-get-time-of-a-python-program-execution/1557906#1557906
# I put this timing.py module into my own site-packages directory, and just insert import timing at the top of my module
# I can also call timing.log from within my program if there are significant stages within the program I want to show. 
# But just including import timing will print the start and end times, and overall elapsed time.

import atexit
from datetime import datetime

import timeit

def secondsToStr(t):
    return "%dh:%02dm:%02d.%03d sec" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60])

line = "="*55
def log(s, elapsed=None):
    print line
    #print secondsToStr(clock()), '-', s
    print datetime.now().strftime('%d-%m-%Y %H:%M:%S'), '-', s
    if elapsed:
        print "Elapsed time:", elapsed
    print line
    print

def endlog(s="End Program"):
    global start
    #end = clock()
    end = timeit.default_timer()
    elapsed = end-start
    log(s, secondsToStr(elapsed))
    return elapsed

def now():
    return secondsToStr(clock())

def startlog(s="Start Program"):    
    global start    
    #start = clock()
    start = timeit.default_timer()
    log(s)
    
def tic():
    global start
    start = timeit.default_timer()
    
def toc():
     global start
     return secondsToStr(timeit.default_timer()-start)

#start=0
