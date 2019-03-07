#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 修改后的代码如下：
import logging
import os.path
import sys
from gensim.corpora import WikiCorpus
if __name__ == '__main__':
    
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # check and process input arguments
    if len(sys.argv) < 3:
        print (globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    space = b' '
    i = 0
    output = open(outp, 'w',encoding='utf-8')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        s=space.join(text)
        s=s.decode('utf8') + "\n"
        output.write(s)
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")
    output.close()
    logger.info("Finished Saved " + str(i) + " articles")
#python process.py zhwiki-latest-pages-articles.xml.bz2 wiki.zh.text