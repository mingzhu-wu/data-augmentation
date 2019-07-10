import sys
import gzip
import json
import os


os.makedirs("contexts")
id = 0
with gzip.open(sys.argv[1], 'rt') as f_in:
    next(f_in)
    for line in f_in:
        context = json.loads(line)["context"]
        id += 1
        with open("contexts/context_"+str(id)+"txt", 'w') as f_out:
            json.dump({id: context}, f_out)
