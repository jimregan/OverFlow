import json


s = {}

for line in open('data/trimmed-mtm-fem/filelists/swe-fem-train.txt').readlines():
    syms = line.split('|')[1].split()
    for sym in syms:
        s[sym] = True
allsyms = list(s.keys())
allsyms.sort()
print('all symbols:')
print(allsyms)
json.dump({'symbols':allsyms},open('mtm_symbols.json','w'))