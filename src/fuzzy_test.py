from fuzzywuzzy import fuzz
from fuzzywuzzy import process

choices = []
with open("corpora/transmir/transmir_mirnas.txt", 'r') as t:
    for l in t:
        choices.append(l.strip())


for c in choices:
    alt = c.upper().replace("-", "")
    match = process.extract(alt, choices, limit=3)
    if c != match[0][0]:
        print(c, alt, match)
    elif match[0][1] < 85:
        print(c, alt, match)