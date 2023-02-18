import re

# The `\b` characters mean the string needs to be between word boundaries
# so we won't match words like `history` or `tread`
regex = re.compile(r"\bread\b|\bstory\b|book")

phrases = [
    "I liked that story.",
    "the car treaded up the hill",
    "this hat is nice",
]

matches = []
for phrase in phrases:
    if re.search(regex, phrase):
        matches.append(phrase)

print(matches)
