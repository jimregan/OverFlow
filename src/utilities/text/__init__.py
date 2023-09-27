"""from https://github.com/keithito/tacotron."""
import re

from src.utilities.text import cleaners
from src.utilities.text.symbols import symbols

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


'''
    Converts a string of space-separated binary feature vectors into a tensor
    v1
    example input "0001 1001 & 0010 1111 ."
    v2
    example input "0,0,0,0.5 0,0,1,0.75 ..."
'''
def feat_to_sequence(feattext, featureset = 'v2'):
    tokens = feattext.split()
    tokens.append('.')
    seq = []

    if featureset == 'v1':
        # 36 binary parameters (34 + wordfinal, sentfinal)

        for token,nexttoken in zip(tokens[:-1],tokens[1:]):
            sentfinal = int(nexttoken=='.')
            wordfinal = int(nexttoken=='&' or sentfinal)
            #print(token,nexttoken)
            if len(token)>1:
                #print(token,sentfinal,wordfinal)
                ntoken = [int(x) for x in list(token)] + [int(wordfinal), int(sentfinal)]
                seq.append(ntoken)

    elif featureset == 'v2':
        # 33 binary & continous params (31 + wfinal, sfinal), comma separated string
        for token,nexttoken in zip(tokens[:-1],tokens[1:]):
            sentfinal = int(nexttoken=='.')
            wordfinal = int(nexttoken=='&' or sentfinal)
            #print(token,nexttoken)
            if len(token)>1:
                #print(token,sentfinal,wordfinal)
                ntoken = [0.0 if x=='' else float(x) for x in token.split(',')] + [float(wordfinal), float(sentfinal)]
                seq.append(ntoken)

    return seq
 

def text_to_sequence(text, cleaner_names):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []
    #print('text = ', text)
    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        # re.match searches only for the beginning of the string, so it keeps on returning in groups
        # so each pair of brackets will make this loop iterate
        m = _curly_re.match(text)
        
        if not m:
            # clean text function preprocesses text converts to lowercase and stuff
            sequence += _symbols_to_sequence(clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    #print('seq = ', sequence)
    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string."""
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")


def clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != "_" and s != "~"


# Custom Functions


def phonetise_text(cmu_phonetiser, text, word_tokenize):
    text = word_tokenize(text)
    text = " ".join(
        ["{" + cmu_phonetiser.lookup(word)[0] + "}" if cmu_phonetiser.lookup(word) else word for word in text]
    )
    return text
