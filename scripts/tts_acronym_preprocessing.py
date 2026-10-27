"""Optional uppercase dotted initialisms via ONE A-Z letter-name table.
Default policy: preserve abbreviations; always repair supported numbers/money.
Only character inputs; does not assume ARPAbet embeddings were trained.
Lowercase e.g./i.e., titles, decimals, domains and undotted acronyms are not guessed.
"""
import re
from tts_text_preprocessing import expand_numbers_and_money

LETTER_NAMES=dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',[
    'ay','bee','see','dee','ee','ef','jee','aitch','eye','jay','kay','el','em',
    'en','oh','pee','cue','ar','ess','tee','you','vee','double you','ex','why','zee']))
DOTTED_INITIALISM=re.compile(r'(?<![A-Za-z0-9_.])(?:[A-Z]\.)+[A-Z]\.?(?![A-Za-z0-9_]|\.[A-Za-z0-9])')

def expand_dotted_initialisms(text):
    def expand(match):
        token=match.group(0)
        letters=token.replace('.','')
        return ', '.join(LETTER_NAMES[letter] for letter in letters)+('.' if token.endswith('.') else '')
    return DOTTED_INITIALISM.sub(expand,text)

def prepare_acronym_text(text, *, expand_initialisms=False):
    """Always repair numbers/money; preserve abbreviations unless explicitly enabled.

    Example: prepare_acronym_text(text, expand_initialisms=True) is an experimental
    opt-in, not a guaranteed pronunciation improvement (U.N. has a human regression).
    """
    if expand_initialisms:
        text=expand_dotted_initialisms(text)
    return expand_numbers_and_money(text)
