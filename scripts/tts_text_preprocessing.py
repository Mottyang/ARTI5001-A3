"""English cardinal numbers, decimals and US dollar amounts for TTS input.

Expand before SpeechBrain english_cleaners/tokenization, which otherwise drops digits.
No package monkey-patching or model changes. This is an explicit inference front end;
ambiguous dates/times, ordinals, scientific notation and other currencies are out of scope.
"""
import re

ONES = ('zero one two three four five six seven eight nine ten eleven twelve '
        'thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split()
TENS = 'zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split()
SCALES = [(10**12,'trillion'),(10**9,'billion'),(10**6,'million'),(1000,'thousand')]
NUMBER = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
MONEY_RE = re.compile(r'(?<![\w$])(?P<sign>-?)\$(?P<amount>'+NUMBER+r')(?!\w|,\d|\.\d)')
NUMBER_RE = re.compile(r'(?<![\w.])(?P<sign>-?)(?P<number>'+NUMBER+r')(?![\w]|\.\d)')

def integer_words(number):
    number=int(number)
    if number<0:return 'minus '+integer_words(-number)
    if number>=10**15:raise ValueError('Numbers of a quadrillion or greater are unsupported')
    if number<20:return ONES[number]
    if number<100:return TENS[number//10]+(' '+ONES[number%10] if number%10 else '')
    if number<1000:return ONES[number//100]+' hundred'+(' '+integer_words(number%100) if number%100 else '')
    for scale,label in SCALES:
        if number>=scale:
            return integer_words(number//scale)+' '+label+(' '+integer_words(number%scale) if number%scale else '')

def _money(match):
    value=match['amount'].replace(',','')
    whole,_,fraction=value.partition('.')
    if len(fraction)>2:raise ValueError('Dollar amounts must have at most two decimal places')
    dollars=int(whole);cents=int(fraction.ljust(2,'0') or '0')
    components=[]
    if dollars or not cents:components.append(integer_words(dollars)+(' dollar' if dollars==1 else ' dollars'))
    if cents:components.append(integer_words(cents)+(' cent' if cents==1 else ' cents'))
    return ('minus ' if match['sign'] else '')+' and '.join(components)

def _number(match):
    value=match['number'].replace(',','')
    whole,point,fraction=value.partition('.')
    # Leading-zero strings are read digit by digit (e.g. 007), not as seven.
    words=' '.join(ONES[int(c)] for c in whole) if len(whole)>1 and whole.startswith('0') else integer_words(whole)
    if point:words+=' point '+' '.join(ONES[int(c)] for c in fraction)
    return ('minus ' if match['sign'] else '')+words

def expand_numbers_and_money(text):
    """Return text with numeric content spoken explicitly; preserve other text.

    Reject unsupported leftover numeric content rather than silently losing it at
    character tokenization. This function intentionally does not guess dates/times.
    """
    if re.search(r'\d:\d|\d/\d|\d(?:st|nd|rd|th)\b|\d[eE][+-]?\d',text):
        raise ValueError('Write times, dates, ordinals or scientific notation in spoken words')
    if re.search(r'(?<!\w)\.\d',text):
        raise ValueError('Use an explicit leading zero for decimals, e.g. 0.5')
    if re.search(r'(?<!\d)\d{1,2},\d{1,2}(?!\d)',text):
        raise ValueError('Invalid thousands grouping')
    text=MONEY_RE.sub(_money,text)
    if '$' in text:raise ValueError('Unsupported dollar amount; use $12.50 or write spoken words')
    if re.search(r'[£€¥]',text):raise ValueError('Write non-USD currency amounts in spoken words')
    text=NUMBER_RE.sub(_number,text)
    if re.search(r'\d',text):raise ValueError('Unsupported numeric token; write it in spoken words')
    return text
