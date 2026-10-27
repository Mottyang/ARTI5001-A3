import unittest
from tts_acronym_preprocessing import prepare_acronym_text as prepare, expand_dotted_initialisms as expand, LETTER_NAMES

class DottedInitialismsTests(unittest.TestCase):
    def test_letter_table_is_complete(self):
        self.assertEqual(set(LETTER_NAMES),set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    def test_unlisted_combinations(self):
        for text,expected in [('N.R.A.','en, ar, ay.'),('U.S.A.','you, ess, ay.'),
            ('F.B.I.','ef, bee, eye.'),('C.P.U.','see, pee, you.'),('U.N.','you, en.'),
            ('B.B.C.','bee, bee, see.'),('E.U.','ee, you.'),('X.Y.Z.','ex, why, zee.'),
            ('A.B','ay, bee'),('W.H.O.','double you, aitch, oh.')]:
            with self.subTest(text=text):self.assertEqual(expand(text),expected)
    def test_context_and_idempotence(self):
        text='The U.S.A. and U.N. have 12 reports.'
        fixed=prepare(text, expand_initialisms=True)
        self.assertEqual(fixed,'The you, ess, ay. and you, en. have twelve reports.')
        self.assertEqual(prepare(fixed, expand_initialisms=True),fixed)
    def test_no_false_expansions(self):
        for text in ['Dr. Smith','NASA and CPU','e.g. and i.e.','3.14','example.co.uk',
                     'n.r.a.','ABC.U.S.A.COM','A.','N.R.A.com']:
            with self.subTest(text=text):self.assertEqual(expand(text),text)

    def test_default_preserves_acronyms_but_repairs_numbers_money(self):
        self.assertEqual(prepare('U.N. has 12 reports costing $12.50.'),
            'U.N. has twelve reports costing twelve dollars and fifty cents.')
        for raw in ['N.R.A.', 'U.S.A.', 'U.N.', 'NASA', 'CPU', 'Dr. Smith']:
            with self.subTest(raw=raw):self.assertEqual(prepare(raw),raw)
    def test_explicit_opt_in(self):
        self.assertEqual(prepare('U.N.', expand_initialisms=True),'you, en.')
        self.assertEqual(prepare('N.R.A.', expand_initialisms=True),'en, ar, ay.')
        self.assertEqual(prepare('U.N. has 3 reports.', expand_initialisms=False),'U.N. has three reports.')

if __name__=='__main__':unittest.main()
