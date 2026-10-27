"""Regression checks for content that previously vanished before tokenization."""
import unittest
from tts_text_preprocessing import expand_numbers_and_money as expand

class NumberPreprocessingTests(unittest.TestCase):
    def test_integer_decimal_and_sign(self):
        for raw,expected in [('12','twelve'),('3','three'),('0','zero'),('101','one hundred one'),
            ('1,234','one thousand two hundred thirty four'),('-12','minus twelve'),
            ('3.05','three point zero five'),('007','zero zero seven')]:
            with self.subTest(raw=raw):self.assertEqual(expand(raw),expected)
    def test_us_dollars(self):
        for raw,expected in [('$12.50','twelve dollars and fifty cents'),('$1.01','one dollar and one cent'),
            ('$0.50','fifty cents'),('$0.01','one cent'),('$0.00','zero dollars'),
            ('$1','one dollar'),('$12.5','twelve dollars and fifty cents'),
            ('$1,234.56','one thousand two hundred thirty four dollars and fifty six cents'),
            ('-$2.00','minus two dollars')]:
            with self.subTest(raw=raw):self.assertEqual(expand(raw),expected)
    def test_real_prompts_and_no_extra_changes(self):
        self.assertEqual(expand('There are 12 books on the shelf and 3 pencils on the desk.'),
            'There are twelve books on the shelf and three pencils on the desk.')
        self.assertEqual(expand('The ticket costs $12.50, including the booking fee.'),
            'The ticket costs twelve dollars and fifty cents, including the booking fee.')
        self.assertEqual(expand('Dr. Smith and Mrs. Brown arrived.'),'Dr. Smith and Mrs. Brown arrived.')
        text='The ticket costs $12.50.'
        self.assertEqual(expand(expand(text)),expand(text))
    def test_unsupported_fails_explicitly(self):
        for raw in ['$1.234','7:30','21st','2026/10/26','1e3','.5','$','£12','12AB','12,34']:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):expand(raw)

if __name__=='__main__':unittest.main()
