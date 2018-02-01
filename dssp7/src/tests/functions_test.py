# -*- coding: UTF-8 -*-

# Import from standard library
import unittest
import datetime

# Import from our lib
from dssp7.annex.functions import date_datetime_to_str, compare


class TestUtils(unittest.TestCase):
    def test_functions(self):
        """Test function."""
        self.assertTrue(compare('bonjour', 'bon jou*r'))
        test_dt = datetime.datetime(2017, 07, 14, 23, 59)
        self.assertEqual(date_datetime_to_str(test_dt), "2017_07_14_23_59")


if __name__ == '__main__':
    unittest.main()
