from .math_equivalence import is_equiv
import unittest


class TestIsEquiv(unittest.TestCase):

	def test_fractions(self):
	    test_in = "\\tfrac{1}{2} + \\frac1{72}"
	    test_out = "\\\\frac{1}{2} + 2/3"
	    self.assertFalse(is_equiv(test_in, test_out))

	def test_order(self):
		test_in = "10, 4, -2"
		test_out = "4, 10, -2"
		self.assertFalse(is_equiv(test_in, test_out))

	def test_order2(self):
		test_in = "10, 4, 2"
		test_out = "4, 12, 2"
		self.assertFalse(is_equiv(test_in, test_out))

	def test_dfrac(self):
		test_in = "\\tfrac{1}{2} +\\! \\frac1{72}"
		test_out = "\\\\dfrac{1}{2} +\\frac{1}{72}"
		self.assertTrue(is_equiv(test_in, test_out))

	def test_units(self):
		test_in = "10\\text{ units}"
		test_out = "10 "
		self.assertTrue(is_equiv(test_in, test_out))

	def test_units2(self):
		test_in = "10\\text{ units}"
		test_out = "100 "
		self.assertFalse(is_equiv(test_in, test_out))

	def test_dollars(self):
		test_in = "10"
		test_out = "\\$10"
		self.assertTrue(is_equiv(test_in, test_out))

	def test_parentheses(self):
		test_in = "\\left(x-2\\right)\\left(x+2\\right)"
		test_out = "(x-2)(x+2)"
		self.assertTrue(is_equiv(test_in, test_out))

	def test_decimals(self):
		test_in = "0.1, 4, 2"
		test_out = "4, .1, 2"
		self.assertFalse(is_equiv(test_in, test_out))

	def test_decimals2(self):
		test_in = "0.1"
		test_out = ".1"
		self.assertTrue(is_equiv(test_in, test_out))

	def test_percentage(self):
		test_in = "10\\%"
		test_out = "10"
		self.assertTrue(is_equiv(test_in, test_out))

	def test_sqrt(self):
		test_in = "10\\sqrt{3} + \\sqrt4"
		test_out = "10\\sqrt3 + \\sqrt{4}"
		self.assertTrue(is_equiv(test_in, test_out))

	def test_frac(self):
		test_in = "\\frac34i"
		test_out = "\\frac{3}{4}i"
		self.assertTrue(is_equiv(test_in, test_out))

	def test_tfrac(self):
		test_in = "\\tfrac83"
		test_out = "\\frac{8}{3}"
		self.assertTrue(is_equiv(test_in, test_out))

	def test_expression(self):
		test_in = "5x - 7y + 11z + 4 = 0"
		test_out = "x + y - z + 2 = 0"
		self.assertFalse(is_equiv(test_in, test_out))

	def test_half(self):
		test_in = "1/2"
		test_out = "\\frac{1}{2}"
		self.assertTrue(is_equiv(test_in, test_out))