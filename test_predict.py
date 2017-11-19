import unittest
import predict


class TestStringMethods(unittest.TestCase):
	def test_convert_wait_time_to_days(self):
		tests = [
			["2 Yrs 7 Mths 16 Days", 2 * 365 + 7 * 30 + 16]
		]
		for t in tests:
			assert(predict.convert_wait_time(t[0], month=False) == t[1])

if __name__ == '__main__':
	unittest.main()
