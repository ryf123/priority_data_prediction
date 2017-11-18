from predict import convert_wait_time_to_days
def test_convert_wait_time_to_days():
	tests = [
		["2 Yrs 7 Mths 16 Days", 2 * 365 + 7 * 30 + 16]
	]
	for t in tests:
		assert(convert_wait_time_to_days(t[0]) == t[1])
		
test_convert_wait_time_to_days()