from assignment2 import OrfFinder
#idk :3
genome = OrfFinder('BCBCACCCB')
assert sorted(genome.find('B', 'A')) == sorted(['BCA', 'BCBCA'])
genome = OrfFinder('BADCABDADB')
assert sorted(genome.find('D', 'C')) == sorted(['DC'])
genome = OrfFinder('CCDDBBACABBBBBCDC')
assert sorted(genome.find('B', 'A')) == sorted(['BA', 'BACA', 'BBA', 'BBACA'])
genome = OrfFinder('ACBBCCCBBAABDCCBDDAC')
assert sorted(genome.find('D', 'C')) == sorted(['DAC', 'DC', 'DCC', 'DCCBDDAC', 'DDAC'])
genome = OrfFinder("AA")
assert(genome.find("AA", "A") == [])
genome1 = OrfFinder("ABCABC")
print(genome1.find("A", "C"))
assert sorted(genome1.find('A', 'C')) == sorted(['ABC', 'ABC', 'ABCABC'])
genome = OrfFinder("AAAAAAA")
expected_res = ["A"*2]*6 + ["A"*3]*5 + ["A"*4]*4 + ["A"*5]*3 + ["A"*6]*2 + ["A"*7]*1
assert(sorted(genome.find("A", "A")) == sorted(expected_res))
print("Initial edge case tests passed!")

def test_given_cases():
    #2004 document given
    genome1 = OrfFinder("AAABBBCCC")
    try:
        result = genome1.find("AAA", "BB")
        assert sorted(result) == sorted(["AAABB", "AAABBB"]), f"Test failed for 'AAA', 'BB': {result}"
        print("Test for 'AAA', 'BB' passed.")
    except AssertionError as e:
        print(e)

    try:
        result = genome1.find("BB", "A")
        assert result == [], "Test failed for 'BB', 'A'"
        print("Test for 'BB', 'A' passed.")
    except AssertionError as e:
        print(e)

    try:
        result = genome1.find("AA", "BC")
        assert sorted(result) == sorted(["AABBBC", "AAABBBC"]), f"Test failed for 'AA', 'BC': {result}"
        print("Test for 'AA', 'BC' passed.")
    except AssertionError as e:
        print(e)

    try:
        result = genome1.find("A", "B")
        assert sorted(result) == sorted(["AAAB", "AAABB", "AAABBB", "AAB", "AABB", "AABBB", "AB", "ABB", "ABBB"]), f"Test failed for 'A', 'B': {result}"
        print("Test for 'A', 'B' passed.")
    except AssertionError as e:
        print(e)

    try:
        result = genome1.find("AA", "A")
        assert result == ["AAA"], f"Test failed for 'AA', 'A': {result}"
        print("Test for 'AA', 'A' passed.")
    except AssertionError as e:
        print(e)

    try:
        result = genome1.find("AAAB", "BBB")
        assert result == [], "Test failed for 'AAAB', 'BBB'"
        print("Test for 'AAAB', 'BBB' passed.")
    except AssertionError as e:
        print(e)

    #strangecase
    genome2 = OrfFinder("ABCDCBCBCCA")
    expected_output = ['ABCDCBCBCCA']
    try:
        result = genome2.find('A', 'A')
        assert sorted(result) == sorted(expected_output), f"Test failed for 'A', 'A': {result}"
        print("Test for 'A', 'A' passed.")
    except AssertionError as e:
        print(e)

    #weird edge maybe?
    genome3 = OrfFinder('CACACBDACCBBBBD')
    expected_output = ['CAC', 'CAC', 'CACAC', 'CACACBDAC', 'CACACBDACC', 'CACBDAC', 'CACBDACC', 'CBDAC', 'CBDACC', 'CC']
    try:
        result = genome3.find('C', 'C')
        assert sorted(result) == sorted(expected_output), f"Test failed for 'C', 'C': {result}"
        print("Test for 'C', 'C' passed.")
    except AssertionError as e:
        print(e)

test_given_cases()