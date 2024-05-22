from assignment2 import OrfFinder

genome = OrfFinder('AAABBAAABBAAA')
assert sorted(genome.find('AA', 'AB')) == sorted(['AAABBAAAB', 'AAAB', 'AABBAAAB', 'AAAB']), sorted(genome.find('AA', 'AB'))

genome = OrfFinder('DCDCADCDC')
assert sorted(genome.find('DC', 'AD')) == sorted(['DCDCAD', 'DCAD'])

genome = OrfFinder("A")
assert genome.find('A', 'A') == []

genome = OrfFinder('DDDAAADDD')
assert (sorted(genome.find('D', 'A')) == sorted(['DDDAAA', 'DDDAA', 'DDDA', 'DDAAA', 'DDAA', 'DDA', 'DAAA', 'DAA', 'DA']))

genome = OrfFinder('DDDAAADDD')
assert (sorted(genome.find('D', 'AA')) == sorted(['DDDAAA', 'DDDAA', 'DDAAA', 'DDAA', 'DAAA', 'DAA']))

genome = OrfFinder('DDDAAADDD')
assert (sorted(genome.find('DD', 'AA')) == sorted(['DDDAAA', 'DDDAA', 'DDAAA', 'DDAA']))

genome = OrfFinder('DDDAAADDD')
assert (sorted(genome.find('DD', 'DD')) == sorted(['DDDAAADDD', 'DDDAAADD', 'DDAAADDD', 'DDAAADD']))

genome = OrfFinder('DDDAAADDD')
assert (sorted(genome.find('D', 'D')) == sorted(['DDDAAADDD', 'DDDAAADD', 'DDDAAAD', 'DDD', 'DD', 'DDAAADDD', 'DDAAADD', 'DDAAAD', 'DD', 'DAAADDD', 'DAAADD', 'DAAAD', 'DDD', 'DD', 'DD']))

genome = OrfFinder("AA")
assert(genome.find("AA", "A") == [])

genome = OrfFinder("BACACA")
assert sorted(genome.find("B", "A")) == sorted(["BACA", "BA", "BACACA"])

genome = OrfFinder("AAAAAAA")
expected_res = ["A"*2]*6 + ["A"*3]*5 + ["A"*4]*4 + ["A"*5]*3 + ["A"*6]*2 + ["A"*7]*1
assert(sorted(genome.find("A", "A")) == sorted(expected_res))





def compare(l1, l2):
    return sorted(l1) == sorted(l2)

genome1 = OrfFinder("ABCABC")

assert compare(genome1.find("A", "C"), ['ABC', 'ABC', 'ABCABC'])
assert compare(genome1.find("A", "B"), ['AB', 'AB', 'ABCAB'])
assert compare(genome1.find("B", "C"), ['BC', 'BC', 'BCABC'])
assert compare(genome1.find("C", "A"), ['CA'])
assert compare(genome1.find("AB", "C"), ['ABC', 'ABC', 'ABCABC'])

# start and end should not overlap
assert compare(genome1.find("C", "C"), ['CABC'])
assert compare(genome1.find("ABCABC", "ABCABC"), [])

genome1 = OrfFinder("AAA")

assert compare(genome1.find("A", "A"), ['AA', 'AA', 'AAA'])




genome = OrfFinder("AAABBBCCC")
lst = genome.find("B","C")
expected = ["BBBCCC", "BBBCC", "BBCCC", "BC", "BCC", "BBC", "BCCC", "BBBC", "BBCC"]
for i in lst:
    assert i in expected