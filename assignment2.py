from dataclasses import dataclass
from typing import Any
inf = float('inf')


#Complexity notes:
#   C is the number of characters in the given alphabet; per the specs. C = 4
#   k be the complexity of the key function that turns the characters into indicies
#   Assume space complexity is aux space unless otherwise specified
#   All complexity analysis is based on worst case scenarios


# QUESTION 1: Open Reading Frames


class SLPrefixNode:
    """
    Class Description:
    Class for nodes of a 'string list' prefix trie.
        ie. a prefix trie that stores a list of included strings
    Each node stores:
        self.nodes | an array of child nodes
        self.substrings | an array of strings that pass through this node
    """
    
    def __init__(self, characters : int) -> None:
        """
        Function description:
            Sets up the prefix trie node given a character length
        :Input:
            :characters: The size of the alphabet to be used in the prefix trie
        :Output:
            node object set up as described by class, substring list is empty

        Complexity Analysis
            Let C be the value of 'characters'
        :Time complexity: O(C)
        :Time complexity analysis: Self nodes is n long, so takes n time to set up
        :Space complexity: O(C)
        :Space complexity analysis: Self nodes is n long, substrings is empty
        """
        self.nodes = [None]*characters
        self.substrings = []

class SLPrefixTrie:
    """
    Class Desciption:

    Creates a prefix trie that also stores a list of substrings starting with that prefix at every node
    Uses SLPrefixNode for the node implementation

    In initalisation takes a number of characters that may be in the alphabet and a key function that converts characters into indices

    """

    def __init__(self, characters : int, refString : str, key : callable) -> None:
        """
        Function description:
            Sets up a prefix trie, 
            Uses a naive implementation, adding each string one by one 

        :Input:
        characters:
            The highest index outputted by key. 
            Ideally should be equal to:
                either the length of the alphabet used for the refrence string, 
                or the number of unique characters in the refrence string
            depending on implementation
        refString: 
            The string from which to build the prefix tree.
        key:
            A function mapping the characters in the input alphabet to unique indicies
            Should act as a perfect hash for possible imput characters
        
        :Output:
            Createst a prefix tree of the refString, with substring lists as in the class description.
            Inserts suffixes in order from longest to shortest, resulting in substring lists begin sorted.
            Also stores input variables.
        
        Complexity Analysis
            let R be the length of refString
            let C be the length of characters
            let k be the complexity of the key function

        :Time complexity: O(kCN^2)*
        :Time complexity analysis:
            N strings to add in total: 
            
            Each string length is upper bound by N
            Insert is O(kCn), where n is length of string to be added.

            Thus upper bound of O(kCN * N) = O(kCN^2)
        
        *Generally, upon implementation, C should be a given constant and k should work in O(1) time, making complexity O(N^2)

        :Space complexity: O(CN^2)
        :Space complexity analysis:
            N inserts,
            O(nC) space where n is length of insert,
                At each step n is bound by N
            
            Therefore O(CN^2)
        """
        self.characters = characters
        self.refString = refString
        self.key = key

        self.root = SLPrefixNode(self.characters)

        #Insert every substring into the prefix tree
        for index in range(len(refString)):
            self.insert(index)

    def insert(self, index) -> None:
        """
        Function description: 
            Inserts a string into the given prefix trie:
            Gets string by reference to start index

            Starting at the root, use self.key to find the location in the array that the current character corresponds to
            If there is no node there yet, create one.
            Move to that node, add the current substring to the list of substrings
            Repeat until end of input string is reached

            
        :Input:
        index: The starting index of the substring to be inserted (index in reference string)

        :Output, return or postcondition:
        substring reference inserted as described above

        Complexity Analysis:
            Let n be length of the substring referenced by index
                (Can be calculated as len(self.refString) - index)
            let C be the length of characters
            let k be the complexity of the key function

        :Time complexity: O(kCn)
        :Time complexity analysis:
            Loops through the inserted character (n)
            for each loop, at worst
                find node indes with key (k)
                create node if necessary (O(C))
                add current node to substring list (O(append))

        :Space complexity: O(nC)
        :Space complexity analysis:
            node array takes C spaces. It is possible a new node is added for every character.
            Adding the substring to the substring list is a total of n new elements. 
            THEREFORE O(nC + n) = O(n)
        """

        current = self.root
        for i in range(index,len(self.refString)):
            

            cIndex = self.key(self.refString[i])

            if current.nodes[cIndex] == None:
                current.nodes[cIndex] = SLPrefixNode(self.characters)
            current = current.nodes[cIndex]
            current.substrings.append(index)
    
    def get(self, prefix) -> list:
        """
        Function description:
            Searches for the prefix through the prefix tree. 
            
            Starting at the root node, match the first character in the node list and move to that node
            Continue as such until either 
                a) there is no node at the location:  therefore the prefix cannot be found
                b) the end of the prefix is reached: return the substring list stored at that node

            For consistancy, if no result is found an empty string is returned.

        :Input:
        prefix: The prefix to search the string for

        :Return:
        Returns the list of substrings beginning with the given prefix

        Complexity
            N is the length of the prefix string
        :Time complexity: O(kN)
        :Time complexity analysis:
            attempts to loop through one node for every character in the prefix (n)
            at each node uses key to find next one
            upon termination just returns the pre-assembled array O(1)
        :Space complexity: O(1)
        """

        current = self.root
        for char in prefix:
            cIndex = self.key(char)

            if type(current.nodes[cIndex]) == SLPrefixNode:
                current = current.nodes[cIndex]
            else: 
                return []

        return current.substrings
        
class OrfFinder:
    """
    Class Description:

    Takes a genome (string comtaining some combination of characters A through D) and sets it up as a 
    'substring list' prefix tree (see SLPrefixTree class above).

    Has a method 'find' that returns all strings matching a given prefix and suffix
    """

    CHARACTERS = 4
    KEY = lambda self, x: ord(x) - ord('A')

    """ KEY lambda function/method analysis:
    Function description:
        Converts a character into a index using ord()
        See conversions below

    Approach description (if main function):
    :Input: character
    :Return:
        A => 0
        B => 1
        C => 2
        D => 3

    :Time complexity: O(1)
    :Time complexity analysis:
        Ord is assumed to be O(1). 
        Even if it is not, there is a constant/non-variable range of inputs to the function: A-D.
        Therefore the time will not scale with the input.
    :Space complexity: O(1)
    """
    

    def __init__(self, genome) -> None:  
        """
        Function description: Creates SLP trie
        Approach description (if main function):
        :Input:
        genome: The string to be prepped for searching (see specifications)

        :Output, return or postcondition:
        Sets up a SLPrefixTrie that can be used to match suffix-prefix substrings in the future.
        See SLPrefixTrie class for further detail

        Complexity Analysis:
            n is the length of genome

        :Time complexity: O(n^2)
        :Time complexity analysis:
            Compleity of SLPrefixTrie initalisation is O(kCN^2)
            Within OrfFinder: 
                k is constant (see KEY analysis above)
                C is constant, equal to 4 (given in specs.)
                N is the length of the input string, in this case genome

            Therefore: Complexity is O(n^2) where n is the length of the given genome

        :Space complexity: O(N^2)
        :Space complexity analysis:
            Space complexity of SLPrefixTrie is O(CN^2)
            C is constant in this case 

        """
 
        self.tree = SLPrefixTrie(OrfFinder.CHARACTERS, genome, OrfFinder.KEY)
        self.genome = genome



    def find(self, start, end) -> list:
        """
        Function description:
            Finds substrings of 'genome' that start and end with the given strings

        Approach:
            Uses the SLPrefixTrie to get:
                a list of substrings that start with the given prefix (indices) (pre)
                a list of substrings that end with the given suffix (indices) (post)

            Then given those lists compares the first (earliest) prefix with the last (lastest) suffix
            IF the prefix index is far enough before the suffix for there to be a legitimate pair, then add that index pair (as a tuple) to the list of solutions
                then move to the next latest suffix and compare
            WHEN a suffix is reached that does not match or there are no more suffixes to pair
                move to the next prefix.

            IF at any point a prefix does not pair with any suffix terminate.
                This is checked with a flag.

            Afterwards, use list splicing to retrieve the strings that correspond to the pairs. 

        :Input: 
        start: the prefix to match
        end: the suffix to match

        :Output, return, post condition:
        returns a list of substrings that match the provided prefix and suffix


        Complexity Analysis
            T is the length of start
            U is the length of end
            V is the number of characters in the output

        :Time complexity: O(T+U+V)
        :Time complexity analysis:
            SLPrefixTrie get method is O(CN) as descirbed above
                C is the number of characters which in OrfFinder is constant
                N is the length of the input

            Thus get(start) is O(T)
            get(end) is O(U)

            complexity of match_substrings is O(len(output)) (see method docstring below)
            complexity of retrieve_substrings is O(V) (see method docstring below)

            As retrieve_substrings runs on every output of match_substrings, O(retrive...) >= O(match...)

            Therefore time is O(T, U, V)

        :Space complexity: O(V)
        :Space complexity analysis: 
            Space complexity is O(V))

            Substring lists are merely referenced, not copied
            Output lists of match... and retireve... are same length,
            However, there are V total characters in retireve

            Thus O(V)
            
        """
        
        #Get prefix and suffix lists
        pre = self.tree.get(start)
        post = self.tree.get(end)

        
        #Match viable prefix and suffix indices together (see match_substrings for more detailed logic)
        substrings = self.match_substrings(pre, post, len(start), len(end))
        
        #Use list splicing to copy out the strings given the start and end indices
        substrings = self.retrieve_substrings(substrings)

        #Return the list of substrings
        return substrings
    
    def match_substrings(self, pre, post, startLen, endLen):
        """
        Function description: 
            Given a list of starting indicies, a list of ending indecies and the number of characters needed after each one, 
            creates a list of all viable string splices.
        Approach:
            From above:
            Then given those lists compares the first (earliest) prefix with the last (lastest) suffix
            IF the prefix index is far enough before the suffix for there to be a legitimate pair, then add that index pair (as a tuple) to the list of solutions
                then move to the next latest suffix and compare
            WHEN a suffix is reached that does not match or there are no more suffixes to pair
                move to the next prefix.
            IF at any point a prefix does not pair with any suffix terminate.
                This is checked with a flag.

            
            A start/pre is sufficiently early if there are 'startLen' characters between it and the end string
            
            The corresponding splice can be calculated as (start, end+endlen)

        :Input:
            pre: the list of substrings that start with the given prefix
            post: the list of substrings that end with the given prefix
            startLen: The length of start/prefix
            endLen: The length of end/post
        :Output, return, post condition:
            outputs a list of substrings descibed by their start and ending indices, as descirbed earlier.

        Complexity Analysis
        :Time complexity: O(len(output))
        :Time complexity analysis:
            Single loop complexity: 
                O(1) for arithmatic and comparison 
                O(1) for calculationg pairs)
            Number of cycles
                O(len(output)) for each correct solution
                Additional O(len(output)) for each faultly start end pair that comes difectly after a correct one. 
                Additional O(1) for comparing the first prefix with no solutions for early termination.

                Therfore O(len(output)) 
            
        :Space complexity: O(len(output))
        :Space complexity analysis: 
            One additional space is used for every viable solution
        """
        i = 0
        j = len(post)-1

        res = []

        while i < len(pre):
            matched = False
            j = len(post)-1

            while j >= 0 and pre[i] + startLen -1 < post[j]:
                res.append((pre[i], post[j]+endLen))
                matched = True
                j-=1
                
            if matched == False:
                return res
            i +=1
            
        return res

    def retrieve_substrings(self, substrings: list[tuple[int,int]]) -> list:
        """
        Function description:
            for every index pair in substrings, get the string splice corresponding to that range
        :Input:
        substrings: a list of pairs of start and end indicies
        :Output, return, post condition:
        List of actual substrings based on given pairs

        Complexity Analysis
            V is the length of characters in the correct output
        :Time complexity: O(V)
        :Time complexity analysis:
            Splicing takes time and space of the number of characters in the slice. 
            Therefore the time to copy the entire list of substring index pairs is will be the time and space of the entire list of corresponding characters
        :Space complexity: O(V)
        :Space complexity analysis: 
            Same as time
        """
        res = []
        for s in substrings:
            res.append(self.genome[s[0]:s[1]])
        return res
    




# QUESTION 1: Open Reading Frames
class AdjacencyListGraph:
    """
    Class Description:
    Basic adjacency list graph implementation, with get_adjacent() and len() implemented

    Acts as a base for the flow network class
    Does not have insert implemented because flow network implements its own version.
    """

    def __init__(self, nodes: int) -> None:
        """
        Function description:
        Sets up adjacency list, as a list of empty lists
        
        :Input:
        nodes: the number of edges in the graph
        :Output, return, post condition:
        creates empty adjacency representation of a graph with the given amount of nodes, to add edges to later

        Complexity Analysis
        :Time complexity: O(nodes)
        :Time complexity analysis:
            Makes a list for every node
        :Space complexity: O(nodes)
        :Space complexity analysis: 
            Stores a list for every node
        """
        self.length = nodes
        self.graph = [[] for _ in range(nodes)]

    def get_adjacent(self, s: int) -> list:
        """
        Function description:
            Gets all edges going out of the the given node
        
        :Input:
        s: source node
        :Output, return, post condition:
        returns list of edges


        Complexity Analysis
        :Time complexity: O(1)
        :Time complexity analysis:
            array access is O(1)
        :Space complexity: O(1)
        :Space complexity analysis: 
            Does not need additonal space, just references array
        """
        return self.graph[s]
    
    def __len__(self) -> int:
        """
        Function description:
            Returns number of edges in the graph
            (stored upon initalisation)

        Complexity Analysis
        :Time complexity: O(1)
        :Space complexity: O(1)
        """
        return self.length

@dataclass
class FNEdge:
    """
    Class Description:
        Function description:
        Approach:
        :Input:
        :Output, return, post condition:


        Complexity Analysis
        :Time complexity: 
        :Time complexity analysis:
        :Space complexity: 
        :Space complexity analysis: 
        """
    edge: int
    capacity: int
    reverse: 'FNEdge' = None
    flow: int = 0

class FlowNetwork(AdjacencyListGraph):
    def insert(self, u, t, capacity):
        """
        Function description:
        Approach:
        :Input:
        :Output, return, post condition:


        Complexity Analysis
        :Time complexity: 
        :Time complexity analysis:
        :Space complexity: 
        :Space complexity analysis: 
        """
        forward = FNEdge(t, capacity)
        back = FNEdge(u, 0, forward)
        forward.reverse = back

        self.graph[u].append(forward)
        self.graph[t].append(back)

        self.maxFlow = False #Flag to check if current flows are max

    def __getattribute__(self, name: str) -> Any:
        """
        Function description:
        Approach:
        :Input:
        :Output, return, post condition:


        Complexity Analysis
        :Time complexity: 
        :Time complexity analysis:
        :Space complexity: 
        :Space complexity analysis: 
        """
        return super().__getattribute__(name)


    def DFSAugment(self, u, t, bottleneck):
        """
        Function description:
        Approach:
        :Input:
        :Output, return, post condition:


        Complexity Analysis
        :Time complexity: 
        :Time complexity analysis:
        :Space complexity: 
        :Space complexity analysis: 
        """
        if u==t:
            return bottleneck

        self.visited[u] = True
        for e in self.get_adjacent(u):
            available = e.capacity - e.flow
            if available > 0 and self.visited[e.edge] == False:
                # print('hiii', available, e, sep=':')
                augment = self.DFSAugment(e.edge, t, min(bottleneck, available))
                if augment > 0:
                    e.flow += augment
                    e.reverse.flow -= augment
                    return augment
        return 0


    def fordFulkerson(self, s, t):
        """
        Function description:
        Approach:
        :Input:
        :Output, return, post condition:


        Complexity Analysis
        :Time complexity: 
        :Time complexity analysis:
        :Space complexity: 
        :Space complexity analysis: 
        """
        flow = 0
        augment = 1
        while augment > 0:
            self.visited = [False for _ in range(len(self.graph))]
            augment = self.DFSAugment(s, t, inf)
            flow += augment

        self.maxFlow = True
        return flow


#Solution
SHIFTS = 3
DAYS = 30

def sum_shifts(officers_per_org):
    """
        Function description:
        Approach:
        :Input:
        :Output, return, post condition:


        Complexity Analysis
        :Time complexity: 
        :Time complexity analysis:
        :Space complexity: 
        :Space complexity analysis: 
        """
    total = 0
    shiftTotals = [0] * SHIFTS
    shiftRequests = [[] for _ in range(SHIFTS)]
    for org in officers_per_org:
        for shift in range(SHIFTS):
            shiftTotals[shift] += org[shift]
            shiftRequests[shift].append(org[shift])
        total += sum(org)*DAYS

    # shiftTotals = [s*DAYS for s in shiftTotals]
    # print(shiftTotals)
    return total, shiftTotals, shiftRequests

def makeAllocateNetwork(preferences, totalShifts, shiftTotals, min_shifts, max_shifts) -> FlowNetwork:
    """
        Function description:
        Approach:
        :Input:
        :Output, return, post condition:


        Complexity Analysis
        :Time complexity: 
        :Time complexity analysis:
        :Space complexity: 
        :Space complexity analysis: 
        """
    """edge ranges: 
        0 supersource, 
        1 excess source, 
        2:n+1 personel, 
        " days, 
        , 
        30*n + 2 (incl.)
    """
    n = len(preferences)
    # m = len(officers_per_org)

    pI = 2
    pdI = pI + n
    dsI = pdI + n*DAYS
    tI = dsI + DAYS*SHIFTS
    # scI = dsI + DAYS*SHIFTS
    # tI = dsI + SHIFTS*m


    nodes = tI+1
    network = FlowNetwork(nodes)

    #excess shift edge
    network.insert(0, 1, totalShifts - n*min_shifts)
    #print(totalShifts - n*min_shifts)
    

    #For each officer
    for officer in range(n):
        #min-shift edges
        network.insert(0, pI + officer, min_shifts)

        #max-shift edges
        network.insert(1, pI + officer, max_shifts-min_shifts)

        #personnel-day edges
        for day in range(DAYS):
            network.insert(pI + officer, pdI + officer*DAYS + day, 1)

            #day-shift edges
            for shift in range(SHIFTS):
                #day-shift edges
                if preferences[officer][shift]:
                    network.insert(pdI + officer*DAYS + day, dsI + SHIFTS*day + shift, 1)

    #print(network.get_adjacent(0))

    for day in range(DAYS):  
        for shift in range(SHIFTS):      
            network.insert(dsI + SHIFTS*day + shift, tI, shiftTotals[shift]) #ds - cs
    
    return tI, network
    

    # #Shift-Company edges                   
    # for day in range(DAYS):  
    #     for shift in range(SHIFTS):      
    #         for company in range(m):
    #             network.insert(dsI + SHIFTS*day + shift, scI + SHIFTS*company + shift, officers_per_org[company][shift]) #ds - cs
    
    # #Company sink edges
    # for company in range(m):        
    #     network.insert(scI + SHIFTS*company + shift, tI, inf)

#Allocation Helper Functions
def getTrueEdge(network, i):
    """
        Function description:
        Approach:
        :Input:
        :Output, return, post condition:


        Complexity Analysis
        :Time complexity: 
        :Time complexity analysis:
        :Space complexity: 
        :Space complexity analysis: 
        """
    for edge in network.get_adjacent(i):
        if edge.capacity > 0 and edge.flow == 1: 
            return edge
    return None
    
def makeRequests(shiftRequest):
    """
        Function description:
        Approach:
        :Input:
        :Output, return, post condition:


        Complexity Analysis
        :Time complexity: 
        :Time complexity analysis:
        :Space complexity: 
        :Space complexity analysis: 
        """
    return [[shiftRequest[j].copy() for j in range(SHIFTS)] for _ in range(DAYS)] ##This is aliasing for some reason!!!

def makeAllocationList(n, m, network: FlowNetwork, shiftRequest: list[list]):
    """
        Function description:
        Approach:
        :Input:
        :Output, return, post condition:


        Complexity Analysis
        :Time complexity: 
        :Time complexity analysis:
        :Space complexity: 
        :Space complexity analysis: 
        """
    """Otherwise, it returns a list of lists allocation, where allocation[i][j][d][k] is equal
    to 1 if 
    security officer SOi
    is allocated to work for company Cj 
    during shift Sk 
    on day Dd."""


    #Set up output
    allocation = [[[[0 for i in range(SHIFTS) ] for _ in range(DAYS) ] for __ in range(m)] for ___ in range(n)]

    #Set up shift requirements:
    requests = makeRequests(shiftRequest)

    #Start indices
    pI = 2
    pdI = pI + n
    dsI = pdI + n*DAYS
    tI = dsI + DAYS*SHIFTS

    """
    personnel = pI + officer
    day       = pdI + officer*DAYS + day
    day-shift = dsI + SHIFTS*day + shift
    company   = scI + SHIFTS*company + shift
    """

    calP = lambda per: per - pI
    calD = lambda day, i: day - pdI - i*DAYS
    calS = lambda ds, d: ds - dsI - d*SHIFTS
    # calC = lambda com, d: (com - scI - d)//SHIFTS

    for personIndex in range(pI, n + pI):
        # dayEdge = getTrueEdge(network, personIndex)
        # if dayEdge == None:
        #     break
        for dayEdge in network.get_adjacent(personIndex):
            if dayEdge.capacity > 0 and dayEdge.flow == 1: 
                shiftEdge = getTrueEdge(network, dayEdge.edge)

                #i,j,d,k = p, c, s, d
                i = calP(personIndex)
                d = calD(dayEdge.edge, i)
                k = calS(shiftEdge.edge, d)

                #print(i,k,d, personIndex, dayEdge, shiftEdge, sep=':')
                #print(requests)
                dsRequest = requests[d][k]
                #print(dsRequest)
                while dsRequest[-1] <= 0:
                    dsRequest.pop()

                dsRequest[-1] -= 1
                j = len(dsRequest) - 1

                allocation[i][j][d][k] = 1
    
    return allocation

def potSolExists(preferences, min_shifts, max_shifts, totalShifts):
    """
        Function description:
        Approach:
        :Input:
        :Output, return, post condition:


        Complexity Analysis
        :Time complexity: 
        :Time complexity analysis:
        :Space complexity: 
        :Space complexity analysis: 
        """
    n = len(preferences)
    if min_shifts*n > totalShifts or totalShifts > max_shifts*n:
        return False
    return True

def allocate(preferences, officers_per_org, min_shifts, max_shifts):
    """
        Function description:
        Approach:
        :Input:
        :Output, return, post condition:


        Complexity Analysis
        :Time complexity: 
        :Time complexity analysis:
        :Space complexity: 
        :Space complexity analysis: 
        """
    totalShifts, shiftTotals, shiftRequests = sum_shifts(officers_per_org)

    if not potSolExists(preferences, min_shifts, max_shifts,totalShifts):
        return None

    t, network = makeAllocateNetwork(preferences, totalShifts, shiftTotals, min_shifts, max_shifts)
    flow = network.fordFulkerson(0, t)

    if flow != totalShifts:
        return None
    
    return makeAllocationList(len(preferences), len(officers_per_org), network, shiftRequests)



if __name__ == '__main__':
    #sol = allocate([[1,1,1], [1,1,1],[1,1,1],[1,1,1]], [[1,1,1]], 0, 25)
    print(dir())
    #print(sol)


    # o = OrfFinder("AAABAAABBCCDDAA")
    # print(o.find('AB','A'))

    # f = FlowNetwork(2)
    # f.insert(0,1,10)
    # print(f.fordFulkerson(0, 1))
    # print(f.graph)
