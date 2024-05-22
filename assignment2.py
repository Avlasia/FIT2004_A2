from dataclasses import dataclass
from typing import Any
inf = float('inf')

# QUESTION 1: Open Reading Frames
class SLPrefixNode:
    def __init__(self, characters) -> None:
        self.nodes = [None]*characters
        self.substrings = []

class SLPrefixTree:

    def __init__(self, characters, refString, key) -> None:
        self.characters = characters
        self.refString = refString
        self.key = key


        self.root = SLPrefixNode(self.characters)
        for index in range(len(refString)):
            self.insert(index)

    def insert(self, index) -> None:
        current = self.root
        for i in range(index,len(self.refString)):
            

            cIndex = self.key(self.refString[i])

            if current.nodes[cIndex] == None:
                current.nodes[cIndex] = SLPrefixNode(self.characters)
            current = current.nodes[cIndex]
            current.substrings.append(index)
    
    def get(self, prefix) -> list:
        current = self.root
        for char in prefix:
            cIndex = self.key(char)

            if type(current.nodes[cIndex]) == SLPrefixNode:
                current = current.nodes[cIndex]
            else: 
                return []

        return current.substrings
        
 
class OrfFinder:
    CHARACTERS = 4
    KEY = lambda self, x: ord(x) - ord('A')

    def __init__(self, genome) -> None:   
        self.tree = SLPrefixTree(self.CHARACTERS, genome, self.KEY)
        self.genome = genome

    def find(self, start, end) -> list:
        pre = self.tree.get(start)
        post = self.tree.get(end)

        substrings = self.match_substrings(pre, post, len(start), len(end))
        substrings = self.retrieve_substrings(substrings)

        return substrings
    
    def match_substrings(self, pre, post, startLen, endLen):
        i = 0
        j = len(post)-1

        res = []

        while i < len(pre):
            matched = False
            j = len(post)-1

            while j >= 0 and pre[i] + startLen -1 < post[j]:
                res.append((pre[i], post[j]+endLen))
                j-=1
                
            if matched == False:
                return res
            i +=1
            
        return res

    def retrieve_substrings(self, substrings):
        res = []
        for s in substrings:
            res.append(self.genome[s[0]:s[1]])
        return res
    


# QUESTION 1: Open Reading Frames
class AdjacencyListGraph:
    def __init__(self, nodes) -> None:
        self.length = nodes
        self.graph = [[] for _ in range(nodes)]

    def get_adjacent(self, s) -> list:
        return self.graph[s]
    
    def __len__(self):
        return self.length

@dataclass
class FNEdge:
    edge: int
    capacity: int
    reverse: 'FNEdge' = None
    flow: int = 0

class FlowNetwork(AdjacencyListGraph):
    def insert(self, u, t, capacity):

        forward = FNEdge(t, capacity)
        back = FNEdge(u, 0, forward)
        forward.reverse = back

        self.graph[u].append(forward)
        self.graph[t].append(back)

        self.maxFlow = False #Flag to check if current flows are max

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)


    def DFSAugment(self, u, t, bottleneck):
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
    print(totalShifts - n*min_shifts)
    

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

    print(network.get_adjacent(0))

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
    for edge in network.get_adjacent(i):
        if edge.capacity > 0 and edge.flow == 1: 
            return edge
    return None
    
def makeRequests(shiftRequest):
    return [[shiftRequest[j].copy() for j in range(SHIFTS)] for _ in range(DAYS)] ##This is aliasing for some reason!!!

# def convertToID(n, m, per, day, ds, com):
#     #Start indices
#     pI = 2
#     pdI = pI + n
#     dsI = pdI + n*DAYS
#     tI = dsI + DAYS*SHIFTS

#     i = per - pI
#     j = day - pdI - i*DAYS
#     d = ds - dsI - j*SHIFTS
#     k = (com - scI - d)//SHIFTS

#     return i,j,d,k


def makeAllocationList(n, m, network: FlowNetwork, shiftRequest: list[list]):
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

def potSolExists(preferences, officers_per_org, min_shifts, max_shifts,totalShifts, shiftTotals, shiftRequests):
    n = len(preferences)
    if len(preferences) < totalShifts:
        return False
    if min_shifts*n

def allocate(preferences, officers_per_org, min_shifts, max_shifts):
    totalShifts, shiftTotals, shiftRequests = sum_shifts(officers_per_org)

    if not potSolExists(preferences, officers_per_org, min_shifts, max_shifts,totalShifts, shiftTotals, shiftRequests):
        return False

    t, network = makeAllocateNetwork(preferences, totalShifts, shiftTotals, min_shifts, max_shifts)
    flow = network.fordFulkerson(0, t)

    if flow != totalShifts:
        return None
    
    return makeAllocationList(len(preferences), len(officers_per_org), network, shiftRequests)



if __name__ == '__main__':
    sol = allocate([[1,1,1], [1,1,1],[1,1,1],[1,1,1]], [[1,1,1]], 0, 25)

    #print(sol)


    # o = OrfFinder("AAABAAABBCCDDAA")
    # print(o.find('AB','A'))

    # f = FlowNetwork(2)
    # f.insert(0,1,10)
    # print(f.fordFulkerson(0, 1))
    # print(f.graph)
