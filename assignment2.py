from dataclasses import dataclass
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

    def find(self, start, end) -> list:
        pre = self.tree.get(start)
        post = self.tree.get(end)

        substrings = self.match_substrings(pre, post, len(start), len(end))
        substrings = self.retrieve_substrings(substrings)

        return substrings
    
    def match_substrings(self, pre, post, startLen, endLen):
        print (pre,post)
        i = 0
        j = len(post)-1

        res = []

        while i < len(pre):
            matched = False
            j = len(post)-1

            while j >= 0 and pre[i] + startLen < post[j]:
                res.append((pre[i], pre[j]+endLen))
                j-=1
                
            if matched == False:
                return res
            
        return res

    def retrieve_substrings(self, substrings):
        res = []
        for s in substrings:
            res.append(self.refString[s[0], s[1]])
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


    def DFSAugment(self, u, t, bottleneck):
        if u==t:
            return [t], bottleneck

        self.visited[u] = True
        for e in self.get_adjacent(u):
            available = e.capacity - e.flow
            if available > 0 and self.visited[e.edge] == False:
                route, augment = self.DFSAugment(e.edge, t, min(bottleneck, available))
                if augment > 0:
                    e.flow += augment
                    e.reverse.flow -= augment
                    return route.append(t), augment
        return 0


    def fordFulkerson(self, s, t):
        flow = 0
        augment = 1
        backroutes = []
        while augment > 0:
            self.visited = [False for _ in range(len(self.graph))]
            route, augment = self.DFSAugment(s, t, inf)
            flow += augment
            backroutes.append(route)
        return flow, backroutes


#Solution
SHIFTS = 3
DAYS = 30

def sum_shifts(officers_per_org):
    total = 0
    for org in officers_per_org:
        total +=sum(org)
    return total

def makeAllocateNetwork(preferences, totalShifts, officers_per_org, min_shifts, max_shifts) -> FlowNetwork:
    """edge ranges: 
        0 supersource, 
        1 excess source, 
        2:n+1 personel, 
        " days, 
        , 
        30*n + 2 (incl.)
    """
    n = len(preferences)
    m = len(officers_per_org)

    pI = 2
    pdI = pI + n
    dsI = pdI + n*DAYS
    scI = dsI + DAYS*SHIFTS
    tI = scI + SHIFTS*m

    nodes = tI+1
    network = FlowNetwork(nodes)

    #excess shift edge
    network.insert(0, 1, totalShifts - n*min_shifts)

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

    #Shift-Company edges                   
    for day in range(DAYS):  
        for shift in range(SHIFTS):      
            for company in range(m):
                network.insert(dsI + SHIFTS*day + shift, scI + SHIFTS*company + shift, officers_per_org[company][shift]) #ds - cs
    
    #Company sink edges
    for company in range(m):        
        network.insert(scI + SHIFTS*company + shift, tI, inf)

def makeAllocationList(n, m, selections):
    """Otherwise, it returns a list of lists allocation, where allocation[i][j][d][k] is equal
    to 1 if 
    security officer SOi
    is allocated to work for company Cj 
    during shift Sk 
    on day Dd."""

    """Selections will look like [0)sink, 1)company, 2)day-shift, 3)day, 4)personnel, ...]"""

    #Set up output
    allocation = [[[[0 for i in range(DAYS) ] for i in range(SHIFTS) ] for i in range(m)] for i in range(n)]

    pI = 2
    pdI = pI + n
    dsI = pdI + n*DAYS
    scI = dsI + DAYS*SHIFTS
    tI = scI + SHIFTS*m
    """
    personnel = pI + officer
    day       = pdI + officer*DAYS + day
    day-shift = dsI + SHIFTS*day + shift
    company   = scI + SHIFTS*company + shift


    """

    for selection in selections:
        i = selection[4] - pI
        j = selection[3] - pdI - i*DAYS
        d = selection[2] - dsI - j*SHIFTS
        k = (selection[1] - scI - d)//SHIFTS

        allocation[i][j][d][k] = 1
    
    return allocation

def allocate(preferences, officers_per_org, min_shifts, max_shifts):
    totalShifts = sum_shifts(officers_per_org)
    network = makeAllocateNetwork(preferences, totalShifts, officers_per_org, min_shifts, max_shifts)
    
    selections, flow = network.fordFulkerson()
    if network.fordFulkerson() != totalShifts:
        return None
    
    return makeAllocationList(len(preferences), len(officers_per_org), selections)



if __name__ == '__main__':
    o = OrfFinder("AABBCCDDAA")
    print(o.find('A','D'))
    # f = FlowNetwork(2)
    # f.insert(0,1,10)
    # print(f.fordFulkerson(0, 1))
    # print(f.graph)
