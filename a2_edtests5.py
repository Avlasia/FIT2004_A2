from assignment2 import allocate
preferences = [
                [1, 0, 0], 
                [1, 0, 0], 
            ]
officers_per_org = [
                [1, 0, 0],  
            ]

allocation = allocate(preferences, officers_per_org, min_shifts=15, max_shifts=30)
assert allocation is not None, allocation

allocation = allocate(preferences, officers_per_org, min_shifts=16, max_shifts=30)
assert allocation == None, allocation

allocation = allocate(preferences, officers_per_org, min_shifts=0, max_shifts=14)
assert allocation == None, allocation

allocation = allocate(preferences, officers_per_org, min_shifts=13, max_shifts=14)
assert allocation == None, allocation

allocation = allocate(preferences, officers_per_org, min_shifts=14, max_shifts=15)
assert allocation is not None, allocation