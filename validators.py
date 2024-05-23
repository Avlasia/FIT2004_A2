#Pass

from assignment2 import allocate
from time import time


def no_allocation(preferences, officers_per_org, min_shifts, max_shifts):
    assert allocate(preferences, officers_per_org, min_shifts, max_shifts) == None


def valid_allocation(preferences, officers_per_org, min_shifts, max_shifts):
    allocation = allocate(preferences, officers_per_org, min_shifts, max_shifts)

    print (':)' , min_shifts)

    assert allocation != None
    assert len(allocation) == len(preferences)
    assert len(allocation[0]) == len(officers_per_org)
    assert len(allocation[0][0]) == 30
    assert len(allocation[0][0][0]) == 3

    # def valid_allocation(allocation, preferences, officers_per_org, min_shifts, max_shifts) -> bool:
    # Calculate total shifts per worker
    worker_shifts = [sum(shift for company in worker for day in company for shift in day) for worker in allocation]
    
    # Check if each worker has a valid number of shifts
    for shifts in worker_shifts:
        if shifts < min_shifts or shifts > max_shifts:
            raise ValueError("Worker has too many or too few shifts")
    
    # Check the allocation of shifts to each company
    num_companies = len(officers_per_org)
    num_days = 30
    num_shifts_per_day = 3
    
    for company in range(num_companies):
        for day in range(num_days):
            for shift in range(num_shifts_per_day):
                desired_shifts = officers_per_org[company][shift]
                total_shifts = sum(allocation[worker][company][day][shift] for worker in range(len(preferences)))
                
                if total_shifts != desired_shifts:
                    raise ValueError("Wrong amount of shifts given to company")
    
    return True