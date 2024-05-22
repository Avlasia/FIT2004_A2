from assignment2 import allocate
preferences = [
    [0, 1, 1],
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 0],
]
officers_per_org = [[2, 1, 1]]
min_shifts = 0
max_shifts = 30
assert allocate(preferences, officers_per_org, min_shifts, max_shifts) is not None

assert allocate(
    [
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 0],
    ],
    [[2, 1, 1], [1, 0, 1]],
    0,
    30,
) is not None

preferences = [
    [0, 1, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
    [0, 0, 1],
    [0, 0, 1],
]
officers_per_org = [[1, 1, 1]]
min_shifts = 0
assert allocate(preferences, officers_per_org, min_shifts, 14) is None
assert allocate(
    preferences,
    officers_per_org,
    min_shifts,
    15,
) is not None
assert allocate(
    preferences,
    officers_per_org,
    min_shifts,
    16,
) is not None

preferences = [
    [0, 1, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
    [0, 0, 1],
    [0, 0, 1],
]
officers_per_org = [[1, 1, 1]]
assert allocate(
    preferences,
    officers_per_org,
    15,
    15,
) is not None

preferences = [
    [0, 1, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
]
officers_per_org = [[1, 1, 1]]
assert allocate(preferences, officers_per_org, 11, 30) is None
assert allocate(preferences, officers_per_org, 10, 30) is not None

assert allocate(
    [
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 0],
    ],
    [[2, 1, 1], [1, 0, 1]],
    1,
    15,
) is not None