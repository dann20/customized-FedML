import numpy as np

# list of arrays to list of lists
def to_nested_list(listArrays):
    return [listArrays[i].tolist() for i in range(len(listArrays))]

# list of lists to list of arrays
def to_list_arrays(nestedList):
    return [np.asarray(nestedList[i], dtype=object) for i in range(len(nestedList))]

# list of lists to array of arrays
def to_array_arrays(nestedList):
    listArrays = to_list_arrays(nestedList)
    return np.asarray(listArrays, dtype=object)

# array of arrays to list of arrays
def aa_to_list_arrays(arrayArrays):
    return arrayArrays.tolist()
