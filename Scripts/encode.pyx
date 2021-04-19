import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)

cdef str _binary_transfer_AT(str seq):
    seq = seq.replace("A", "0").replace("C", "1").replace("G", "1").replace("T", "0")
    seq = ''.join(list(filter(str.isdigit, seq)))

    return seq    

cdef str _binary_transfer_AC(str seq):
    seq = seq.replace("A", "0").replace("C", "0").replace("G", "1").replace("T", "1")
    seq = ''.join(list(filter(str.isdigit, seq)))

    return seq

cdef list _binary_transfer_loc(str binary_seq, int K):
    cdef list loc
    loc = []
    for i in range(0, len(binary_seq)-K+1):
        loc.append(int(binary_seq[i:i+K], 2))
    
    return loc

cdef np.ndarray[np.float64_t, ndim=2] _loc_transfer_matrix(list loc_list, int dis, int K):
    cdef np.ndarray[np.float64_t, ndim=2] matrix
    matrix = np.zeros((2**K, 2**K))
    for i in range(0, len(loc_list)-K-dis):
        matrix[loc_list[i]][loc_list[i+K+dis]] += 1
    
    return matrix

cdef np.ndarray[np.float64_t, ndim=1] _matrix_encoding(str seq, int K):
    cdef int length
    cdef np.ndarray[np.float64_t, ndim=1] feature
    cdef str binary_seq_1, binary_seq_2
    cdef list loc_1, loc_2
    seq = seq.upper()
    length = len(seq)
    binary_seq_1 = _binary_transfer_AT(seq)
    binary_seq_2 = _binary_transfer_AC(seq)
    loc_1 = _binary_transfer_loc(binary_seq_1, K)
    loc_2 = _binary_transfer_loc(binary_seq_2, K)
    
    feature = np.hstack((
        _loc_transfer_matrix(loc_1, 0, K).flatten(), _loc_transfer_matrix(loc_2, 0, K).flatten(),
        _loc_transfer_matrix(loc_1, 1, K).flatten(), _loc_transfer_matrix(loc_2, 1, K).flatten(),
        _loc_transfer_matrix(loc_1, 2, K).flatten(), _loc_transfer_matrix(loc_2, 2, K).flatten()))
    
    return feature/(length*1.0) * 100

def matrix_encoding(seq, K):

    return _matrix_encoding(seq, K)

