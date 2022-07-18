
from sympy import Matrix
from sklearn.preprocessing import normalize
import scipy as sp
import numpy as np

import math
import qsharp
#from HostPython import SayHello

def is_unitary(m):
    return np.allclose(np.eye(m.shape[0]), m.H * m, atol=0.1)

def ColumnStochasticMatrixToUnitaryMatrix(input):
    s = len(input)
    n = len(input[0])
    output = np.zeros((n*s,n*s))
    for i in range(n):
        ci = input[i]
        #print("ci: ", ci)
        temps = [[ math.sqrt(j) for j in ci] for k in range(s)]
        #print("temp: ", temps)
        temps_matrix = Matrix(temps)
        #print("temps_matrix: ", temps_matrix)
        nspace = temps_matrix.nullspace()
        #print("nspace: ", nspace)
        nspace = [j.tolist() for j in nspace]
        nspace = np.concatenate(np.concatenate(nspace))
        nspace = [float(o) for o in nspace]
        nspace = np.array(nspace).reshape(s-1,s).tolist()
        #print("nspace concat: ", nspace)
        nspace.insert(0,ci)
        nspace = np.array(nspace).transpose()
        #print("nspace: ",nspace)
        nspace = normalize(nspace,axis=1, norm= 'l1')
        #print("nspace normalized: ", nspace)
        #print("is nspace unitary:", is_unitary(np.asmatrix( nspace.astype(complex) )))
        #row_sum = nspace.sum(axis=1)
        #nspace = nspace / row_sum[:,np.newaxis]


        output[i*s:s+i*s,i*s:s+i*s] = nspace
        #nspace_matrix = Matrix(nspace)
        #print("finali: ", finali)
    output = output.astype(complex)
    #return output
    print("final: ",output)
    print("is unitary: ", is_unitary(np.asmatrix(output)))

def ConstructW(n,s):
    # sxs density matrix is tensored with nxn density matrix
    output = np.zeros((n*s,s))
    for i in range(n-1):
        output[i*n][i] = 1
    return output

def ConstructV(n):
    pass

def SimulateHMMwithHQMM(transitions, emissions):
    # s - outputs
    # n - hidden states
    # yt - observed output at time t
    p = np.zeros((5,5))
    KwSet = ()
    KySet = ()

    U1 = ColumnStochasticMatrixToUnitaryMatrix(transitions)
    U2 = ColumnStochasticMatrixToUnitaryMatrix(emissions)
    #operators =


def main():
    #input = [[0,0.5,0.5],[0.1,0,0.2],[0.9,0.5,0.3]]
    input = [[0.5,0.5],[0.5,0.5]]
    ConstructW(4,3)
    #ColumnStochasticMatrixToUnitaryMatrix(input)

    #print(SayHello.simulate(name="vasya"))



if __name__ == "__main__":
    main()