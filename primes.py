import numpy
from collections import Counter


def primesfrom3to(n):
    """ Returns a array of primes, 3 <= p < n """
    sieve = numpy.ones(n//2, dtype=numpy.bool)
    for i in range(3,int(n**0.5)+1,2):
        if sieve[i//2]:
            sieve[i*i//2::i] = False
    return 2*numpy.nonzero(sieve)[0][1::]+1


def primesfrom2to(n):
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = numpy.ones(n//3 + (n%6==2), dtype=numpy.bool)
    for i in range(1,int(n**0.5)//3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[       k*k//3     ::2*k] = False
            sieve[k*(k-2*(i&1)+4)//3::2*k] = False
    return numpy.r_[2,3,((3*numpy.nonzero(sieve)[0][1:]+1)|1)]


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

n = 10000
primes = primesfrom2to(n)
primes_index = {v: k for v, k in zip(range(len(primes)), primes)}
inv_primes_index = {v: k for k, v in primes_index.items()}

maximum_prime = primes[-1]
primes_predecessors = primes - 1

N = numpy.zeros(n)
array_prime = numpy.zeros(n)
array_prime_predecessor = numpy.zeros(n)
decompositions = []
decompositions_predecessor = []

for i in range(n):
    N[i] = i
    array_prime[i] = 1 if i in primes else 0
    array_prime_predecessor[i] = 1 if i+1 in primes else 0
    decompositions.append(prime_factors(i))
    decompositions_predecessor.append(prime_factors(i-1))
reference = numpy.column_stack((N, array_prime, array_prime_predecessor))

print(decompositions)

decomposition_based = []
for decomposition in decompositions:
    counted_dec = Counter(decomposition)
    decomposition_based.append(counted_dec)

print(decomposition_based)

temp = numpy.zeros((len(primes), n))
inv_temp = numpy.zeros((n, len(primes)))

print(temp.shape)

count = 0
for db in decomposition_based:
    for prime, power in dict(db).items():
        print(db)
        prime_index = inv_primes_index[prime]
        temp[prime_index][count] = power
        inv_temp[count][prime_index] = power

    count += 1


print(temp)


def create_is_two_n_plus_1(inv_temp):
    array = numpy.zeros(inv_temp.shape[0])
    for i in range(inv_temp.shape[0]):
        if sum(inv_temp[i][1:]) == 0:
            if inv_temp[i][0] > 0:
                array[i] = 1

    return array


def create_is_ppp(inv_temp):
    array = numpy.zeros(inv_temp.shape[0])
    for i in range(inv_temp.shape[0]):
        j=0
        while inv_temp[i][j] == 1:
            j+= 1
        if j > 0 and sum(inv_temp[i][j:]) == 0:
            array[i] = 1
    return array

two_n_plus_1 = create_is_two_n_plus_1(inv_temp)
ppp = create_is_ppp(inv_temp)


def compute_number_from_row(array):
    numbers = numpy.prod(array ** inv_temp)
    return numbers

import pandas as pd
df = pd.DataFrame(inv_temp)
df['prime'] = array_prime
df['prime_minus_1'] = array_prime_predecessor
df['ppp'] = ppp
df['two_n_plus_one'] = two_n_plus_1

print('a')

