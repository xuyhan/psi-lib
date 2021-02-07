from typing import List
import numpy as np
from crt import crt, extended_gcd, inverse

class Bases:
    def __init__(self, bases: List[int]):
        self.bases = bases
        self.num_bases = len(bases)
        self.prods, self.prods_inv = self.get_prods()
        self.big_base = self.big_base()

    def big_base(self):
        x = self.bases[0]
        for i in range(1, self.num_bases):
            x *= self.bases[i]
        return x

    def get_prods(self):
        prods = [1 for _ in range(self.num_bases)]
        prods_inv = [1 for _ in range(self.num_bases)]

        for i in range(self.num_bases):
            for j in range(self.num_bases):
                if i == j:
                    continue
                prods[i] *= self.get(j)

        for i in range(self.num_bases):
            prods_inv[i] = extended_gcd(prods[i], self.get(i))[0]

        return prods, prods_inv

    def get(self, i: int):
        return self.bases[i]

    def prod(self, i: int):
        return self.prods[i]

    def prod_inv(self, i: int):
        return self.prods_inv[i]

    def join(self, bases):
        bases_combined = self.bases + bases.bases
        return Bases(bases_combined)


class RNSInt:
    def __init__(self, bases: Bases, base_values=None):
        self.bases = bases
        if not base_values:
            self.base_values = [0 for _ in range(self.bases.num_bases)]
        elif isinstance(base_values, List):
            self.base_values = base_values
        elif isinstance(base_values, int):
            self.base_values = [base_values % self.bases.get(i) for i in range(self.bases.num_bases)]
        else:
            raise Exception('Invalid base_values argument.')

    def debug(self):
        print('Bases: ' + str(self.bases.bases))
        print('Big base: ' + str(self.bases.big_base))
        print('Base values: ' + str(self.base_values))
        print('True val: ' + str(self.get_val()))

    def get_val(self):
        return crt(self.base_values, self.bases.bases)

    def switch_bases(self, bases_new: Bases):
        z = sum([
            ((self.base_values[j] * self.bases.prod_inv(j)) % self.bases.get(j)) * self.bases.prod(j) for j in range(self.bases.num_bases)
        ])

        base_values = [z % bases_new.get(i) for i in range(bases_new.num_bases)]
        return RNSInt(bases_new, base_values)

    def raise_bases(self, bases_additional: Bases):
        conv = self.switch_bases(bases_additional)

        bases_combined = self.bases.join(bases_additional)
        base_values_combined = self.base_values + conv.base_values

        return RNSInt(bases_combined, base_values_combined)

    def reduce_bases(self, k):
        # find compliment bases
        l = self.bases.num_bases - k
        bases_target = Bases(self.bases.bases[k:])

        bases_comp = Bases(self.bases.bases[0:k])
        values_comp = self.base_values[0:k]

        # transform bases of complement
        comp = RNSInt(bases_comp, values_comp)
        conv = comp.switch_bases(bases_target)

        values_new = [0 for _ in range(l)]
        for j in range(l):
            values_new[j] = self.base_values[k + j] - conv.base_values[j]
            values_new[j] *= inverse(comp.bases.big_base, bases_target.get(j))
            values_new[j] = values_new[j] % bases_target.get(j)

        return RNSInt(bases_target, values_new)

D = Bases([47, 53, 643, 653, 659, 661, 673])
B = Bases([829, 911])

a = RNSInt(D, 5687878)
a.debug()

# b = a.switch_bases(C)
# b.debug()

b = a.raise_bases(B)
b.debug()