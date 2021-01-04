from typing import List

def mult_inverse(a, b):
    s = None
    for j in range(1, b):
        if a * j % b == 1:
            s = j
    return s

def gcd(a, b):
    r = a % b
    while r != 0:
        a = b
        b = r
        r = a % b
    return b

def extended_gcd(a, b):
    s = 0; old_s = 1
    t = 1; old_t = 0
    r = b; old_r = a

    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_s, old_t, old_r


def crt(p: List[int], q: List[int]):
    if len(p) != len(q):
        raise Exception("Invalid number of ps and qs")

    t = q[0]
    for i in range(1, len(q)):
        t *= q[i]
    x = 0

    for i in range(len(p)):
        r = t // q[i]

        s = extended_gcd(r, q[i])[0] % q[i]

        x += (p[i] * s) % q[i] * r

    return x % t

if __name__ == '__main__':
    p = [1,2,1123,1123111,2234]
    q = [65537,163841,786433,1769473,37323411123111441451231312312312313131232321315553]
    x = crt(p, q)
    for i in range(5):
        print(x % q[i])

    print(x)