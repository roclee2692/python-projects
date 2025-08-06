def is_prime(num):
    if num<=1:
        return False
    for i in range(2,num):
        if num%i==0:
            return False
        return True

print(is_prime(5))
print(is_prime(6))
print(is_prime(7))
print(is_prime(8))