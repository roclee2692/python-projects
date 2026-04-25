def greet(name):
    print(f"Hello, {name}!")

greet("Alice")  # 调用
def mihoyo(x,y):
    x+=x
    y-=y
    return print(x+y)
mihoyo(34,56)
def fact(n):
    s=1
    for i in range(1,n+1):
        s*=i
    return s
print(fact(3))
def genshin():
    print("米哈游你没良心")
    print("你说我抽卡什么时候出吧")
    print("出不出")
    print("我劝你赶紧出")
    print("信不信我求你")

genshin()

def grid():
    global j
    for i in range(0,21,2):
        for j in range (0,i,2):
            print(f"{i}*2+{j}*2={i+i+j+j}",end="   ")
        print()

Hi=lambda x:x+x-4+3*5
print(Hi(10))
def outer(x):
    def inner(y):
        return x + y
    return inner

add5 = outer(5)
print(add5(3))  # 输出 8

def decorator(func):
    def wrapper():
        print("Before")
        func()
        print("After")
    return wrapper

@decorator
def greet():
    print("Hello")

greet()


