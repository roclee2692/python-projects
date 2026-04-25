class Person:
    def __init__(self,name,age):
        self.name=name
        self.age=age

    def greet(self):
        print(f"Hello {self.name}")
        print(self.age)

    def __del__(self):  # 析构函数（对象销毁时自动调用
        print(f"{self.name} has been deleted.")


p=Person("John",20)
p.greet()

flags = ["even" if x % 2 == 0 else "odd" for x in range(5)]
# ['even', 'odd', 'even', 'odd', 'even']
print(flags)
pairs = [(x, y) for x in range(2) for y in range(5)]
# [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
print(pairs)
s = " h e l l o "
clean = [c for c in s if c != " "]
# ['h', 'e', 'l', 'l', 'o']
print(clean)