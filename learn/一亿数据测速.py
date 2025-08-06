import time
start = time.time()
total=sum(range(1,100_000_001))
end=time.time()
print("sum: ",total)
print(f"Time: {end - start:.4f} seconds")
