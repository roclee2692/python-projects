for i in range(1,10,2): #1-9
    for j in range(1,i+1,2): #每次到i
        print(f"{i}*{j}={i*j}", end ="\t")
    print()
