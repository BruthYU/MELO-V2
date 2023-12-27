a = ["1", "2", "3"]
b = ["a", "b", "c"]


for idx,(x,y) in enumerate(zip(a,b)):
    print(f"{idx}, {x}, {y}")