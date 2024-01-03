def non_return():
    return 1, None

a, b = non_return()
print(a,b)