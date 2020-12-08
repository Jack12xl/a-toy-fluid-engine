def swap(a, b):
    a, b = b, a


if __name__ == "__main__":
    a = [2, 3, 3]
    b = [3, 4, 4]
    swap(a, b)
    print(a)
    print(b)
