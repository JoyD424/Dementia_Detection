import sys, re

def wordDiversity(listWords):
    dict = {}
    for word in listWords:
        if word not in dict:
            dict[word] = 1
    return len(dict)

def main():
    return

if __name__ == "__main__":
    main()