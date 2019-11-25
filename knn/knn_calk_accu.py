f = open("out1.txt", "r")

lines = f.readlines()
trialLen = len(lines)

neigh = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for line in lines:
    lineList = list(map(float, line.split()))
    neigh[lineList[1]] += lineList[2]

for i in range(1, 10, 1):
    neigh[i] /= 100

for i in range(1, 10, 1):
    print(i, neigh[i])


