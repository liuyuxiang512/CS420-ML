import csv

with open("world-happiness.csv", "r") as file:
    read = csv.reader(file)

    i = 0
    data = []
    for line in read:
        if i == 0:
            factors = line[1:]
            print(factors)
        else:
            miss = False
            for n in line:
                if n == '':
                    miss = True
                    break
            if not miss:
                data.append(line[1:])
        i += 1

print(data)

with open("world-happiness.txt", "w", newline="") as file:
    file.write(factors[0])
    for factor in factors[1:]:
        file.write("\t" + factor)
    file.write("\n")

    for line in data:
        file.write(line[0])
        for n in line[1:]:
            file.write("\t" + n)
        file.write("\n")
