import csv

def pad(lst, size, sym):
  if len(lst) < size:
    return lst[:-1] + [sym for i in range(size - len(lst))] + [10001]
  elif len(lst) > size:
    return lst[:(size-1)] + [10001]
  else:
    return lst 

sym = 10002
size = 100

mini_ts = []
with open("./data/ts_mini_32_10000_v01.csv", 'rb') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		line = [int(i) for i in line]
		line = pad(line, size, sym)
		mini_ts.append(line)

train = mini_ts[:10000]
valid = mini_ts[10000:]

trainfile = "./data/blackholes.train.txt"
validfile = "./data/blackholes.valid.txt"
testfile  = "./data/blackholes.total.txt"

with open(trainfile, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for line in train:
        writer.writerow(line)

with open(validfile, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for line in valid:
        writer.writerow(line)

with open(testfile, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for line in mini_ts:
        writer.writerow(line)