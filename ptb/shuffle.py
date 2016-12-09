import random
with open('data/rt-polaritydata/rt-polarity-test.txt','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('data/rt-polaritydata/rt-polarity-test-shuffled.txt','w') as target:
    for _, line in data:
        target.write( line )