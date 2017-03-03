import cPickle as pkl

# Save the file in the desired format

final_string = ""

with open("test_data.txt") as f:
    sents = f.readlines()

with open ('final_results.pkl', 'rb') as fp:
    labels = pkl.load(fp)


for i in range(len(sents)):
    final_string += str(labels[i]) + "\t" + sents[i]

# print final_string

with open("final_results2.txt", "wb") as f:
    f.write(final_string)
