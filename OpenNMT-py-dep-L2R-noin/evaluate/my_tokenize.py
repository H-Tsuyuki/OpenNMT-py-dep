import argparse
from nltk import word_tokenize

def read(file_path):
    with open(file_path) as f:
        sents = [" ".join(word_tokenize(sent.lower())) for sent in f.readlines()]
    return sents

def write(file_path, output):
    with open(file_path, "w") as f:
        f.write("\n".join(output))
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_file', required=True)
    parser.add_argument('-out_file', required=True)
    parser.add_argument('-lower', action='store_true')

    args = parser.parse_args()

    output = read(args.in_file)
    write(args.out_file, output)

if __name__ == '__main__':
    main()
