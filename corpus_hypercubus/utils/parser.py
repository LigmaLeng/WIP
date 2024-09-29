import argparse
from typing import Optional
import numpy as np

class Argparser:
    def __init__(self, input:str) -> object:
        self._input = input
        self.vocab = []
        with open(input, "rb") as infile:
            self.vocab_size, self.dim = map(int, infile.readline().split())
            self.vectors = np.empty(shape=(self.vocab_size, self.dim), dtype=np.float32)
            for i, line in enumerate(infile):
                embedding = line.split()
                self.vocab.append(embedding[0].decode("utf-8"))
                self.vectors[i, :] = np.fromiter((float(_vec) for _vec in embedding[1:]), dtype=np.float32)


    def save(self, output:Optional[str]=None) -> None:
        if not output:
            output = self._input
        output = output.split(".")
        output = output if len(output) == 1 else "".join(output[-2::-1])
        np.save(output+"-vec.npy", self.vectors)
        vocab = "\n".join(self.vocab)
        with open(output+"-vocab.txt", "wb") as outfile:
            outfile.write(vocab.encode("utf-8"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Text format embedding file')
    parser.add_argument("-o", '--output', default=None, required=False, help='Output name extension optional')
    args = parser.parse_args()
    parser = Argparser(args.input)
    parser.save(args.output)

