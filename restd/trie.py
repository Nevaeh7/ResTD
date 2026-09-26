from typing import List, Optional


class Trie(object):
    def __init__(self, sequences: Optional[List[List[int]]] = None):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences or []:
                self.add(sequence)

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        node = self.trie_dict
        for token in sequence:
            if token not in node:
                node[token] = {}
            node = node[token]
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        node = self.trie_dict

        for token in prefix_sequence:
            if token in node:
                node = node[token]
            else:
                if self.append_trie:
                    return self.append_trie.get(prefix_sequence)
                else:
                    return []

        output = list(node.keys())

        if self.append_trie and self.bos_token_id is not None:
            if self.bos_token_id in output:
                output.remove(self.bos_token_id)
                output += list(self.append_trie.trie_dict.keys())

        return output

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        return trie

    def __iter__(self):
        stack = [([], self.trie_dict)]
        while stack:
            prefix, node = stack.pop()
            if node:
                for token, child_node in node.items():
                    stack.append((prefix + [token], child_node))
            else:
                yield prefix

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)


def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens
