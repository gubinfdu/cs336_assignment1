import os
import time
from typing import BinaryIO, List, Tuple, Dict, Any, Optional, Iterable
from collections import Counter
import regex as re
from multiprocessing import Process, Queue

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> List[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def compare_value(s1: Tuple[str, ...], s2: Tuple[str, ...]) -> Tuple[str, ...]:
    # return larger value
    i = 0
    l = min(len(s1), len(s2))
    while i < l:
        if s1[i] > s2[i]:
            return s1
        elif s1[i] < s2[i]:
            return s2
        else:
            i += 1
    return None

# def get_largest_value(vs):
#     if len(vs) == 1:
#         return vs[0]
#     elif len(vs) == 2:
#         return compare_value(vs[0], vs[1])
#     s1 = vs.pop(0)
#     while len(vs) > 0:
#         s2 = vs.pop(0)
#         s1 = compare_value(s1, s2)
#     return s1

def flatten_list(d):
    d2 = []
    for i in d:
        if isinstance(i, list):
            d2 += flatten_list(i)
        else:
            d2.append(i)
    return d2

def get_pre_token(chunk: str, special_tokens: List[str], drop_special_token: bool = True) -> List[str]:
    # segs = [chunk]
    # for token in special_tokens:
    #     segs = [i.split(token) for i in segs]
    #     segs = flatten_list(segs)

    special_tokens = sorted(special_tokens, key=lambda x: -len(x))
    if not special_tokens:
        segs = [chunk]
    else:
        pattern = '|'.join(re.escape(token) for token in special_tokens)
        segs = re.split('(' + pattern + ')', chunk)

    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    d = []
    for seg in segs:
        if seg in special_tokens:
            if drop_special_token:
                continue
            else:
                d += [seg]
        else:
            d += re.findall(pat, seg)
    return d

def update_counter(counter: Dict[Any, int], temp_counter: Dict[Any, int]) -> Dict[Any, int]:
    for k, v in temp_counter.items():
        if k not in counter:
            counter[k] = 0
        counter[k] += v
    return counter

def gen_pre_token_counter(input_path: str, special_tokens: List[str]) -> Dict[str, int]:
    counter = {}
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, 100, '<|endoftext|>'.encode('utf-8'))
        for start, end in zip(boundaries[: -1], boundaries[1: ]):
            f.seek(start)
            chunk = f.read(end - start).decode('utf-8', errors='ignore')
            
            temp_counter = dict(Counter(get_pre_token(chunk, special_tokens)))
            counter = update_counter(counter, temp_counter)
    return counter

def gen_pair_counter(byte_counter: Dict[bytes, int]) -> Dict[Tuple[bytes, bytes], int]:
    pair_counter = {}
    for k, v in byte_counter.items():
        for i in range(len(k) - 1):
            if (k[i], k[i + 1]) not in pair_counter:
                pair_counter[(k[i], k[i + 1])] = 0
            pair_counter[(k[i], k[i + 1])] += v
    return pair_counter

def get_merge_pair(pair_counter):
    temp_pair = None
    temp_v = 0
    for pair, v in pair_counter.items():
        if v > temp_v:
            temp_pair = pair
            temp_v = v
        elif v == temp_v:
            temp_pair = compare_value(temp_pair, pair)
    merge_pair = temp_pair
    return merge_pair

def merge_counter(counter: Dict[Tuple[bytes, ...], int], pair: Tuple[bytes, bytes]) -> Dict[Tuple[bytes, ...], int]:
    new_counter = {}
    for k, v in counter.items():
        if (pair[0] not in k) or (pair[1] not in k):
            new_counter[k] = v
            continue
        k2 = []
        l = len(k)
        i = 0
        while i < l:
            if (i < l - 1) and (k[i] == pair[0]) and (k[i + 1] == pair[1]):
                k2.append(pair[0] + pair[1])
                i += 2
            else:
                k2.append(k[i])
                i += 1
        new_counter[tuple(k2)] = v
    return new_counter

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str], train_mode: bool = True) -> Any:
    pre_token_counter = gen_pre_token_counter(input_path, special_tokens)
    byte_counter = {tuple([bytes([i]) for i in k.encode('utf-8')]): v for k, v in pre_token_counter.items()}
    # pair_counter = gen_pair_counter(byte_counter)

    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')

    # merge token
    merges = []
    time_cost = []
    log = {}
    for i in range(vocab_size - len(vocab)):
        t0 = time.time()
        pair_counter = gen_pair_counter(byte_counter)
        t1 = time.time()
        
        merge_pair = get_merge_pair(pair_counter)
        t2 = time.time()

        merges.append((merge_pair[0], merge_pair[1]))
        vocab[len(vocab)] = merge_pair[0] + merge_pair[1]
        byte_counter = merge_counter(byte_counter, merge_pair)
        t3 = time.time()
        time_cost.append([i, t1 - t0, t2 - t1, t3 - t2])
        log[i] = [byte_counter, pair_counter]
    if train_mode:
        return vocab, merges
    else:
        return vocab, merges, log, time_cost

class BPETokenizer():
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = []) -> None:
        self.vocab = vocab # {int: bytes}
        self.vocab_reverse = {v: k for k, v in vocab.items()} # {bytes: int}
        self.merges = merges # [(bytes, bytes) ...]
        if special_tokens:
            self.special_tokens = special_tokens
        else:
            self.special_tokens = []

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None) -> 'BPETokenizer':
        pass

    def pre_token2ind(self, pre_token: str) -> List[int]:
        pre_token_bytes = pre_token.encode('utf-8')
        if pre_token_bytes in self.vocab_reverse:
            return [self.vocab_reverse[pre_token_bytes]]

        pre_token_bytes = tuple([bytes([i]) for i in pre_token_bytes])
        # pre_token_int = [self.vocab_reverse[i] for i in pre_token_bytes]
        # new_token_int = []
        for merge in self.merges:
            new_token_bytes = []
            if (merge[0] not in pre_token_bytes) or (merge[1] not in pre_token_bytes):
                continue
            i = 0
            l = len(pre_token_bytes)
            while i < l:
                if (i < l - 1) and (merge[0] == pre_token_bytes[i]) and (merge[1] == pre_token_bytes[i + 1]):
                    new_token_bytes.append(merge[0] + merge[1])
                    i += 2
                else:
                    new_token_bytes.append(pre_token_bytes[i])
                    i += 1
            pre_token_bytes = new_token_bytes
        token_int = [self.vocab_reverse[i] for i in pre_token_bytes]
        return token_int

    def encode(self, text: str) -> List[int]:
        pre_tokens = get_pre_token(text, self.special_tokens, drop_special_token=False)
        token_int = []
        for pre_token in pre_tokens:
            token_int += self.pre_token2ind(pre_token)
        return token_int

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            for token_int in self.encode(text):
                yield token_int

    def decode(self, ids: List[int]) -> str:
        token_bytes = bytes()
        vocab_size = len(self.vocab)
        for i in ids:
            if i < vocab_size:
                token_bytes += self.vocab[i]
            else:
                token_bytes += bytes('\uFFFD', encoding='utf-8')
        s = token_bytes.decode(encoding='utf-8', errors='replace')
        return s








if __name__ == '__main__':
    input_path = '../../data/TinyStoriesV2-GPT4-valid.txt'
    vocab, merges = train_bpe(input_path, 267, ['<|endoftext|>'])
    print (len(vocab), len(merges))







