import pickle
import os


class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def save(obj, file_name):

    dir_path = os.path.dirname(file_name)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    print(f"Saving file '{file_name}'...")
    with open(file_name, 'wb') as f:
        pickle.dump(obj=obj, file=MacOSFile(f))


def load(file_name):

    print(f"Loading file '{file_name}'...", end=" ")
    with open(file_name, 'rb')as f:
        data = pickle.load(file=f)
    print("Done!\n")
    return data
