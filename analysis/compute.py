

class MyObject:

    def __init__(self):
        self.a = 7

    def __getattr__(self, item):
        print("utujire")
        return getattr(self, name=item)


def main():

    mo = MyObject()
    print(mo.a)


main()