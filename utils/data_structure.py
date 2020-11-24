

class bufferPair:
    # a ping-pong buffer for gpu computation
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


class MultiBufferPair:
    """
    Specifically designed for FaceGrid
    """
    def __init__(self, cur, nxt):
        """

        :param cur: one FaceGrid
        :param nxt: another
        """
        assert(len(cur.fields) == len(nxt.fields))
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        for w_c, w_n in zip(self.cur.fields, self.nxt.fields):
            w_c, w_n = w_n, w_c
        self.cur, self.nxt = self.nxt, self.cur