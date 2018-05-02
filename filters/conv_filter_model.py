# the filter model for a CONV layer


class CONVFilterModel:

    def __init__(self, filter_size: int, channels_in: int, channels_out):
        self.filter_size = filter_size
        self.channels_in = channels_in
        self.channels_out = channels_out
