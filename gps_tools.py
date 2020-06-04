#!/usr/bin/python3


import math
import re
import numpy
import argparse


class IGC_path(object):
    def __init__(self, input_files):
        self.path = []
        self.stats = {}
        for each in input_files:
            self.append(each)
        self.calc_stats()

    def __str__(self):
        text = """
        avg. GPS sample rate = {sr:.1f} sec/sample
        log duration = {dura:.1f} min
        avg. speed = {aspd:.2f} mph
        min alti. = {mina:.0f} ft
        max alti. = {maxa:.0f} ft

        """
        return text.format(
            sr=self.stats["avg sample rate"],
            dura=self.stats["duration"],
            aspd=self.stats["avg speed"],
            mina=self.stats["min alti"],
            maxa=self.stats["max alti"])

    def _parse(self, lines):
        # B HH MM SS DD MMmmm N DDD MMmmm E V PPPPP GGGGG CR LF
        # Description Size    Element    Remarks
        # Time        6 bytes HHMMSS     Valid characters 0-9
        # Latitude    8 bytes DDMMmmmN   Valid characters N, S, 0-9
        # Longitude   9 bytes DDDMMmmmE  Valid characters E,W, 0-9
        # Fix valid   1 byte  V          A: valid, V:nav warning
        # Press Alt.  5 bytes PPPPP      Valid characters -, 0-9
        # GNSS alt.   5 bytes GGGGG      Valid characters -, 0-9
        pattern = "B(\d{2})(\d{2})(\d{2})(\d{7})(N|S)(\d{8})(E|W)(A|V)([0-9\-]{5})([0-9\-]{5})"
        matches = re.findall(pattern, lines)
        path = []
        for args in matches:
            path.append([
                # seconds
                int(args[0]) * 3600 + int(args[1]) * 60 + int(args[2]),
                # longitude
                (float(args[5][:3]) + float(args[5][3:]) / 60000) * (-1 if args[6] == "W" else 1),
                # latitude
                (float(args[3][:2]) + float(args[3][2:]) / 60000),
                # altitude
                float(args[8]) * 3.28084])

        # merge points with same timestamp (rolling avg.)
        i = 0
        while i < len(path) - 1:
            while path[i][0] == path[i+1][0]:
                path[i] = list((numpy.array(path[i]) + numpy.array(path[i+1])) / 2.0)
                del path[i+1]
            i += 1

        path = numpy.array(path)

        # fix 12/24 hour wrap
        for i in range(len(path)-1):
            while path[i, 0] > path[i+1, 0]:
                path[i+1:, 0] += 43200

        # convert to flat projection (estimated) and center
        mean_lat = numpy.mean(path[:, 2])
        print("        avg. latitude: {:.2f} deg".format(mean_lat))
        long_scale = 2.09246e7 * math.pi / 180 * math.cos(mean_lat * 0.01745329)
        lat_scale = 364173.2
        path -= path[0]
        path *= [1.0, long_scale, lat_scale, 1.0]

        return path

    def append(self, filename):
        with open(filename, "r") as file:
            print("\n+++ Parsing: " + filename)
            data = self._parse(file.read())
        if len(self.path):
            self.path = numpy.append(self.path, data)
        else:
            self.path = data

    def calc_stats(self):
        path = self.path
        delta = path[1:] - path[:-1]

        self.stats["duration"] = (path[-1, 0] - path[0,0]) / 60

        self.stats["start"] = path[0]
        self.stats["end"] = path[-1]

        self.stats["avg sample rate"] = numpy.mean(delta[:, 0])

        self.stats["min alti"] = numpy.min(path[:, 3])
        self.stats["max alti"] = numpy.max(path[:, 3])

        self.velocity = numpy.linalg.norm(delta[:, 1:], axis=1) / delta[:, 0] * 3600 / 5280
        self.velocity_smooth = self._smooth_recur(self.velocity, math.ceil(100 / self.stats["avg sample rate"]))
        self.stats["avg speed"] = numpy.mean(self.velocity)

        self.max_vario_hist = {}
        self.min_vario_hist = {}
        for i in range(len(self.path)-1):
            d = 1
            while i + d < len(self.path) and self.path[i+d, 0] - self.path[i, 0] <= 30:
                delta_time = self.path[i+d, 0] - self.path[i, 0]
                rate = (self.path[i+d, 3] - self.path[i, 3]) / delta_time * 60
                bin_size = math.ceil(self.stats["avg sample rate"])
                bin_time = math.ceil(delta_time / bin_size) * bin_size
                self.max_vario_hist[bin_time] = max(rate, self.max_vario_hist.get(bin_time, 0))
                self.min_vario_hist[bin_time] = min(rate, self.min_vario_hist.get(bin_time, 0))
                d += 1

    def plots(self):
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots()

        # Figure 1 - speed and altitude
        color = 'tab:red'
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('speed (mph)', color=color)
        ax1.plot(self.path[1:, 0], self.velocity, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.plot(self.path[1:, 0], self.velocity_smooth, color="maroon")

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('alti. (ft)', color=color)  # we already handled the x-label with ax1
        ax2.grid(b=True, which='major', color="lightskyblue", linestyle='-')
        ax2.plot(self.path[:, 0], self.path[:, 3], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_yticks(numpy.arange(round(min(self.path[:, 3]) / 100) * 100, round(max(self.path[:, 3]) / 100) * 100, 100))
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

        # Figure 2 - speed histogram
        plt.title("speed histogram")
        plt.ylabel("")
        plt.xlabel("speed (mph)")
        marks = numpy.arange(0, math.ceil(max(self.velocity_smooth)) + 1)
        plt.xticks(marks)
        plt.yticks([])
        plt.hist(self.velocity_smooth, bins=marks)
        plt.show()


        # Figure 3 - sustained climb/sink
        plt.title("vario: maximum sustained climb/sink")
        plt.ylabel("rate   (ft / min)")
        plt.xlabel("duration (s)")
        res = math.ceil(self.stats["avg sample rate"])
        plt.xticks(numpy.arange(0, max(self.max_vario_hist.keys()) + res, res))
        plt.bar(self.max_vario_hist.keys(), self.max_vario_hist.values(), width=self.stats["avg sample rate"] - 0.1)
        plt.bar(self.min_vario_hist.keys(), self.min_vario_hist.values(), width=self.stats["avg sample rate"] - 0.1)
        plt.show()

    def _smooth_recur(self, data, window_len=7):
        if len(data.shape) > 1:
            retval = numpy.zeros(data.shape)
            for axis in range(data.shape[1]):
                retval[:, axis] = self._smooth_recur(data[:, axis], window_len)
            return retval
        else:
            s = numpy.r_[data[window_len - 1:0:-1], data, data[-2:-window_len - 1:-1]]
            w = numpy.hamming(window_len)
            return numpy.convolve(w / w.sum(), s, mode='valid')[window_len // 2:-window_len // 2 + 1]

    def smooth(self, factor=20):
        self.path = self._smooth_recur(self.path, math.ceil(factor / self.stats["avg sample rate"]))
        self.calc_stats()


# ==============================================================================


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description='GPS log insights!')
    parser.add_argument('input_files', nargs='+',
                        help='comma separated input .IGC files')
    parser.add_argument('-g', dest='show_graph', action='store_true', default=False,
                        help='show graphs')
    parser.add_argument('-s', metavar='amount', dest='smooth', nargs=1, type=int,
                        help='path smoothing factor (recommend 20)')
    args = parser.parse_args()

    # DO THINGS
    for each in args.input_files:
        path = IGC_path([each])

        if args.smooth:
            path.smooth(min(100, max(1, args.smooth[0])))

        print(path)

        if args.show_graph:
            path.plots()
