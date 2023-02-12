import logging
import numpy as np
from matplotlib import pyplot as plt
import os


class Graph:
    def __init__(self, x: list, y: list, title: str) -> None:
        self.axis_x = x
        self.axis_y = y
        self.title = title

    def set_axis(self, x: list, y: list) -> None:
        self.axis_x, self.axis_y = x, y

    def export(self, episode: int, path: str) -> None:
        plt.plot(self.axis_x, self.axis_y)
        plt.savefig(os.path.join(path, f'{episode}-{self.name}.jpg'))
        plt.close()


class MultiGraph:
    def __init__(self, title: str) -> None:
        self.title = title
        self.graphs: list[Graph]
        self.graphs = []

    def new_graph(self, graph: Graph) -> None:
        self.graphs.append(graph)

    def export(self, episode: int, path: str) -> None:
        d = int(np.ceil(np.sqrt(len(self.graphs))))
        fig, ax = plt.subplots(d, d)
        # fig.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.set_figwidth(16)
        fig.set_figheight(16)

        def index2location(index: int) -> tuple:
            return index // d, index % d

        for i in range(len(self.graphs)):
            ax[index2location(i)].plot(self.graphs[i].axis_x,
                                       self.graphs[i].axis_y)
            ax[index2location(i)].set_title(self.graphs[i].title)
        print("save path:{}".format(os.path.join(path, f'{episode}-{self.title}.jpg')))
        fig.savefig(os.path.join(path, f'{episode}-{self.title}.jpg'))
        plt.close()


class MLogger:
    def __init__(self, name: str, base_dir: str) -> None:
        self.name = name
        self.path = os.path.join(base_dir, self.name)
        os.makedirs(self.path, exist_ok=True)

        self.logger = logging.getLogger(name)
        fh = logging.FileHandler(os.path.join(self.path, f'{name}.log'))
        fh.setLevel(logging.WARNING)
        self.logger.addHandler(fh)

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.graphs: list[Graph]
        self.graphs = []
        self.multigraphs: list[MultiGraph]
        self.multigraphs = []

    def new_graph(self, graph: Graph) -> None:
        self.graphs.append(graph)

    def new_multigraph(self, graph: MultiGraph) -> None:
        self.multigraphs.append(graph)

    def export(self, episode: int) -> None:
        for g in self.graphs:
            g.export(episode, self.path)
        for g in self.multigraphs:
            g.export(episode, self.path)

    def reset(self):
        self.graphs = []
        self.multigraphs = []
