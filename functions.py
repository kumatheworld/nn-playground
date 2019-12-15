import numpy as np
from visdom import Visdom

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env='main', title='Loss', xlabel='Epoch', ylabel=''):
        self.viz = Visdom()
        self.plots = {}
        self.env = env
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(
                X=np.array([x,x]),
                Y=np.array([y,y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=self.title,
                    xlabel=self.xlabel,
                    ylabel=self.ylabel
                )
            )
        else:
            self.viz.line(
                X=np.array([x]),
                Y=np.array([y]),
                env=self.env,
                win=self.plots[var_name],
                name=split_name,
                update = 'append'
            )
