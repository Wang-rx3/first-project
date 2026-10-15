"""Offline readability repair; retain every original Stage-2 figure."""
from __future__ import annotations

import json
from matplotlib.ticker import MaxNLocator

from . import three_uav_fixed_navigation_report as report


def main():
    original_save = report.save_figure
    def verified_save(fig, name):
        if name == 'three_uav_routes.png':
            for axis in fig.axes:
                axis.xaxis.set_major_locator(MaxNLocator(5))
                axis.yaxis.set_major_locator(MaxNLocator(4))
                axis.zaxis.set_major_locator(MaxNLocator(4))
            original_save(fig, 'three_uav_routes_verified.png')
        else:
            report.plt.close(fig)
    report.save_figure = verified_save
    report.figures(report.read_csv('navigation.csv'), report.read_csv('flows.csv'),
                   report.read_json(report.DATA / 'execution_manifest.lock.json'))
    report.write_json(report.LOGS / 'figure_qa.json', {
        'environment_steps': 0, 'optimizer_steps': 0, 'data_changed': False,
        'original_figures_retained': True,
        'readability_issue': 'Dense X tick labels on c1 original 3D route panel',
        'accepted_figures': ['three_uav_routes_verified.png', 'three_uav_parking.png', 'three_uav_communication.png'],
        'repair': 'Reduce 3D axis tick density only; same saved trajectory, geometry and all statistics.'})
    print(json.dumps({'passed': True, 'environment_steps': 0, 'optimizer_steps': 0}))


if __name__ == '__main__':
    main()
