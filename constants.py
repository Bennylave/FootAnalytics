from matplotlib.lines import Line2D


shots_markers_dict = {
    "LeftFoot" : "<",
    "RightFoot": ">",
    "Head": "^",
    "Other": "s"
}

shots_colors_dict = {
    "Goal": "lime",
    "Saved": "yellow",
    "Post": "cyan",
    "OffT": "red",
    "Blocked": "orange",
    "Wayward": "black"
}

shots_tuple_legend = tuple([Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=shots_colors_dict[i],
                             markeredgecolor="black") for i in shots_colors_dict.keys()] + \
                     [Line2D(range(1), range(1), color="white", marker=shots_markers_dict[j], markerfacecolor="white",
                             markeredgecolor="black") for j in shots_markers_dict.keys()])

shots_tuple_labels = tuple(list(shots_colors_dict.keys()) + list(shots_markers_dict.keys()))


passes_markers_dict = {
    "High": "d",
    "Low": "o"
}

passes_colors_dict = {

}