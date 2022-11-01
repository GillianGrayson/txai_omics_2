import plotly.graph_objects as go


def add_violin_trace(fig, y, name, showlegend=True):
    fig.add_trace(
        go.Violin(
            y=y,
            name=name,
            box_visible=True,
            meanline_visible=True,
            showlegend=showlegend,
        )
    )
