import plotly.graph_objects as go


def add_bar_trace(fig, x, y, text, name="", orientation='v'):
    showlegend = False if name == "" else True
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            name=name,
            text=text,
            textposition='auto',
            showlegend=showlegend,
            orientation=orientation
        )
    )
