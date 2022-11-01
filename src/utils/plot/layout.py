from plotly import graph_objects as go


def add_layout(fig, x_label, y_label, title, font_size=25):
    fig.update_layout(
        template="none",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5,
            itemsizing='constant'
        ),
        title=dict(
            text=title,
            font=dict(
                size=font_size
            )
        ),
        autosize=True,
        margin=go.layout.Margin(
            l=120,
            r=20,
            b=80,
            t=100,
            pad=0
        ),
        showlegend=True,
        xaxis=get_axis(x_label, font_size, font_size),
        yaxis=get_axis(y_label, font_size, font_size),
    )


def get_axis(title, title_size, tick_size):
    axis = dict(
        title=title,
        autorange=True,
        showgrid=True,
        zeroline=False,
        linecolor='black',
        showline=True,
        gridcolor='gainsboro',
        gridwidth=0.001,
        mirror="allticks",
        ticks='outside',
        titlefont=dict(
            color='black',
            size=title_size
        ),
        showticklabels=True,
        tickangle=0,
        tickfont=dict(
            color='black',
            size=tick_size
        ),
        exponentformat='e',
        showexponent='all'
    )
    return axis