import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

def add_p_value_annotation(fig, array_dict, subplot=None, _format=dict(interline=0.07, text_height=1.07, color='black')):
    ''' Adds notations giving the p-value between two box plot data (t-test two-sided comparison)

    Parameters:
    ----------
    fig: figure
        plotly boxplot figure
    array_columns: np.array
        array of which columns to compare
        e.g.: [[0,1], [1,2]] compares column 0 with 1 and 1 with 2
    subplot: None or int
        specifies if the figures has subplots and what subplot to add the notation to
    _format: dict
        format characteristics for the lines

    Returns:
    -------
    fig: figure
        figure with the added notation
    '''
    # Specify in what y_range to plot for each pair of columns
    y_range = np.zeros([len(array_dict), 2])
    for i in range(len(array_dict)):
        y_range[i] = [1.01 + i * _format['interline'], 1.02 + i * _format['interline']]

    # Get values from figure
    fig_dict = fig.to_dict()

    # Get indices if working with subplots
    if subplot:
        if subplot == 1:
            subplot_str = ''
        else:
            subplot_str = str(subplot)
        indices = []  # Change the box index to the indices of the data for that subplot
        for index, data in enumerate(fig_dict['data']):
            # print(index, data['xaxis'], 'x' + subplot_str)
            if data['xaxis'] == 'x' + subplot_str:
                indices = np.append(indices, index)
        indices = [int(i) for i in indices]
        print((indices))
    else:
        subplot_str = ''

    # Print the p-values
    for index, column_pair in enumerate(array_dict):
        if subplot:
            data_pair = [indices[column_pair[0]], indices[column_pair[1]]]
        else:
            data_pair = column_pair

        # Mare sure it is selecting the data and subplot you want
        # print('0:', fig_dict['data'][data_pair[0]]['name'], fig_dict['data'][data_pair[0]]['xaxis'])
        # print('1:', fig_dict['data'][data_pair[1]]['name'], fig_dict['data'][data_pair[1]]['xaxis'])

        # Get the p-value
        pvalue = array_dict[column_pair]
        symbol = f'{pvalue:0.2e}'
        # Vertical line
        fig.add_shape(type="line",
                      xref="x" + subplot_str, yref="y" + subplot_str + " domain",
                      x0=column_pair[0], y0=y_range[index][0],
                      x1=column_pair[0], y1=y_range[index][1],
                      line=dict(color=_format['color'], width=2, )
                      )
        # Horizontal line
        fig.add_shape(type="line",
                      xref="x" + subplot_str, yref="y" + subplot_str + " domain",
                      x0=column_pair[0], y0=y_range[index][1],
                      x1=column_pair[1], y1=y_range[index][1],
                      line=dict(color=_format['color'], width=2, )
                      )
        # Vertical line
        fig.add_shape(type="line",
                      xref="x" + subplot_str, yref="y" + subplot_str + " domain",
                      x0=column_pair[1], y0=y_range[index][0],
                      x1=column_pair[1], y1=y_range[index][1],
                      line=dict(color=_format['color'], width=2, )
                      )
        ## add text at the correct x, y coordinates
        ## for bars, there is a direct mapping from the bar number to 0, 1, 2...
        fig.add_annotation(dict(font=dict(color=_format['color'], size=14),
                                x=(column_pair[0] + column_pair[1]) / 2,
                                y=y_range[index][1] * _format['text_height'],
                                showarrow=False,
                                text=symbol,
                                textangle=0,
                                xref="x" + subplot_str,
                                yref="y" + subplot_str + " domain"
                                ))
    return fig