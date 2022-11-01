import plotly
import time
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio

def save_figure(fig, fn, width=800, height=600, scale=2):
    #py.plot(fig, filename=f"{fn}.png", include_mathjax='cdn')
    #py.plot(fig, filename=f"{fn}.pdf", include_mathjax='cdn')
    fig.write_image(f"{fn}.png")
    fig.write_image(f"{fn}.pdf", format="pdf")
    #fig.write_image(f"{fn}.pdf", format="pdf")
    #plotly.io.write_image(fig, f"{fn}.png", width=width, height=height, scale=scale)
    #plotly.io.write_image(fig, f"{fn}.pdf", width=width, height=height, scale=scale)
