import streamlit as st
import numpy as np
import plotly.graph_objs as go

# Generate some fake data
ap_pos = np.array([[100, 200], [300, 400], [500, 600]])
ue_pos = np.array([[150, 250], [350, 450], [550, 650]])

st.write("AP positions:", ap_pos)
st.write("UE positions:", ue_pos)

traces = [
    go.Scatter(
        x=ap_pos[:, 0], y=ap_pos[:, 1],
        mode='markers',
        marker=dict(color='red', size=14),
        name='APs'
    ),
    go.Scatter(
        x=ue_pos[:, 0], y=ue_pos[:, 1],
        mode='markers',
        marker=dict(color='blue', size=10),
        name='UEs'
    )
]

layout = go.Layout(
    width=600, height=600,
    xaxis=dict(range=[0, 1000], title='X (m)', automargin=True),
    yaxis=dict(range=[0, 1000], title='Y (m)', automargin=True),
    legend=dict(x=0, y=1),
    margin=dict(l=40, r=40, t=40, b=40),
)

fig = go.Figure(data=traces, layout=layout)
st.plotly_chart(fig) 