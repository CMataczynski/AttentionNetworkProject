import datetime

import base64
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import plotly.graph_objects as go
import dash_html_components as html
import scipy.misc

import numpy as np
from io import BytesIO
from PIL import Image
from frontend_utils import Predictor

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    dcc.Store(id='my-store', storage_type="memory", data={'img': None,
                                                          'img_heat': None,
                                                          'prob': 0}),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Przeciągnij i upuść lub ',
            html.A('Wybierz plik')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div([html.H5('Prawdopodobieństwo Zwłóknienia')],
             style={
                 'width': '100%',
                 'textAlign': 'center',
                 'margin': '10px'
             }
             ),
    dcc.Slider(
        id='slider-score',
        marks={i: '{}%'.format(i) for i in range(0,101,10)},
        min=0,
        max=100,
        value=0,
        step=0.1,
        disabled=True,
        tooltip={"always_visible": True,
                 "placement": "bottom"}
    ),
    html.Div([
        dcc.Tabs(id="tabs", value='tab-1', children=[
            dcc.Tab(label='Oryginał', value='tab-1'),
            dcc.Tab(label='Powody predykcji', value='tab-2'),
        ]),
        html.Div(id='tabs-content')
    ]),
])

app.config['suppress_callback_exceptions'] = True

predict = Predictor()

def b64_to_pil(string):
    decoded = base64.b64decode(string)
    buffer = BytesIO(decoded)
    im = Image.open(buffer)

    return im
def b64_to_numpy(string, to_scalar=True):
    im = b64_to_pil(string)
    np_array = np.asarray(im)

    if to_scalar:
        np_array = np_array / 255.

    return np_array

def pil_to_b64(im, enc_format='png', verbose=False, **kwargs):
    """
    Converts a PIL Image into base64 string for HTML displaying
    :param im: PIL Image object
    :param enc_format: The image format for displaying. If saved the image will have that extension.
    :return: base64 encoding
    """
    buff = BytesIO()
    im.save(buff, format=enc_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")
    return encoded


def numpy_to_b64(np_array, enc_format='png', scalar=True, **kwargs):
    """
    Converts a numpy image into base 64 string for HTML displaying
    :param np_array:
    :param enc_format: The image format for displaying. If saved the image will have that extension.
    :param scalar:
    :return:
    """
    # Convert from 0-1 to 0-255
    if scalar:
        np_array = np.uint8(255 * np_array)
    else:
        np_array = np.uint8(np_array)

    im_pil = Image.fromarray(np_array)

    return pil_to_b64(im_pil, enc_format, **kwargs)



def parse_contents(contents):
    return html.Div([
        html.Img(src=contents),
        html.Hr(),
    ])


@app.callback([Output('my-store', 'data')], [Input('upload-image', 'contents')], [State('my-store', 'data')])
def update_output(list_of_contents, data):
    #np_img = np.frombuffer(base64.b64decode(list_of_contents.split(",")[1]), np.uint8)
    if list_of_contents is not None:
        start, end = list_of_contents.split(',')
        img = b64_to_pil(end)
        crop_horizontal = 250
        crop_vertical = 10
        img_size = img.size
        box = (crop_horizontal, crop_vertical, img_size[0] - crop_horizontal, img_size[1] - crop_vertical)
        img = img.crop(box)
        img = np.array(img)
        predict.predict(img)
        img_heat = predict.getGradcamImage()
        print(img_heat.max())
        data['img'] = start+","+numpy_to_b64(img, scalar=False)
        data['img_heat'] = start+","+numpy_to_b64(img_heat,scalar=False)
        data['prob'] = predict.getPrediction()
        return [data]
    return [{'img': None,
             'img_heat': None,
             'prob': 0}]


@app.callback([Output('slider-score', 'value'),
               Output('tabs-content', 'children')],
              [Input('tabs', 'value'),
               Input('my-store', 'modified_timestamp')],
              [State('my-store','data')])
def render_content(tab, ts, data):
    if tab == 'tab-1':
        return [data['prob'],
                html.Div(
                    [html.Img(src=data['img']),
                    html.Hr()],
                    id='output-image-upload',
                    style={
                        'display': 'table',
                        'height': '200px',
                        'object-fit': 'contain',
                        'margin-right': 'auto',
                        'margin-left': 'auto'
                    })]
    elif tab == 'tab-2':
        return [data['prob'],
                html.Div(
                    [html.Img(src=data['img_heat']),
                    html.Hr()],
                    id='output-image-heat-upload',
                    style={
                        'display': 'table',
                        'height': '200px',
                        'object-fit': 'contain',
                        'margin-right': 'auto',
                        'margin-left': 'auto'
                    }
                )]


if __name__ == '__main__':
    app.run_server(debug=False)