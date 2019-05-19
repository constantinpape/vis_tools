import os
from shutil import rmtree, move, copy2
from subprocess import call

import numpy as np
import vigra
from scipy.ndimage.measurements import center_of_mass
import nifty.graph.rag as nrag


def get_nodes(seg, exclude_nodes, node_style):
    nodes = np.unique(seg)
    # centers = vigra.filters.eccentricityCenters(seg)
    centers = {node_id: center_of_mass(seg == node_id) for node_id in nodes}

    sx = seg.shape[0]
    sy = seg.shape[1]

    node_str = ""
    # generate the nodes
    for node_id in nodes:

        if node_id in exclude_nodes:
            continue

        coord = np.round(centers[node_id], 2)
        x = float(coord[0])/float(sx)
        y = float(coord[1])/float(sy)
        y = 1.0 - y
        node_str += "\\node[%s] at (%f, %f) (n%d){};\n" % (node_style, x, y, node_id)
    return node_str


def get_edges_with_weights(uv_ids, edge_weights, exclude_nodes, edge_threshold,
                           edge_style):
    style, _ = edge_style
    edge_str = ""
    for edge_id in range(uv_ids.shape[0]):
        u = uv_ids[edge_id, 0]
        v = uv_ids[edge_id, 1]
        if u in exclude_nodes or v in exclude_nodes:
            continue

        repulsive = edge_weights[edge_id] > edge_threshold
        if repulsive:
            edge_str += "\\draw (n%d) edge[%s,red] (n%d);\n" % (u, style, v)
        else:
            edge_str += "\\draw (n%d) edge[%s,green] (n%d);\n" % (u, style, v)
    return edge_str


def get_edges_without_weights(uv_ids, exclude_nodes, edge_style):
    edge_str = ""
    style, color = edge_style
    for edge_id in range(uv_ids.shape[0]):
        u = uv_ids[edge_id, 0]
        v = uv_ids[edge_id, 1]
        if u in exclude_nodes or v in exclude_nodes:
            continue

        edge_str += "\\draw (n%d) edge[%s,%s] (n%d);\n" % (u, style, color, v)
    return edge_str


def compute_edge_weights(rag, inp):
    weights = nrag.accumulateEdgeMeanAndLength(rag, inp)
    return weights[:, 0]


def get_edges(rag, edge_weights, exclude_nodes, edge_threshold,
              edge_style):
    uv_ids = rag.uvIds()
    if edge_weights is None:
        edge_str = get_edges_without_weights(uv_ids, exclude_nodes, edge_style)
    else:
        if edge_weights.shape == tuple(rag.shape):
            edge_weights = compute_edge_weights(rag, edge_weights)
        assert len(edge_weights) == rag.numberOfEdges
        edge_str = get_edges_with_weights(uv_ids, edge_weights,
                                          exclude_nodes, edge_threshold,
                                          edge_style)
    return edge_str


def compile_tikz(tikz_script, out_path):
    # compile tikz
    this_folder = os.path.split(os.path.realpath(__file__))[0]
    tikz_script_in = os.path.join(this_folder, '%s.tex' % tikz_script)
    tikz_script_out = './tmp_tex/%s.tex' % tikz_script
    copy2(tikz_script_in, tikz_script_out)

    pwd = os.getcwd()
    os.chdir('tmp_tex')
    call(['pdflatex', '%s.tex' % tikz_script])
    os.chdir(pwd)

    # move the output file
    out_tmp = 'tmp_tex/%s.pdf' % tikz_script
    move(out_tmp, out_path)

    # clean up
    rmtree('tmp_tex')


# \tikzstyle{rag_node} = [fill=white,draw=yellow,circle,inner sep=0pt,minimum size=5pt,very thin,scale=0.05]
def get_node_style(key='default'):
    styles = {'default': {'fill': 'white', 'draw': 'black', 'inner sep': '0pt',
                          'minimum size': '5pt', 'scale': '0.05', 'circle': None, 'very thin': None}}
    return styles[key]


# \tikzstyle{in_plane_edge} = [line width=0.1pt,scale=0.05]
def get_edge_style(key='default'):
    styles = {'default': {'line width': '0.1pt', 'scale': '0.05', 'opacity': '1',
                          'color': 'black'}}
    return styles[key]


# \tikzstyle{lifted_edge} = [line width=0.4pt,scale=0.15,densely dashdotted, dash pattern=on \pgflinewidth off .1mm]
def get_lifted_edge_style(key='default'):
    styles = {'default': {'line width': '0.4pt', 'scale': '0.15',
                          'color': 'black', 'densely dashdotted': None,
                          'dash pattern': 'on \\pgflinewidth off .1mm'}}
    return styles[key]


def style_to_string(style, return_color=False):
    if return_color:
        color = style.pop('color', 'black')
        return ",".join(k if v is None else "%s=%s" % (k, v) for k, v in style.items()), color
    else:
        return ",".join(k if v is None else "%s=%s" % (k, v) for k, v in style.items())


def seg_to_region_graph(image, seg, out_path, edge_weights=None, exclude_nodes=[],
                        edge_threshold=.5, node_style=get_node_style(), edge_style=get_edge_style()):
    if image.ndim == 3:
        assert image.shape[:-1] == seg.shape, "%s, %s" % (str(image.shape), str(seg.shape))
    else:
        assert image.shape == seg.shape, "%s, %s" % (str(image.shape), str(seg.shape))
    shape = seg.shape
    assert shape[0] == shape[1], "Only works for square shapes"
    rag = nrag.gridRag(seg, numberOfLabels=int(seg.max()) + 1)

    # generate the nodes
    node_str = get_nodes(seg, exclude_nodes,
                         style_to_string(node_style))
    # generate the edges
    edge_str = get_edges(rag, edge_weights, exclude_nodes, edge_threshold,
                         style_to_string(edge_style, return_color=True))

    # write the tex files
    os.makedirs('./tmp_tex', exist_ok=True)
    with open("tmp_tex/nodes.tex", "w") as text_file:
        text_file.write(node_str)
    with open("tmp_tex/edges.tex", "w") as text_file:
        text_file.write(edge_str)

    # save the image
    vigra.impex.writeImage(image, 'tmp_tex/im.png')
    compile_tikz('seg_to_graph', out_path)


def get_lifted_edges_without_weights(lifted_ids, edge_style):
    style, color = edge_style
    lifted_edge_str = ''
    for edge_id in range(lifted_ids.shape[0]):
        u = lifted_ids[edge_id, 0]
        v = lifted_ids[edge_id, 1]
        lifted_edge_str += "\\draw (n%d) edge[%s,%s] (n%d);\n" % (u, style, color, v)
    return lifted_edge_str


def get_lifted_edges_with_weights(lifted_ids, lifted_weights, edge_threshold, edge_style):
    lifted_edge_str = ''
    style, _ = edge_style
    for edge_id in range(lifted_ids.shape[0]):
        u = lifted_ids[edge_id, 0]
        v = lifted_ids[edge_id, 1]
        repulsive = lifted_weights[edge_id] > edge_threshold
        if repulsive:
            lifted_edge_str += "\\draw (n%d) edge[%s,red] (n%d);\n" % (u, style, v)
        else:
            lifted_edge_str += "\\draw (n%d) edge[%s,green] (n%d);\n" % (u, style, v)
    return lifted_edge_str


def get_lifted_edges(lifted_ids, lifted_weights, edge_threshold, edge_style):
    if lifted_weights is None:
        lifted_edge_str = get_lifted_edges_without_weights(lifted_ids, edge_style)
    else:
        assert len(lifted_ids) == len(lifted_weights)
        lifted_edge_str = get_lifted_edges_with_weights(lifted_ids, lifted_weights, edge_threshold, edge_style)
    return lifted_edge_str


def lifted_graph(image, seg, lifted_ids, out_path, lifted_weights=None, edge_weights=None,
                 exclude_nodes=[], edge_threshold=.5, node_style=get_node_style(),
                 edge_style=get_edge_style(), lifted_edge_style=get_lifted_edge_style()):
    if image.ndim == 3:
        assert image.shape[:-1] == seg.shape, "%s, %s" % (str(image.shape), str(seg.shape))
    else:
        assert image.shape == seg.shape, "%s, %s" % (str(image.shape), str(seg.shape))
    shape = seg.shape
    assert shape[0] == shape[1], "Only works for square shapes"
    rag = nrag.gridRag(seg, numberOfLabels=int(seg.max()) + 1)

    # generate the nodes
    node_str = get_nodes(seg, exclude_nodes,
                         style_to_string(node_style))

    # generate the edges
    edge_str = get_edges(rag, edge_weights, exclude_nodes, edge_threshold,
                         style_to_string(edge_style, return_color=True))

    # generate the lifted edges
    lifted_edge_str = get_lifted_edges(lifted_ids, lifted_weights, edge_threshold,
                                       style_to_string(lifted_edge_style, return_color=True))

    # write the tex files
    os.makedirs('./tmp_tex', exist_ok=True)
    with open("tmp_tex/nodes.tex", "w") as text_file:
        text_file.write(node_str)
    with open("tmp_tex/edges.tex", "w") as text_file:
        text_file.write(edge_str)
    with open("tmp_tex/lifted_edges.tex", "w") as text_file:
        text_file.write(lifted_edge_str)

    # save the image
    vigra.impex.writeImage(image, 'tmp_tex/im.png')
    compile_tikz('lifted_graph', out_path)
