import os
from shutil import rmtree, move
from subprocess import call

import numpy as np
import vigra
import nifty.graph.rag as nrag


def get_edges_with_weights(uv_ids, edge_weights):
    edge_str = ""
    for edge_id in range(uv_ids.shape[0]):
        u = uv_ids[edge_id, 0]
        v = uv_ids[edge_id, 1]
        repulsive = edge_weights[edge_id] > 0.5
        if repulsive:
            edge_str += "\\draw (n%d) edge[in_plane_edge,red] (n%d);\n" % (u, v)
        else:
            edge_str += "\\draw (n%d) edge[in_plane_edge,green] (n%d);\n" % (u, v)
    return edge_str


def get_edges(uv_ids):
    edge_str = ""
    for edge_id in range(uv_ids.shape[0]):
        u = uv_ids[edge_id, 0]
        v = uv_ids[edge_id, 1]
        edge_str += "\\draw (n%d) edge[in_plane_edge,black] (n%d);\n" % (u, v)
    return edge_str


def get_nodes(seg):
    nodes = np.unique(seg)
    centers = vigra.filters.eccentricityCenters(seg)

    sx = seg.shape[0]
    sy = seg.shape[1]

    node_str = ""
    # generate the nodes
    for node_id in nodes:
        coord = np.round(centers[node_id], 2)
        x = float(coord[0])/float(sx)
        y = float(coord[1])/float(sy)
        y = 1.0 - y
        node_str += "\\node[rag_node] at (%f, %f) (n%d){};\n" % (x, y, node_id)
    return node_str


def compute_edge_weights(rag, inp):
    weights = nrag.accumulateEdgeMeanAndLength(rag, inp)
    return weights[:, 0]


def seg_to_region_graph(image, seg, out_path, edge_weights=None):
    assert image.shape == seg.shape
    rag = nrag.gridRag(seg, numberOfLabels=int(seg.max()) + 1)

    # generate the nodes
    node_str = get_nodes(seg)

    # generate the edges
    uv_ids = rag.uvIds()
    if edge_weights is None:
        if edge_weights.shape == seg.shape:
            edge_weights = compute_edge_weights(rag, edge_weights)
        assert len(edge_weights) == rag.numberOfEdges
        edge_str = get_edges(uv_ids)
    else:
        edge_str = get_edges_with_weights(uv_ids, edge_weights)

    # write the tex files
    os.mkdir('./tmp_tex')
    with open("tmp_tex/nodes.tex", "w") as text_file:
        text_file.write(node_str)
    with open("tmp_tex/edges.tex", "w") as text_file:
        text_file.write(edge_str)

    # save the image
    vigra.impex.writeImage(image, 'tmp_tex/im.png')

    # compile tikz
    this_folder = os.path.split(os.path.realpath(__file__))[0]
    tikz_script = os.path.join(this_folder, 'seg_graph.tex')
    call(['pdflatex', tikz_script])

    # move the output file
    out_tmp = ''  # TODO
    move(out_tmp, out_path)

    # clean up
    rmtree('tmp_tex')
