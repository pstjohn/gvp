from gvp.data.transforms import NodeOrientation, create_gvp_transformer_stack


def test_node_orientation(data):
    xform = NodeOrientation()
    data_xform = xform(data)
    assert data_xform.node_v.shape[-2:] == (2, 3)


def test_transformer_stack(data):
    xform = create_gvp_transformer_stack()
    data_xform = xform(data)
    assert data_xform.node_v.shape[-2:] == (2, 3)
    assert data_xform.edge_s.shape[-1] == 1
    assert data_xform.edge_v.shape[-2:] == (1, 3)
