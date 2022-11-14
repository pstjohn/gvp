from torch_gvp.data.transforms import NodeOrientation, create_gvp_transformer_stack


def test_node_orientation(prot_data):
    xform = NodeOrientation()
    data_xform = xform(prot_data)
    assert data_xform.node_v.shape[-2:] == (2, 3)


def test_transformer_stack(prot_data):
    xform = create_gvp_transformer_stack()
    data_xform = xform(prot_data)

    edge_s, edge_v = prot_data.edge_attr
    prot_data.cpu()

    assert data_xform.node_v.shape[-2:] == (2, 3)
    assert edge_s.shape[-1] == 1
    assert edge_v.shape[-2:] == (1, 3)
