# -*- coding: utf-8 -*-
import pytest
import os
from hccf.clustering import FeatureClustering, Node
from hccf.utils.helpers import silentremove


class TestNode(object):

    def test_node_get_leaves_parents(self):
        node = Node(
            value=1,
            left=Node(
                value=2,
                left=Node(4),
                right=Node(5),
            ),
            right=Node(3),
        )
        leaves_parents = node.get_leaves_parents()
        assert leaves_parents == {
            3: [1],
            4: [1, 2],
            5: [1, 2],
        }


class TestFeatureClustering(object):
    def test_load_save(self):
        output_file = 'test_clustering_save.dat'
        fc = FeatureClustering()
        fc.save(output_file)
        assert os.path.isfile(output_file)

        fc_loaded = FeatureClustering.load(output_file)
        assert isinstance(fc_loaded, FeatureClustering)
        assert fc_loaded.trees == fc.trees
        silentremove(output_file)

    def test_convert(self):
        pass

    def test_cluster(self):
        pass
