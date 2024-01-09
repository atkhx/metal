package pipeline

import (
	"github.com/atkhx/metal/nn/num"
)

type Nodes []*num.Data

type NodeLayers []Nodes

func getDistinctNodes(aData *num.Data) Nodes {
	nodes := Nodes{}
	added := map[*num.Data]struct{}{}

	var addFunc func(node *num.Data)

	addFunc = func(node *num.Data) {
		for _, srcNode := range node.Deps {
			if _, ok := added[srcNode]; !ok {
				addFunc(srcNode)
			}
		}

		if _, ok := added[node]; !ok {
			added[node] = struct{}{}
			nodes = append(nodes, node)
		}
	}

	addFunc(aData)
	return nodes
}

func getNodeLayers(aData *num.Data, skipNodeFn func(*num.Data) bool) NodeLayers {
	nodeList := getDistinctNodes(aData)
	notAdded := len(nodeList)

	nodeLayers := NodeLayers{}
	nodeLevel := map[*num.Data]int{}

	getMaxSrcNodeLevel := func(srcNodes Nodes) int {
		result := -1
		for _, node := range srcNodes {
			if nodeLevel[node] > result {
				result = nodeLevel[node]
			}
		}
		return result
	}

	for notAdded > 0 {
		for _, node := range nodeList {
			if _, nodeProcessed := nodeLevel[node]; nodeProcessed {
				continue
			}
			notAdded--

			if skipNodeFn != nil && skipNodeFn(node) {
				nodeLevel[node] = -1
				continue
			}

			level := getMaxSrcNodeLevel(node.Deps) + 1
			if len(nodeLayers) == level {
				nodeLayers = append(nodeLayers, Nodes{})
			}

			nodeLayers[level] = append(nodeLayers[level], node)
			nodeLevel[node] = level
		}
	}
	return nodeLayers
}

func getForwardNodeLayers(aData *num.Data) NodeLayers {
	return getNodeLayers(aData, func(node *num.Data) bool {
		return node.CalcData == nil
	})
}

func getBackwardNodeLayers(aData *num.Data) NodeLayers {
	nodes := getNodeLayers(aData, func(node *num.Data) bool {
		return node.CalcGrad == nil
	})

	bwdNodes := make(NodeLayers, 0, len(nodes))
	for i := len(nodes); i > 0; i-- {
		bwdNodes = append(bwdNodes, nodes[i-1])
	}
	return bwdNodes
}

func getResetGradsNodes(aData *num.Data) Nodes {
	nodes := getDistinctNodes(aData)

	var result Nodes
	for i := len(nodes); i > 0; i-- {
		if nodes[i-1].SkipResetGrad {
			continue
		}
		result = append(result, nodes[i-1])
	}
	return result
}
