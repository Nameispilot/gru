package gru

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type network struct {
	gru GRU

	//embedding matrix
	embedding *gorgonia.Node

	//decoder
	whd    *gorgonia.Node
	bias_d *gorgonia.Node

	//previous node simulation
	prev *gorgonia.Node
}

func MakeNetwork(g *gorgonia.ExprGraph, inputSize, embeddingSize, outputSize, hiddenSize int) *network {

	n := new(network)
	n.gru = MakeGRU(g, embeddingSize, hiddenSize)

	n.embedding = gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(inputSize, embeddingSize), gorgonia.WithInit(gorgonia.GlorotN(0.8)), gorgonia.WithName("embedding_"))
	n.whd = gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(outputSize, hiddenSize), gorgonia.WithInit(gorgonia.GlorotN(0.8)), gorgonia.WithName("whd_"))
	n.bias_d = gorgonia.NewVector(g, tensor.Float32, gorgonia.WithShape(outputSize), gorgonia.WithInit(gorgonia.GlorotN(0.08)), gorgonia.WithName("bias_d_"))

	// this is to simulate previous state
	n.prev = gorgonia.NewVector(g, tensor.Float32, gorgonia.WithShape(hiddenSize))
	return n
}

func (n *network) Learnables() gorgonia.Nodes {
	retVal := make(gorgonia.Nodes, 0)
	retVal = append(retVal, n.gru.Learnables()...)
	retVal = append(retVal, n.embedding)
	retVal = append(retVal, n.whd)
	retVal = append(retVal, n.bias_d)
	return retVal
}

type GRUOut struct {
	hidden, prob *gorgonia.Node
}

func (n *network) Fwd(sourceIdx int, prev *gorgonia.Node) (*GRUOut, error) {
	var prevState *gorgonia.Node
	if prev == nil {
		prevState = n.prev
	} else {
		prevState = prev
	}

	var inputVector, hidden *gorgonia.Node
	var err error
	if inputVector, err = gorgonia.Slice(n.embedding, gorgonia.S(sourceIdx)); err != nil {
		return nil, errors.Wrap(err, "Can't slice embedding matrix")
	}
	if hidden, err = n.gru.Activate(inputVector, prevState); err != nil {
		return nil, errors.Wrap(err, "Can't activate gru layer")
	}

	var output, prob *gorgonia.Node
	if output, err = gorgonia.Mul(n.whd, hidden); err != nil {
		return nil, errors.Wrap(err, "Can't multiplicate output weights and hidden")
	}
	if output, err = gorgonia.Add(output, n.bias_d); err != nil {
		return nil, errors.Wrap(err, "Can't add biases to output")
	}
	if prob, err = gorgonia.SoftMax(output); err != nil {
		return nil, errors.Wrap(err, "Can't softmax output")
	}

	retVal := &GRUOut{
		hidden: hidden,
		prob:   prob,
	}
	return retVal, nil

}
