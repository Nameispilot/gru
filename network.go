package gru

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type network struct {
	gru GRU

	// embedding matrix
	embedding *gorgonia.Node

	// decoder
	whd    *gorgonia.Node
	bias_d *gorgonia.Node

	// previous node simulation
	prev *gorgonia.Node

	// cost node
	cost *gorgonia.Node
}

func MakeNetwork(g *gorgonia.ExprGraph, inputSize, embeddingSize, outputSize, hiddenSize int) *network {

	n := new(network)
	n.gru = makeGRU(g, embeddingSize, hiddenSize)

	n.embedding = gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(inputSize, embeddingSize), gorgonia.WithInit(gorgonia.GlorotN(0.8)), gorgonia.WithName("embedding_"))
	n.whd = gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(outputSize, hiddenSize), gorgonia.WithInit(gorgonia.GlorotN(0.8)), gorgonia.WithName("whd_"))
	n.bias_d = gorgonia.NewVector(g, tensor.Float32, gorgonia.WithShape(outputSize), gorgonia.WithInit(gorgonia.GlorotN(0.08)), gorgonia.WithName("bias_d_"))

	// this is to simulate previous state
	n.prev = gorgonia.NewVector(g, tensor.Float32, gorgonia.WithShape(hiddenSize), gorgonia.WithName("prev_"), gorgonia.WithInit(gorgonia.Zeroes()))
	return n
}

func (n *network) Learnables() gorgonia.Nodes {
	retVal := make(gorgonia.Nodes, 0)
	retVal = append(retVal, n.gru.learnables()...)
	retVal = append(retVal, n.embedding)
	retVal = append(retVal, n.whd)
	retVal = append(retVal, n.bias_d)
	return retVal
}

type GRUOut struct {
	hidden, prob *gorgonia.Node
}

func (n *network) fwd(sourceIdx int, prev *GRUOut) (*GRUOut, error) {
	var prevState *gorgonia.Node
	if prev == nil {
		prevState = n.prev
	} else {
		prevState = prev.hidden
	}

	var inputVector, hidden *gorgonia.Node
	var err error
	if inputVector, err = gorgonia.Slice(n.embedding, gorgonia.S(sourceIdx)); err != nil {
		return nil, errors.Wrap(err, "Can't slice embedding matrix")
	}
	if hidden, err = n.gru.activate(inputVector, prevState); err != nil {
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

func (n *network) Fwd(input *gorgonia.Node, vocabIndex map[int]int) error {
	var sentence []int
	var err error
	if input.Type().String() == "Vector int" {
		sentence = input.Value().Data().([]int)
	} else {
		return errors.Wrap(err, "Input vector is not' 'Int'")
	}

	var source, target int
	var prev *GRUOut
	var cost *gorgonia.Node
	for i := 0; i < len(sentence)-1; i++ {
		source = sentence[i]
		target = sentence[i+1]
		sourceId := vocabIndex[source]
		targetId := vocabIndex[target]

		if prev, err = n.fwd(sourceId, prev); err != nil {
			return errors.Wrap(err, "Can't feed forward")
		}

		if cost, err = costLSTM(prev.prob, cost, targetId); err != nil {
			return errors.Wrap(err, "Can't calculate cost")
		}
	}

	n.cost = cost
	return nil
}

func costLSTM(prob, cost *gorgonia.Node, targetId int) (*gorgonia.Node, error) {
	logprob, err := gorgonia.Log(prob)
	if err != nil {
		return nil, errors.Wrap(err, "...")
	}
	logprob, err = gorgonia.Neg(logprob)
	if err != nil {
		return nil, errors.Wrap(err, "...")
	}
	loss, err := gorgonia.Slice(logprob, gorgonia.S(targetId))
	if err != nil {
		return nil, errors.Wrap(err, "...")
	}
	if cost == nil {
		cost = loss
	} else {
		cost, err = gorgonia.Add(cost, loss)
		if err != nil {
			return nil, errors.Wrap(err, "...")
		}
	}

	return cost, nil
}

func (n *network) GetCost() (*gorgonia.Node, error) {
	if n.cost != nil {
		return n.cost, nil
	} else {
		return nil, fmt.Errorf("Cost node is nil")
	}
}

func (n *network) Predict(g *gorgonia.ExprGraph, input *gorgonia.Node, vocabIndex map[int]int) (int, error) {
	var sentence []int
	var err error
	if input.Type().String() == "Vector int" {
		sentence = input.Value().Data().([]int)
	} else {
		return 0, errors.Wrap(err, "Input vector is not' 'Int'")
	}

	var id int
	if len(sentence) > 0 {
		id = vocabIndex[sentence[len(sentence)-1]]
	}

	var prev *GRUOut
	if prev, err = n.fwd(id, prev); err != nil {
		return 0, errors.Wrap(err, "Can't forward")
	}

	tm := gorgonia.NewTapeMachine(g)
	defer tm.Close()

	err = tm.RunAll()
	if err != nil {
		return 0, errors.Wrap(err, "Can't run the machine")
	}

	sampledId := sample(prev.prob.Value())

	return sampledId, nil
}

func sample(val gorgonia.Value) int {

	var t tensor.Tensor
	var ok bool
	if t, ok = val.(tensor.Tensor); !ok {
		panic("expects a tensor")
	}
	indT, err := tensor.Argmax(t, -1)
	if err != nil {
		panic(err)
	}
	if !indT.IsScalar() {
		panic("Expected scalar index")
	}
	return indT.ScalarValue().(int)
}
