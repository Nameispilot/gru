package gru

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type GRU struct {

	// weights for memory
	u *gorgonia.Node
	w *gorgonia.Node
	b *gorgonia.Node

	// update gate
	uz *gorgonia.Node
	wz *gorgonia.Node
	bz *gorgonia.Node

	// reset gate
	ur  *gorgonia.Node
	wr  *gorgonia.Node
	br  *gorgonia.Node
	one *gorgonia.Node
}

func makeGRU(g *gorgonia.ExprGraph, inputSize, hiddenSize int) GRU {
	retVal := GRU{}

	// weights for memory
	retVal.u = gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(hiddenSize, hiddenSize), gorgonia.WithInit(gorgonia.GlorotN(1.0)), gorgonia.WithName("u_"))
	retVal.w = gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(hiddenSize, inputSize), gorgonia.WithInit(gorgonia.GlorotN(1.0)), gorgonia.WithName("w_"))
	retVal.b = gorgonia.NewVector(g, tensor.Float32, gorgonia.WithShape(hiddenSize), gorgonia.WithName("b_"), gorgonia.WithInit(gorgonia.Zeroes()))

	// update gate
	retVal.uz = gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(hiddenSize, hiddenSize), gorgonia.WithInit(gorgonia.GlorotN(1.0)), gorgonia.WithName("uz_"))
	retVal.wz = gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(hiddenSize, inputSize), gorgonia.WithInit(gorgonia.GlorotN(1.0)), gorgonia.WithName("uw_"))
	retVal.bz = gorgonia.NewVector(g, tensor.Float32, gorgonia.WithShape(hiddenSize), gorgonia.WithName("bz_"), gorgonia.WithInit(gorgonia.Zeroes()))

	retVal.ur = gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(hiddenSize, hiddenSize), gorgonia.WithInit(gorgonia.GlorotN(1.0)), gorgonia.WithName("ur_"))
	retVal.wr = gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(hiddenSize, inputSize), gorgonia.WithInit(gorgonia.GlorotN(1.0)), gorgonia.WithName("wr_"))
	retVal.br = gorgonia.NewVector(g, tensor.Float32, gorgonia.WithShape(hiddenSize), gorgonia.WithName("br_"), gorgonia.WithInit(gorgonia.Zeroes()))

	// just a number for calculation
	ones := tensor.Ones(tensor.Float32, hiddenSize)
	retVal.one = g.Constant(ones)

	return retVal
}

func (gru *GRU) Learnables() gorgonia.Nodes {
	return gorgonia.Nodes{
		gru.u, gru.w, gru.b,
		gru.uz, gru.wz, gru.bz,
		gru.ur, gru.wr, gru.br,
	}
}

func (gru *GRU) Activate(input, prev *gorgonia.Node) (*gorgonia.Node, error) {
	// update gate
	var uzh, wzh, z *gorgonia.Node
	var err error

	if uzh, err = gorgonia.Mul(gru.uz, prev); err != nil {
		return nil, errors.Wrap(err, "...")
	}
	if wzh, err = gorgonia.Mul(gru.wz, input); err != nil {
		return nil, errors.Wrap(err, "...")
	}
	if z, err = gorgonia.Add(uzh, wzh); err != nil {
		return nil, errors.Wrap(err, "...")
	}
	if z, err = gorgonia.Add(z, gru.bz); err != nil {
		return nil, errors.Wrap(err, "...")
	}
	if z, err = gorgonia.Sigmoid(z); err != nil {
		return nil, errors.Wrap(err, "...")
	}

	// reset gate
	var urh, wrx, r *gorgonia.Node
	if urh, err = gorgonia.Mul(gru.ur, prev); err != nil {
		return nil, errors.Wrap(err, "...")
	}
	if wrx, err = gorgonia.Mul(gru.wr, input); err != nil {
		return nil, errors.Wrap(err, "...")
	}
	if r, err = gorgonia.Add(urh, wrx); err != nil {
		return nil, errors.Wrap(err, "...")
	}
	if r, err = gorgonia.Add(r, gru.br); err != nil {
		return nil, errors.Wrap(err, "...")
	}
	if r, err = gorgonia.Sigmoid(r); err != nil {
		return nil, errors.Wrap(err, "...")
	}

	// memory for hidden
	var filter, wx, mem *gorgonia.Node
	if filter, err = gorgonia.HadamardProd(r, prev); err != nil {
		return nil, errors.Wrap(err, "...")
	}
	if filter, err = gorgonia.Mul(filter, gru.u); err != nil {
		return nil, errors.Wrap(err, "...")
	}
	if wx, err = gorgonia.Mul(gru.w, input); err != nil {
		return nil, errors.Wrap(err, "...")
	}
	if mem, err = gorgonia.Add(filter, wx); err != nil {
		return nil, errors.Wrap(err, "...")
	}
	if mem, err = gorgonia.Add(mem, gru.b); err != nil {
		return nil, errors.Wrap(err, "...")
	}
	if mem, err = gorgonia.Tanh(mem); err != nil {
		return nil, errors.Wrap(err, "...")
	}

	var omz, omzh, upd, retVal *gorgonia.Node
	if omz, err = gorgonia.Sub(gru.one, z); err != nil {
		return nil, errors.Wrap(err, "...")
	}
	if omzh, err = gorgonia.HadamardProd(omz, prev); err != nil {
		return nil, errors.Wrap(err, "...")
	}
	if upd, err = gorgonia.HadamardProd(z, mem); err != nil {
		return nil, errors.Wrap(err, "...")
	}
	if retVal, err = gorgonia.Add(omzh, upd); err != nil {
		return nil, errors.Wrap(err, "...")
	}

	return retVal, nil
}
