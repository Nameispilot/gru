package main

import (
	"fmt"
	"gru"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var trainData = "What did the bartender say to the jumper cables? You better not try to start anything."

var testData = []string{"the bartender say to the", "the jumper", "You better not", "try to start"}

func main() {

	g := gorgonia.NewGraph()

	//hashing
	var tmp []int64
	var words []string
	var err error
	if words, err = gru.ToWords(trainData); err != nil {
		panic(err)
	}
	vocab := 31
	if tmp, err = gru.Hashing(words, vocab); err != nil {
		panic(err)
	}
	sentence := int64ToInt(tmp)

	// define network structure
	inputSize := len(sentence)
	hiddenSize := 25
	embeddingSize := 30
	net := gru.MakeNetwork(g, inputSize, embeddingSize, inputSize, hiddenSize)

	// prepare input node
	inputShape := tensor.Shape{len(sentence)}
	inputNet := gorgonia.NewTensor(g, gorgonia.Int, 1, gorgonia.WithShape(inputShape...), gorgonia.WithName("train_input_"))
	in := tensor.New(tensor.WithShape(inputShape...), tensor.WithBacking(sentence))
	err = gorgonia.Let(inputNet, in)
	if err != nil {
		panic(err)
	}

	vocabIndex := make(map[int]int)
	for i, v := range sentence {
		vocabIndex[v] = i
	}

	// feed forward proccess
	if err = net.Fwd(inputNet, vocabIndex); err != nil {
		panic(err)
	}

	// prepare cost node
	var cost *gorgonia.Node
	if cost, err = net.GetCost(); err != nil {
		panic(err)
	}

	// define gradients
	if _, err = gorgonia.Grad(cost, net.Learnables()...); err != nil {
		panic(err)
	}

	// prepare variable for storing neural network's cost
	var costValue gorgonia.Value
	gorgonia.Read(cost, &costValue)

	// initialize solver
	learning_rate := 0.01
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithLearnRate(learning_rate))

	// define tape machine
	tm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(net.Learnables()...))
	defer tm.Close()

	// training
	evalPrint := 50
	epochs := 500
	for i := 0; i < epochs; i++ {
		err = tm.RunAll()
		if err != nil {
			panic(err)
		}

		err = solver.Step(gorgonia.NodesToValueGrads(net.Learnables()))
		if err != nil {
			panic(err)
		}

		if i%evalPrint == 0 {
			fmt.Printf("Epoch %d:\n", i)
			fmt.Printf("\tDiscriminator's loss: %.3f\n", costValue.Data().(float32))
		}
		tm.Reset()
	}

	//testing
	var toWords []string
	for i := 0; i < len(testData); i++ {
		if toWords, err = gru.ToWords(testData[i]); err != nil {
			panic(err)
		}
		if tmp, err = gru.Hashing(toWords, vocab); err != nil {
			panic(err)
		}
		testSentence := int64ToInt(tmp)

		testShape := tensor.Shape{len(testSentence)}
		in := tensor.New(tensor.WithShape(testShape...), tensor.WithBacking(testSentence))
		testNet := gorgonia.NewTensor(g, gorgonia.Int, 1, gorgonia.WithShape(testShape...), gorgonia.WithName("train_input_"))
		err = gorgonia.Let(testNet, in)
		if err != nil {
			panic(err)
		}

		sampledId, err := net.Predict(g, testNet, vocabIndex)
		if err != nil {
			panic(err)
		}
		fmt.Printf("Sentence: %s ... %s\n", testData[i], words[sampledId])

	}

}

func int64ToInt(s []int64) []int {
	ans := make([]int, len(s))
	for i := range s {
		ans[i] = int(s[i])
	}
	return ans
}
