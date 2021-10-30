package gru

import (
	"crypto/sha256"
	"fmt"
	"math/big"
	"strings"
	"unicode"
)

func ToWords(str string) ([]string, error) {
	if str == "" {
		return nil, fmt.Errorf("String is empty")
	}
	var newStr = strings.ToLower(str)

	f := func(c rune) bool {
		return !unicode.IsLetter(c) && !unicode.IsNumber(c)
	}

	tmp := strings.FieldsFunc(newStr, f)

	return tmp, nil
}

func Hashing(sentenceWords []string, vocab int) ([]int64, error) {
	hashedList := make([]int, vocab)
	ans := make([]int64, len(sentenceWords))
	for i, word := range sentenceWords {
		hashedValue := sha256.New()
		hashedValue.Write([]byte(word))
		hexStr := fmt.Sprintf("%x", hashedValue.Sum(nil))
		hexInt := new(big.Int)
		hexInt, ok := hexInt.SetString(hexStr, 16)
		if !ok {
			return nil, fmt.Errorf("can't create big int from hex")
		}
		bigVectorLength := big.NewInt(int64(vocab))
		modulo := new(big.Int)
		modulo = modulo.Mod(hexInt, bigVectorLength)
		moduloInt64 := modulo.Int64()
		hashedList[moduloInt64] += 1
		if hashedList[moduloInt64] > 0 {
			ans[i] = moduloInt64
		}
	}
	return ans, nil
}
