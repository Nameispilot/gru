package main

func main() {

}

func int64ToInt(s []int64) []int {
	ans := make([]int, len(s))
	for i := range s {
		ans[i] = int(s[i])
	}
	return ans
}
