package tokenizergpt2bpe

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
)

// Tokenizer implements GPT-2 BPE tokenization (byte-level).
type Tokenizer struct {
	vocab       map[string]int
	idToToken   []string
	merges      map[pair]int
	byteEncoder [256]string
	byteDecoder map[string]byte
	re          *regexp.Regexp
}

// NewFromFiles loads vocab.json and merges.txt and returns a tokenizer.
func NewFromFiles(vocabPath, mergesPath string) (*Tokenizer, error) {
	vocab, err := loadVocabEncode(vocabPath)
	if err != nil {
		return nil, err
	}
	idToToken, err := loadVocabDecode(vocabPath)
	if err != nil {
		return nil, err
	}
	merges, err := loadMerges(mergesPath)
	if err != nil {
		return nil, err
	}
	byteEncoder := buildByteEncoder()
	return &Tokenizer{
		vocab:       vocab,
		idToToken:   idToToken,
		merges:      merges,
		byteEncoder: byteEncoder,
		byteDecoder: buildByteDecoder(byteEncoder),
		re: regexp.MustCompile(
			`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`,
		),
	}, nil
}

// Encode converts text into token ids.
func (t *Tokenizer) Encode(text string) ([]uint32, error) {
	if t == nil {
		return nil, fmt.Errorf("tokenizer is nil")
	}
	matches := t.re.FindAllString(text, -1)
	ids := make([]uint32, 0, len(matches))
	cache := map[string][]string{}
	for _, m := range matches {
		token := byteEncode(m, t.byteEncoder)
		parts := bpe(token, t.merges, cache)
		for _, p := range parts {
			id, ok := t.vocab[p]
			if !ok {
				return nil, fmt.Errorf("unknown token: %q", p)
			}
			ids = append(ids, uint32(id))
		}
	}
	return ids, nil
}

// Decode converts token ids into text.
func (t *Tokenizer) Decode(ids []uint32) (string, error) {
	if t == nil {
		return "", fmt.Errorf("tokenizer is nil")
	}
	var b strings.Builder
	for _, id := range ids {
		if int(id) >= len(t.idToToken) || t.idToToken[id] == "" {
			return "", fmt.Errorf("unknown token id: %d", id)
		}
		b.WriteString(t.idToToken[id])
	}
	return byteDecode(b.String(), t.byteDecoder)
}

func loadVocabEncode(path string) (map[string]int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	raw := map[string]int{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	return raw, nil
}

func loadVocabDecode(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	raw := map[string]int{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	maxID := 0
	for _, id := range raw {
		if id > maxID {
			maxID = id
		}
	}
	out := make([]string, maxID+1)
	for tok, id := range raw {
		out[id] = tok
	}
	return out, nil
}

type pair struct {
	a, b string
}

func loadMerges(path string) (map[pair]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	out := map[pair]int{}
	rank := 0
	for scanner.Scan() {
		line := scanner.Text()
		if len(line) == 0 || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Split(line, " ")
		if len(parts) != 2 {
			continue
		}
		out[pair{parts[0], parts[1]}] = rank
		rank++
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func byteEncode(s string, encoder [256]string) string {
	bytes := []byte(s)
	var b strings.Builder
	b.Grow(len(bytes) * 2)
	for _, by := range bytes {
		b.WriteString(encoder[by])
	}
	return b.String()
}

func bpe(token string, merges map[pair]int, cache map[string][]string) []string {
	if val, ok := cache[token]; ok {
		return val
	}
	chars := splitRunes(token)
	if len(chars) == 1 {
		cache[token] = chars
		return chars
	}

	pairs := getPairs(chars)
	for {
		best, ok := lowestRankPair(pairs, merges)
		if !ok {
			break
		}
		chars = mergePair(chars, best)
		if len(chars) == 1 {
			break
		}
		pairs = getPairs(chars)
	}
	cache[token] = chars
	return chars
}

func getPairs(words []string) []pair {
	if len(words) < 2 {
		return nil
	}
	out := make([]pair, 0, len(words)-1)
	for i := 0; i < len(words)-1; i++ {
		out = append(out, pair{words[i], words[i+1]})
	}
	return out
}

func lowestRankPair(pairs []pair, ranks map[pair]int) (pair, bool) {
	bestRank := int(^uint(0) >> 1)
	var best pair
	found := false
	for _, p := range pairs {
		if r, ok := ranks[p]; ok && r < bestRank {
			bestRank = r
			best = p
			found = true
		}
	}
	return best, found
}

func mergePair(words []string, p pair) []string {
	if len(words) < 2 {
		return words
	}
	out := make([]string, 0, len(words))
	i := 0
	for i < len(words) {
		if i < len(words)-1 && words[i] == p.a && words[i+1] == p.b {
			out = append(out, words[i]+words[i+1])
			i += 2
			continue
		}
		out = append(out, words[i])
		i++
	}
	return out
}

func buildByteDecoder(encoder [256]string) map[string]byte {
	out := make(map[string]byte, 256)
	for i := 0; i < 256; i++ {
		out[encoder[i]] = byte(i)
	}
	return out
}

func byteDecode(s string, decoder map[string]byte) (string, error) {
	runes := splitRunes(s)
	bytes := make([]byte, len(runes))
	for i, r := range runes {
		b, ok := decoder[r]
		if !ok {
			return "", fmt.Errorf("unknown byte symbol: %q", r)
		}
		bytes[i] = b
	}
	return string(bytes), nil
}

func splitRunes(s string) []string {
	out := make([]string, 0, len(s))
	for _, r := range s {
		out = append(out, string(r))
	}
	return out
}

func buildByteEncoder() [256]string {
	var bytesList []int
	for i := int('!'); i <= int('~'); i++ {
		bytesList = append(bytesList, i)
	}
	for i := int('¡'); i <= int('¬'); i++ {
		bytesList = append(bytesList, i)
	}
	for i := int('®'); i <= int('ÿ'); i++ {
		bytesList = append(bytesList, i)
	}

	byteSet := map[int]bool{}
	for _, v := range bytesList {
		byteSet[v] = true
	}

	chars := make([]int, len(bytesList))
	copy(chars, bytesList)

	n := 0
	for b := 0; b < 256; b++ {
		if !byteSet[b] {
			bytesList = append(bytesList, b)
			chars = append(chars, 256+n)
			n++
		}
	}

	type pairMap struct {
		b int
		r int
	}
	pairs := make([]pairMap, len(bytesList))
	for i := range bytesList {
		pairs[i] = pairMap{b: bytesList[i], r: chars[i]}
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].b < pairs[j].b })

	var out [256]string
	for _, p := range pairs {
		out[byte(p.b)] = string(rune(p.r))
	}
	return out
}
