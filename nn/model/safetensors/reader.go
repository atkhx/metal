package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
)

type (
	ReadReadAt interface {
		io.Reader
		io.ReaderAt
	}
	header struct {
		DType   string   `json:"dtype"`
		Shape   []int    `json:"shape"`
		Offsets []uint64 `json:"data_offsets"`
	}
	reader struct {
		f ReadReadAt

		headers   map[string]header
		headerLen uint64
	}
)

func NewReader(r ReadReadAt) (*reader, error) {
	headers, headerLen, err := readHeaders(r)
	if err != nil {
		return nil, fmt.Errorf("read headers: %v", err)
	}
	return &reader{f: r, headers: headers, headerLen: headerLen}, nil
}

func (r *reader) ReadTensor(name string) ([]float32, error) {
	if _, ok := r.headers[name]; !ok {
		return nil, fmt.Errorf("block %s not found", name)
	}
	return readTensor(r.f, int64(8+r.headerLen), r.headers[name], name)
}

func readHeaders(r io.Reader) (map[string]header, uint64, error) {
	var headerLen uint64
	if err := binary.Read(r, binary.LittleEndian, &headerLen); err != nil {
		return nil, headerLen, fmt.Errorf("read header length: %v", err)
	}
	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(r, headerBytes); err != nil {
		return nil, headerLen, fmt.Errorf("read header: %v", err)
	}
	raw := map[string]header{}
	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return nil, headerLen, fmt.Errorf("parse header json: %v", err)
	}
	return raw, headerLen, nil
}

func readTensor(r ReadReadAt, offset int64, h header, name string) ([]float32, error) {
	if len(h.Offsets) != 2 {
		return nil, fmt.Errorf("invalid offsets for %s: %d", name, len(h.Offsets))
	}

	start, end := int64(h.Offsets[0]), int64(h.Offsets[1])
	if end < start {
		return nil, fmt.Errorf("invalid offsets for %s: %d, end < start", name, len(h.Offsets))
	}
	size := end - start

	buf := make([]byte, size)
	if _, err := r.ReadAt(buf, offset+start); err != nil {
		return nil, fmt.Errorf("read block %s: %v", name, err)
	}

	switch h.DType {
	case "F32":
		out := make([]float32, 0, len(buf)/4)
		for i := 0; i < len(buf); i += 4 {
			out = append(out, math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:i*4+4])))
		}
		return out, nil
	case "F16":
		out := make([]float32, 0, len(buf)/2)
		for i := 0; i < len(buf); i += 2 {
			out = append(out, halfToFloat32(binary.LittleEndian.Uint16(buf[i:i+2])))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported type %s for %s", h.DType, name)
	}
}

func halfToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := int32((h >> 10) & 0x1f)
	frac := uint32(h & 0x3ff)

	if exp == 0 {
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		// subnormal
		for (frac & 0x400) == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x3ff
	} else if exp == 31 {
		if frac == 0 {
			return math.Float32frombits((sign << 31) | 0x7f800000)
		}
		return math.Float32frombits((sign << 31) | 0x7f800000 | (frac << 13))
	}

	exp = exp + (127 - 15)
	frac = frac << 13
	return math.Float32frombits((sign << 31) | (uint32(exp) << 23) | frac)
}
