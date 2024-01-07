package mtl

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestMTLSize(t *testing.T) {
	actual := MTLSize{W: 1, H: 2, D: 3}
	expected := MTLSize{W: 1, H: 2, D: 3}

	require.Equal(t, expected, MTLSizeFromC(actual.C()))
}

func TestNSRange(t *testing.T) {
	actual := NSRange{Location: 1, Length: 2}
	expected := NSRange{Location: 1, Length: 2}

	require.Equal(t, expected, NSRangeFromC(actual.C()))
}
