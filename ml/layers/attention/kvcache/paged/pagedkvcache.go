// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package pagedkv

import (
	"fmt"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/pkg/errors"
)

// Shape is [nLayers, 2, nBlocks, blockSize, H, D]
type Nodes []*Node
type Pools []*tensors.Tensor

type RequestState struct {
	Blocks       []int
	NextPosition int
}

type Allocator struct {
	Table      map[int]*RequestState
	FreeBlocks []int
}

type Cache struct {
	BlockSize int
	NumBlocks int
}

func NewCache() *Cache {
	return &Cache{
		BlockSize: 16,
		NumBlocks: 512,
	}
}

func (k *Cache) WithBlockSize(size int) *Cache {
	if size <= 0 {
		panic(fmt.Sprintf("pagedkv.WithBlockSize: block size must be positive, received %d", size))
	}
	k.BlockSize = size
	return k
}

func (k *Cache) WithNumBlocks(num int) *Cache {
	if num <= 0 {
		panic(fmt.Sprintf("pagedkv.WithNumBlocks: num blocks must be positive, received %d", num))
	}
	k.NumBlocks = num
	return k
}

func (k *Cache) InitializePools(nLayers, numKVHeads, headDim int, dtype dtypes.DType) (Pools, error) {
	if nLayers <= 0 || headDim <= 0 || numKVHeads <= 0 {
		return nil, errors.New("nLayers, headDim and numKVHeads must be positive")
	}

	pools := make([]*tensors.Tensor, nLayers)
	shape := shapes.Make(dtype, 2, k.NumBlocks, k.BlockSize, numKVHeads, headDim)
	for i := range nLayers {
		pools[i] = tensors.FromShape(shape)
	}
	return pools, nil
}

func (k *Cache) InitializeAllocator() *Allocator {
	table := make(map[int]*RequestState)
	freeBlocks := make([]int, k.NumBlocks)
	for i := range k.NumBlocks {
		freeBlocks[i] = i
	}

	return &Allocator{Table: table, FreeBlocks: freeBlocks}
}

// idempotent: only adds unallocated blocks
func (k *Cache) Allocate(a *Allocator, reqId, seqLen int) error {
	need := k.numAllocateBlocks(seqLen)

	table, ok := a.Table[reqId]
	if ok {
		need -= len(table.Blocks)
	}

	if need > len(a.FreeBlocks) {
		return errors.Errorf("need %d blocks > %d free blocks", need, len(a.FreeBlocks))
	}

	if !ok {
		a.Table[reqId] = &RequestState{}
		table = a.Table[reqId]
	}

	for range need {
		n := len(a.FreeBlocks) - 1
		block := a.FreeBlocks[n]
		table.Blocks = append(table.Blocks, block)
		a.FreeBlocks = a.FreeBlocks[:n]
	}
	return nil
}

func (k *Cache) numAllocateBlocks(currentSeqLen int) int {
	nBlocks := currentSeqLen / k.BlockSize
	if currentSeqLen%k.BlockSize != 0 {
		nBlocks += 1
	}
	return nBlocks
}

func (k *Cache) Free(a *Allocator, reqId int) error {
	table, ok := a.Table[reqId]
	if !ok {
		return errors.Errorf("request %d not found in allocator", reqId)
	}

	for _, block := range table.Blocks {
		a.FreeBlocks = append(a.FreeBlocks, block)
	}
	delete(a.Table, reqId)

	return nil
}

func (k *Cache) NextPhysicalSlot(a *Allocator, reqId int) (int32, error) {
	table, ok := a.Table[reqId]
	if !ok {
		return 0, errors.Errorf("request %d not found in allocator", reqId)
	}

	nextPosition := table.NextPosition
	logical := nextPosition / k.BlockSize
	offset := nextPosition % k.BlockSize

	blocks := table.Blocks

	if logical < 0 {
		panic(fmt.Sprintf("pagedkv.NextPhysicalSlot: logical block %d < 0", logical))
	}
	if logical >= len(blocks) {
		panic(fmt.Sprintf("pagedkv.NextPhysicalSlot: logical block %d >= len(blocks) == %d", logical, len(blocks)))
	}

	return int32(table.Blocks[logical]*k.BlockSize + offset), nil
}

func (k *Cache) Update(cache Nodes, nextK, nextV *Node, layerIdx int, physicalSlot *Node) error {
	// [2, nBlocks, blockSize, H, D]
	// 0/1, blockIdx, offsetIdx, nKVHeads, dim
	if layerIdx >= len(cache) {
		return errors.Errorf("layer index %d >= cache length %d", layerIdx, len(cache))
	}
	pool := cache[layerIdx]

	H := cache[layerIdx].Shape().Dimensions[3]
	D := cache[layerIdx].Shape().Dimensions[4]

	// Normalize to [B, S, H, D], right now designing for S == 1
	// What if S == H?
	if nextK.Shape().Dimensions[2] != H {
		nextK = Transpose(nextK, 1, 2)
		nextV = Transpose(nextV, 1, 2)
	}

	// [1, 1, H, D]
	nextK = Reshape(nextK, 1, 1, H, D)
	nextV = Reshape(nextV, 1, 1, H, D)

	g := nextK.Graph()

	keyIdx := Const(g, int32(0))
	valIdx := Const(g, int32(1))
	headsIdx := Const(g, int32(0))
	dimIdx := Const(g, int32(0))

	origDims := pool.Shape().Dimensions
	transformedDims := []int{origDims[0], origDims[1] * origDims[2], origDims[3], origDims[4]}
	// [2, nBlocks * blockSize, H, D]
	pool = Reshape(pool, transformedDims...)

	pool = DynamicUpdateSlice(pool, nextK, []*Node{keyIdx, physicalSlot, headsIdx, dimIdx})
	pool = DynamicUpdateSlice(pool, nextV, []*Node{valIdx, physicalSlot, headsIdx, dimIdx})

	pool = Reshape(pool, origDims...)
	cache[layerIdx] = pool

	return nil
}

// Advance cursor
func (k *Cache) Advance(a *Allocator, reqId int) error {
	table, ok := a.Table[reqId]
	if !ok {
		return errors.Errorf("request %d not found in allocator", reqId)
	}
	table.NextPosition++
	return nil
}

// Returns in [B, S, H, D]
func (k *Cache) Get(cache Nodes, blockTable *Node, layerIdx int) (*Node, *Node, error) {
	if layerIdx >= len(cache) {
		return nil, nil, errors.Errorf("layer index %d >= cache length %d", layerIdx, len(cache))
	}
	pool := cache[layerIdx]

	// [nBlocks, blockSize, H, D]
	K, V := Squeeze(Slice(pool, AxisElem(0)), 0), Squeeze(Slice(pool, AxisElem(1)), 0)

	batchSize := blockTable.Shape().Dimensions[0]
	maxBlocks := blockTable.Shape().Dimensions[1]

	// [B, maxBlocks, 1]
	blockTable = Reshape(blockTable, batchSize, maxBlocks, 1)

	// [B, maxBlocks, 1, blockSize, H, D]
	K, V = GatherSlices(K, []int{0}, blockTable, []int{1}, false), GatherSlices(V, []int{0}, blockTable, []int{1}, false)

	// [B, maxBlocks, blockSize, H, D]
	K, V = Squeeze(K, 2), Squeeze(V, 2)

	origDims := K.Shape().Dimensions
	dims := []int{origDims[0], origDims[1] * origDims[2], origDims[3], origDims[4]}

	// [B, maxBlocks * blockSize, H, D]
	K, V = Reshape(K, dims...), Reshape(V, dims...)

	return K, V, nil
}
