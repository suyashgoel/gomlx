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

func (k *Cache) InitializePools(orderedScopes []string, numKVHeads, headDim int, dtype dtypes.DType) (Pools, error) {
	if len(orderedScopes) == 0 || headDim <= 0 || numKVHeads <= 0 {
		return nil, errors.New("ordered scopes must be non-empty, and headDim and numKVHeads must be positive")
	}

	pools := make([]*tensors.Tensor, len(orderedScopes))
	shape := shapes.Make(dtype, 2, k.NumBlocks, k.BlockSize, numKVHeads, headDim)
	for i := range orderedScopes {
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

func (k *Cache) Update(a *Allocator, cache Nodes, nextK, nextV *Node, layerIdx, reqId int) error {
	table, ok := a.Table[reqId]
	if !ok {
		return errors.Errorf("request %d not found in allocator", reqId)
	}

	nextPosition := table.NextPosition
	blocks := table.Blocks

	logical := nextPosition / k.BlockSize
	offset := nextPosition % k.BlockSize

	if logical < 0 {
		panic(fmt.Sprintf("pagedkv.Update: logical block %d < 0", logical))
	}
	if logical == len(blocks) && offset != 0 {
		panic(fmt.Sprintf("pagedkv.Update: logical block %d ==  len(blocks) and offset %d != 0", logical, offset))
	}
	if logical > len(blocks) {
		panic(fmt.Sprintf("pagedkv.Update: logical block %d > len(blocks) == %d", logical, len(blocks)))
	}

	// [2, nBlocks, blockSize, H, D]
	// 0/1, blockIdx, offsetIdx, nKVHeads, dim
	if layerIdx >= len(cache) {
		return errors.Errorf("layer index %d >= len(cache)", layerIdx)
	}
	H := cache[layerIdx].Shape().Dimensions[3]
	D := cache[layerIdx].Shape().Dimensions[4]

	// Normalize to [B, S, H, D], right now designing for S == 1
	// What if S == H?
	if nextK.Shape().Dimensions[2] != H {
		nextK = Transpose(nextK, 1, 2)
		nextV = Transpose(nextV, 1, 2)
	}

	// [1, 1, 1, H, D]
	nextK = Reshape(nextK, 1, 1, 1, H, D)
	nextV = Reshape(nextV, 1, 1, 1, H, D)

	g := nextK.Graph()

	// key is first dimension, value is second dimension
	keyIdx := Const(g, int32(0))
	valIdx := Const(g, int32(1))

	offsetIdx := Const(g, int32(offset))
	headsIdx := Const(g, int32(0))
	dimIdx := Const(g, int32(0))

	if logical == len(blocks) {
		if len(a.FreeBlocks) == 0 {
			return errors.New("no free blocks available")
		}

		n := len(a.FreeBlocks) - 1
		block := a.FreeBlocks[n]
		table.Blocks = append(table.Blocks, block)
		a.FreeBlocks = a.FreeBlocks[:n]
	}
	physIdx := Const(g, int32(table.Blocks[logical]))

	cache[layerIdx] = DynamicUpdateSlice(cache[layerIdx], nextK, []*Node{keyIdx, physIdx, offsetIdx, headsIdx, dimIdx})
	cache[layerIdx] = DynamicUpdateSlice(cache[layerIdx], nextV, []*Node{valIdx, physIdx, offsetIdx, headsIdx, dimIdx})

	return nil
}

func (k *Cache) Advance(a *Allocator, reqId int) error {
	table, ok := a.Table[reqId]
	if !ok {
		return errors.Errorf("request %d not found in allocator", reqId)
	}
	table.NextPosition++
	return nil
}

// Returns in [B, S, H, D]
func (k *Cache) Get(a *Allocator, cache Nodes, layerIdx, reqId int) (*Node, *Node, error) {
	table, ok := a.Table[reqId]
	if !ok {
		return nil, nil, errors.Errorf("request %d not found in allocator", reqId)
	}

	nextPos := table.NextPosition
	if nextPos > len(table.Blocks)*k.BlockSize {
		panic(fmt.Sprintf("pagedkv.Get: next position %d > allocated slots == %d", nextPos, len(table.Blocks)*k.BlockSize))
	}
	if nextPos == 0 || len(table.Blocks) == 0 {
		return nil, nil, errors.Errorf("request %d is empty", reqId)
	}

	if layerIdx >= len(cache) {
		return nil, nil, errors.Errorf("layerIdx %d >= len(cache)", layerIdx)
	}
	pool := cache[layerIdx]

	g := pool.Graph()

	blocksInt32 := make([]int32, len(table.Blocks))
	for i, num := range table.Blocks {
		blocksInt32[i] = int32(num)
	}

	// [nBlocks, 1]
	blocks := Reshape(Const(g, blocksInt32), len(blocksInt32), 1)

	// [1, nBlocks, blockSize, H, D]
	K, V := Slice(pool, AxisElem(0)), Slice(pool, AxisElem(1))
	// [nBlocks, 1, 1, blockSize, H, D]
	K, V = GatherSlices(K, []int{1}, blocks, []int{1}, false), GatherSlices(V, []int{1}, blocks, []int{1}, false)

	// [nBlocks, blockSize, H, D]
	K, V = Squeeze(K, 1, 2), Squeeze(V, 1, 2)

	reshapeDims := []int{1, K.Shape().Dimensions[0] * K.Shape().Dimensions[1], K.Shape().Dimensions[2], K.Shape().Dimensions[3]}
	// [1, nBlocks * blockSize, H, D]
	K, V = Reshape(K, reshapeDims...), Reshape(V, reshapeDims...)

	// [1, S, H, D]
	K, V = Slice(K, AxisRange(), AxisRangeFromStart(nextPos)), Slice(V, AxisRange(), AxisRangeFromStart(nextPos))

	return K, V, nil
}
