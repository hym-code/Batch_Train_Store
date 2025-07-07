package main

import (
	"bufio"
	"container/heap"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// 常量定义
const (
	MAX_DISK_NUM      = 11
	MAX_DISK_SIZE     = 16385
	MAX_REQUEST_NUM   = 30000001
	MAX_OBJECT_NUM    = 100001
	MAX_TAG_NUM       = 17
	REP_NUM           = 3
	FRE_PER_SLICING   = 1800
	EXTRA_TIME        = 105
	MAX_OBJECT_SIZE   = 5
	INITIAL_READ_COST = 64
)

// 存储系统核心
type StorageSystem struct {
	T, M, N, V, G int
	Disks          []*Disk
	Requests       []*Request
	Objects        []*Object
	Partitions     [][][]*Partition
	CurrentTime    int
	mu             sync.Mutex
}

// 磁盘管理
type Disk struct {
	ID            int
	HeadPosition  int
	LastTokenCost int
	Load          int
	Blocks        []int
	BlockInfo     [][3]int
	IsUsed        []bool
	mu            sync.Mutex
}

// 分区管理
type Partition struct {
	DiskID    int
	Tag       int
	StartPos  int
	EndPos    int
	SizeIndex *SizeHeap
	mu        sync.Mutex
}

// 空闲区间
type FreeInterval struct {
	Start int
	Size  int
}

// 大小堆
type SizeHeap []*FreeInterval

func (h SizeHeap) Len() int           { return len(h) }
func (h SizeHeap) Less(i, j int) bool { return h[i].Size < h[j].Size }
func (h SizeHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *SizeHeap) Push(x interface{}) {
	*h = append(*h, x.(*FreeInterval))
}
func (h *SizeHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// 请求管理
type Request struct {
	ID         int
	ObjectID   int
	PrevID     int
	IsDone     bool
	BlocksRead uint8
	mu         sync.Mutex
}

// 对象管理
type Object struct {
	ID          int
	Size        int
	Tag         int
	LastRequest int
	IsDeleted   bool
	Replicas    [REP_NUM + 1]int
	Units       [REP_NUM + 1][MAX_OBJECT_SIZE + 1]int
}

// 初始化存储系统
func NewStorageSystem(T, M, N, V, G int) *StorageSystem {
	sys := &StorageSystem{
		T:         T,
		M:         M,
		N:         N,
		V:         V,
		G:         G,
		Disks:     make([]*Disk, N+1),
		Requests:  make([]*Request, MAX_REQUEST_NUM),
		Objects:   make([]*Object, MAX_OBJECT_NUM),
		Partitions: make([][][]*Partition, N+1),
	}

	// 初始化对象
	for i := range sys.Objects {
		sys.Objects[i] = &Object{ID: i}
	}

	// 初始化磁盘
	for i := 1; i <= N; i++ {
		sys.Disks[i] = &Disk{
			ID:            i,
			HeadPosition:  1,
			LastTokenCost: 0,
			Blocks:        make([]int, V+1),
			BlockInfo:     make([][3]int, V+1),
			IsUsed:        make([]bool, V+1),
		}
		sys.Partitions[i] = make([][]*Partition, M+1)
	}

	// 初始化分区
	sys.initPartitions()
	return sys
}

// 初始化分区
func (sys *StorageSystem) initPartitions() {
	partitionSize := sys.V / sys.M
	for tag := 1; tag <= sys.M; tag++ {
		for diskID := 1; diskID <= sys.N; diskID++ {
			start := (tag-1)*partitionSize + 1
			end := tag * partitionSize
			if tag == sys.M {
				end = sys.V
			}

			p := &Partition{
				DiskID:   diskID,
				Tag:      tag,
				StartPos: start,
				EndPos:   end,
				SizeIndex: new(SizeHeap),
			}

			// 添加整个分区作为一个空闲区间
			heap.Push(p.SizeIndex, &FreeInterval{Start: start, Size: end - start + 1})
			sys.Partitions[diskID][tag] = append(sys.Partitions[diskID][tag], p)
		}
	}
}

// 检查块是否已读
func (r *Request) IsBlockRead(blockIdx int) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return (r.BlocksRead & (1 << blockIdx)) != 0
}

// 标记块已读
func (r *Request) MarkBlockRead(blockIdx int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.BlocksRead |= (1 << blockIdx)
}

// 检查请求是否完成
func (r *Request) IsComplete(obj *Object) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	for i := 0; i < obj.Size; i++ {
		if (r.BlocksRead & (1 << i)) == 0 {
			return false
		}
	}
	return true
}

// 处理时间戳动作
func (sys *StorageSystem) TimestampAction(scanner *bufio.Scanner) {
	scanner.Scan()
	parts := strings.Fields(scanner.Text())
	if len(parts) < 2 {
		return
	}
	timestamp, _ := strconv.Atoi(parts[1])
	fmt.Printf("TIMESTAMP %d\n", timestamp)
	sys.CurrentTime = timestamp
}

// 处理删除动作（并行化）
func (sys *StorageSystem) DeleteAction(scanner *bufio.Scanner) {
	scanner.Scan()
	nDelete, _ := strconv.Atoi(scanner.Text())

	if nDelete == 0 {
		fmt.Println("0")
		return
	}

	scanner.Scan()
	objIDs := strings.Fields(scanner.Text())
	objList := make([]int, nDelete)
	for i := 0; i < nDelete; i++ {
		id, _ := strconv.Atoi(objIDs[i])
		objList[i] = id
	}

	abortCount := 0
	abortRequests := make([]int, 0)
	var mu sync.Mutex
	var wg sync.WaitGroup

	for _, objID := range objList {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			obj := sys.Objects[id]
			obj.mu := sync.Mutex{}
			obj.mu.Lock()
			defer obj.mu.Unlock()

			reqID := obj.LastRequest
			localAborts := []int{}

			for reqID != 0 {
				req := sys.Requests[reqID]
				if req != nil {
					req.mu.Lock()
					if !req.IsDone {
						req.IsDone = true
						localAborts = append(localAborts, reqID)
					}
					req.mu.Unlock()
				}
				reqID = req.PrevID
			}

			// 释放空间
			obj.IsDeleted = true
			for rep := 1; rep <= REP_NUM; rep++ {
				diskID := obj.Replicas[rep]
				if diskID == 0 {
					continue
				}

				firstUnit := obj.Units[rep][1]
				isContinuous := true
				for j := 2; j <= obj.Size; j++ {
					if obj.Units[rep][j] != firstUnit+j-1 {
						isContinuous = false
						break
					}
				}

				if isContinuous {
					sys.freeContinuousSpace(diskID, obj.Tag, firstUnit, obj.Size)
				} else {
					for j := 1; j <= obj.Size; j++ {
						unitID := obj.Units[rep][j]
						sys.freeSingleUnit(diskID, obj.Tag, unitID)
					}
				}

				// 更新磁盘负载
				sys.Disks[diskID].mu.Lock()
				sys.Disks[diskID].Load -= obj.Size
				sys.Disks[diskID].mu.Unlock()
			}

			// 合并结果
			if len(localAborts) > 0 {
				mu.Lock()
				abortCount += len(localAborts)
				abortRequests = append(abortRequests, localAborts...)
				mu.Unlock()
			}
		}(objID)
	}

	wg.Wait()

	// 输出结果
	fmt.Println(abortCount)
	for _, reqID := range abortRequests {
		fmt.Println(reqID)
	}
}

// 释放连续空间
func (sys *StorageSystem) freeContinuousSpace(diskID, tag, start, size int) {
	disk := sys.Disks[diskID]
	disk.mu.Lock()
	defer disk.mu.Unlock()

	for i := start; i < start+size; i++ {
		disk.Blocks[i] = 0
		disk.IsUsed[i] = false
		disk.BlockInfo[i] = [3]int{0, 0, 0}
	}

	// 添加到空闲列表
	for _, p := range sys.Partitions[diskID][tag] {
		if p.StartPos <= start && start+size-1 <= p.EndPos {
			p.mu.Lock()
			heap.Push(p.SizeIndex, &FreeInterval{Start: start, Size: size})
			p.mu.Unlock()
			break
		}
	}
}

// 释放单个单元
func (sys *StorageSystem) freeSingleUnit(diskID, tag, unitID int) {
	disk := sys.Disks[diskID]
	disk.mu.Lock()
	defer disk.mu.Unlock()

	disk.Blocks[unitID] = 0
	disk.IsUsed[unitID] = false
	disk.BlockInfo[unitID] = [3]int{0, 0, 0}

	// 添加到空闲列表
	for _, p := range sys.Partitions[diskID][tag] {
		if p.StartPos <= unitID && unitID <= p.EndPos {
			p.mu.Lock()
			heap.Push(p.SizeIndex, &FreeInterval{Start: unitID, Size: 1})
			p.mu.Unlock()
			break
		}
	}
}

// 处理写入动作（并行化）
func (sys *StorageSystem) WriteAction(scanner *bufio.Scanner) {
	scanner.Scan()
	nWrite, _ := strconv.Atoi(scanner.Text())

	if nWrite == 0 {
		return
	}

	// 预先读取所有写入对象的信息
	writeInfos := make([]struct {
		id, size, tag int
	}, nWrite)
	for i := 0; i < nWrite; i++ {
		scanner.Scan()
		parts := strings.Fields(scanner.Text())
		id, _ := strconv.Atoi(parts[0])
		size, _ := strconv.Atoi(parts[1])
		tag, _ := strconv.Atoi(parts[2])
		writeInfos[i].id = id
		writeInfos[i].size = size
		writeInfos[i].tag = tag
	}

	// 用于存储每个对象的输出行
	outputs := make([]string, nWrite)
	var wg sync.WaitGroup
	var mu sync.Mutex

	for idx, info := range writeInfos {
		wg.Add(1)
		go func(idx int, info struct{ id, size, tag int }) {
			defer wg.Done()
			id := info.id
			size := info.size
			tag := info.tag
			obj := sys.Objects[id]
			obj.mu := sync.Mutex{}
			obj.mu.Lock()
			defer obj.mu.Unlock()

			obj.Size = size
			obj.Tag = tag
			obj.IsDeleted = false

			// 选择磁盘（需要同步访问磁盘负载）
			mu.Lock()
			diskLoads := make([]int, sys.N+1)
			for d := 1; d <= sys.N; d++ {
				diskLoads[d] = sys.Disks[d].Load
			}
			mu.Unlock()

			// 为每个副本选择硬盘
			chosenDisks := make([]int, 0, REP_NUM)
			replicaOutputs := make([]string, REP_NUM)

			for rep := 1; rep <= REP_NUM; rep++ {
				minLoad := 1<<31 - 1
				bestDisk := -1

				// 选择负载最低的可用磁盘
				for d := 1; d <= sys.N; d++ {
					// 检查磁盘是否已被选择
					alreadyChosen := false
					for _, chosen := range chosenDisks {
						if chosen == d {
							alreadyChosen = true
							break
						}
					}
					if !alreadyChosen && diskLoads[d] < minLoad {
						minLoad = diskLoads[d]
						bestDisk = d
					}
				}

				if bestDisk == -1 {
					bestDisk = 1
				}
				chosenDisks = append(chosenDisks, bestDisk)
				obj.Replicas[rep] = bestDisk

				// 尝试分配连续空间
				startPos := sys.allocateContinuousSpace(bestDisk, tag, size)
				if startPos != -1 {
					// 连续分配成功
					disk := sys.Disks[bestDisk]
					disk.mu.Lock()
					for j := 0; j < size; j++ {
						unitID := startPos + j
						obj.Units[rep][j+1] = unitID
						disk.Blocks[unitID] = id
						disk.BlockInfo[unitID] = [3]int{id, rep, j}
						disk.IsUsed[unitID] = true
					}
					disk.mu.Unlock()

					// 构建输出行
					line := fmt.Sprintf("%d", bestDisk)
					for j := 1; j <= size; j++ {
						line += fmt.Sprintf(" %d", obj.Units[rep][j])
					}
					replicaOutputs[rep-1] = line
				} else {
					// 分配分散空间
					positions := sys.allocateScatteredSpace(bestDisk, tag, size)
					disk := sys.Disks[bestDisk]
					disk.mu.Lock()
					for j := 0; j < size; j++ {
						unitID := positions[j]
						obj.Units[rep][j+1] = unitID
						disk.Blocks[unitID] = id
						disk.BlockInfo[unitID] = [3]int{id, rep, j}
						disk.IsUsed[unitID] = true
					}
					disk.mu.Unlock()

					// 构建输出行
					line := fmt.Sprintf("%d", bestDisk)
					for j := 0; j < size; j++ {
						line += fmt.Sprintf(" %d", positions[j])
					}
					replicaOutputs[rep-1] = line
				}

				// 更新磁盘负载
				mu.Lock()
				sys.Disks[bestDisk].Load += size
				mu.Unlock()
			}

			// 构建完整输出
			output := fmt.Sprintf("%d\n", id)
			for _, repLine := range replicaOutputs {
				output += repLine + "\n"
			}

			// 保存输出
			mu.Lock()
			outputs[idx] = output
			mu.Unlock()
		}(idx, info)
	}

	wg.Wait()

	// 按顺序输出所有对象的写入结果
	for _, output := range outputs {
		fmt.Print(output)
	}
}

// 分配连续空间
func (sys *StorageSystem) allocateContinuousSpace(diskID, tag, size int) int {
	for _, p := range sys.Partitions[diskID][tag] {
		p.mu.Lock()
		if p.SizeIndex.Len() == 0 {
			p.mu.Unlock()
			continue
		}

		// 查找最小合适区间
		interval := heap.Pop(p.SizeIndex).(*FreeInterval)
		if interval.Size >= size {
			start := interval.Start
			if interval.Size > size {
				heap.Push(p.SizeIndex, &FreeInterval{
					Start: start + size,
					Size:  interval.Size - size,
				})
			}
			p.mu.Unlock()
			return start
		} else {
			// 不够，放回去
			heap.Push(p.SizeIndex, interval)
			p.mu.Unlock()
		}
	}
	return -1
}

// 分配分散空间
func (sys *StorageSystem) allocateScatteredSpace(diskID, tag, size int) []int {
	positions := make([]int, 0, size)

	// 在当前标签分区中查找
	for _, p := range sys.Partitions[diskID][tag] {
		p.mu.Lock()
		for p.SizeIndex.Len() > 0 && len(positions) < size {
			interval := heap.Pop(p.SizeIndex).(*FreeInterval)
			positions = append(positions, interval.Start)

			// 如果还有剩余空间，放回堆中
			if interval.Size > 1 {
				heap.Push(p.SizeIndex, &FreeInterval{
					Start: interval.Start + 1,
					Size:  interval.Size - 1,
				})
			}
		}
		p.mu.Unlock()
		if len(positions) == size {
			break
		}
	}

	// 如果当前分区不够，从其他分区查找
	if len(positions) < size {
		for otherTag := 1; otherTag <= sys.M; otherTag++ {
			if otherTag == tag {
				continue
			}

			for _, p := range sys.Partitions[diskID][otherTag] {
				p.mu.Lock()
				for p.SizeIndex.Len() > 0 && len(positions) < size {
					interval := heap.Pop(p.SizeIndex).(*FreeInterval)
					positions = append(positions, interval.Start)

					if interval.Size > 1 {
						heap.Push(p.SizeIndex, &FreeInterval{
							Start: interval.Start + 1,
							Size:  interval.Size - 1,
						})
					}
				}
				p.mu.Unlock()
				if len(positions) == size {
					break
				}
			}
			if len(positions) == size {
				break
			}
		}
	}

	return positions
}

// 处理读取动作（并行化）
func (sys *StorageSystem) ReadAction(scanner *bufio.Scanner) {
	scanner.Scan()
	nRead, _ := strconv.Atoi(scanner.Text())

	// 处理新的读取请求
	for i := 0; i < nRead; i++ {
		scanner.Scan()
		parts := strings.Fields(scanner.Text())
		if len(parts) < 2 {
			continue
		}

		reqID, _ := strconv.Atoi(parts[0])
		objID, _ := strconv.Atoi(parts[1])

		// 初始化请求
		sys.Requests[reqID] = &Request{
			ID:       reqID,
			ObjectID: objID,
			PrevID:   sys.Objects[objID].LastRequest,
			IsDone:   false,
		}

		// 更新对象的最后请求
		sys.Objects[objID].LastRequest = reqID
	}

	// 为每个磁盘生成动作计划（并行）
	type diskPlan struct {
		id      int
		actions string
		newHead int
		token   int
	}

	diskPlans := make(chan diskPlan, sys.N)
	var wg sync.WaitGroup

	for diskID := 1; diskID <= sys.N; diskID++ {
		wg.Add(1)
		go func(did int) {
			defer wg.Done()
			disk := sys.Disks[did]
			disk.mu.Lock()
			defer disk.mu.Unlock()

			position := disk.HeadPosition
			tokensLeft := sys.G
			actions := ""
			newTokenCost := 0
			wasLastRead := false
			lastCost := disk.LastTokenCost

			for tokensLeft > 0 {
				objID := disk.Blocks[position]
				blockInfo := disk.BlockInfo[position]

				// 检查是否有对象块且未被删除
				if objID > 0 && !sys.Objects[objID].IsDeleted {
					blockIdx := blockInfo[2]
					obj := sys.Objects[objID]
					reqID := obj.LastRequest
					blockNeeded := false

					// 检查是否有请求需要这个块
					for reqID != 0 {
						req := sys.Requests[reqID]
						if req != nil && !req.IsDone {
							if !req.IsBlockRead(blockIdx) {
								blockNeeded = true
								break
							}
						}
						reqID = req.PrevID
					}

					if blockNeeded {
						// 计算读取成本
						var cost int
						if lastCost == 0 { // 首次读取
							cost = INITIAL_READ_COST
						} else {
							// 连续读取成本递减
							cost = (lastCost * 8) / 10 // 80% of last cost
							if cost < 16 {
								cost = 16
							}
						}

						if tokensLeft >= cost {
							// 执行读取操作
							actions += "r"
							tokensLeft -= cost
							lastCost = cost
							wasLastRead = true
							newTokenCost = cost

							// 标记所有相关请求的这个块为已读
							reqID = obj.LastRequest
							for reqID != 0 {
								req := sys.Requests[reqID]
								if req != nil && !req.IsDone {
									req.MarkBlockRead(blockIdx)
								}
								reqID = req.PrevID
							}
						} else {
							// 令牌不足，结束本次时间片
							break
						}
					} else {
						// 无需读取，执行pass操作
						if tokensLeft >= 1 {
							actions += "p"
							tokensLeft -= 1
							wasLastRead = false
						} else {
							break
						}
					}
				} else {
					// 没有对象块，执行pass操作
					if tokensLeft >= 1 {
						actions += "p"
						tokensLeft -= 1
						wasLastRead = false
					} else {
						break
					}
				}

				// 移动到下一个位置
				position = position%sys.V + 1
			}

			// 添加结束标记
			actions += "#"

			diskPlans <- diskPlan{
				id:      did,
				actions: actions,
				newHead: position,
				token:   newTokenCost,
			}
		}(diskID)
	}

	// 等待所有磁盘完成
	wg.Wait()
	close(diskPlans)

	// 收集并应用磁盘计划
	plans := make([]diskPlan, 0, sys.N)
	for plan := range diskPlans {
		plans = append(plans, plan)
	}

	// 按磁盘ID排序
	sort.Slice(plans, func(i, j int) bool {
		return plans[i].id < plans[j].id
	})

	// 输出磁盘动作
	for _, plan := range plans {
		fmt.Println(plan.actions)
		sys.Disks[plan.id].HeadPosition = plan.newHead
		sys.Disks[plan.id].LastTokenCost = plan.token
	}

	// 检查完成的请求（并行）
	completedRequests := make(chan int, 1000)
	var reqWg sync.WaitGroup

	// 分片处理请求
	chunkSize := MAX_REQUEST_NUM / 10
	if chunkSize == 0 {
		chunkSize = 1000
	}

	for start := 0; start < MAX_REQUEST_NUM; start += chunkSize {
		end := start + chunkSize
		if end > MAX_REQUEST_NUM {
			end = MAX_REQUEST_NUM
		}

		reqWg.Add(1)
		go func(start, end int) {
			defer reqWg.Done()
			for i := start; i < end; i++ {
				req := sys.Requests[i]
				if req == nil || req.IsDone {
					continue
				}

				obj := sys.Objects[req.ObjectID]
				if obj == nil {
					continue
				}

				if req.IsComplete(obj) {
					req.IsDone = true
					completedRequests <- req.ID
				}
			}
		}(start, end)
	}

	// 等待所有请求检查完成
	go func() {
		reqWg.Wait()
		close(completedRequests)
	}()

	// 收集完成的请求ID
	completedIDs := []int{}
	for id := range completedRequests {
		completedIDs = append(completedIDs, id)
	}

	// 排序并输出
	sort.Ints(completedIDs)
	fmt.Println(len(completedIDs))
	for _, id := range completedIDs {
		fmt.Println(id)
	}
}

// 主函数
func main() {
	scanner := bufio.NewScanner(os.Stdin)

	// 读取输入参数
	scanner.Scan()
	params := strings.Fields(scanner.Text())
	if len(params) < 5 {
		return
	}

	T, _ := strconv.Atoi(params[0])
	M, _ := strconv.Atoi(params[1])
	N, _ := strconv.Atoi(params[2])
	V, _ := strconv.Atoi(params[3])
	G, _ := strconv.Atoi(params[4])

	// 创建存储系统
	sys := NewStorageSystem(T, M, N, V, G)

	// 读取标签频率信息（忽略）
	for i := 1; i <= 3*M; i++ {
		for j := 1; j <= (T-1)/FRE_PER_SLICING+1; j++ {
			scanner.Scan()
		}
	}

	// 准备就绪
	fmt.Println("OK")

	// 主循环
	for t := 1; t <= T+EXTRA_TIME; t++ {
		scanner.Scan()
		command := scanner.Text()
		switch {
		case strings.HasPrefix(command, "TIMESTAMP"):
			sys.TimestampAction(scanner)
		case strings.HasPrefix(command, "DELETE"):
			sys.DeleteAction(scanner)
		case strings.HasPrefix(command, "WRITE"):
			sys.WriteAction(scanner)
		case strings.HasPrefix(command, "READ"):
			sys.ReadAction(scanner)
		}
	}
}