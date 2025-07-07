Distributed Storage System Simulator Design Overview
This system simulates a distributed Batch-Train storage environment with optimized space management and I/O operations. Key design elements combine efficiency-focused data structures with intelligent allocation strategies:
Core Architecture
Tag-partitioned Storage
Each disk is divided into M equal-sized partitions (except last). Objects are placed using tag affinity - preferentially stored in partitions matching their tag.

Replica Management
Every object maintains REP_NUM (default=3) replicas across different disks using load-aware selection.

Space Allocation Engine
Interval Tree Management
Uses std::multimap<int, int> per partition to track free space:

Key: Free block size

Value: Starting position

Supports O(log n) operations for allocation/deallocation

Two-tier Allocation Strategy

Continuous Allocation
First-fit using lower_bound(size) in tag-matched partition

Scattered Allocation
When continuous space unavailable, gather single blocks from multiple intervals.

Performance Optimizations
Fixed-size arrays avoid dynamic allocation overhead

Load-balanced writes prevent disk hotspots

Bulk operations minimize output flushing

O(1) position checks through precomputed tag boundaries

This hybrid design balances space efficiency through interval tree management with I/O performance via adaptive read costing and head movement optimization, providing a high-fidelity distributed storage simulation environment.
