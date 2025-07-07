#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <utility>
#include <functional>
#include <climits>
#include <string>
#include <map>     // 用于实现区间树
#include <set>     // 用于去重

// 常量定义
#define MAX_DISK_NUM (10 + 1)
#define MAX_DISK_SIZE (16384 + 1)
#define MAX_REQUEST_NUM (30000000 + 1)
#define MAX_OBJECT_NUM (100000 + 1)
#define MAX_TAG_NUM (16 + 1)
#define REP_NUM (3)
#define FRE_PER_SLICING (1800)
#define EXTRA_TIME (105)
#define MAX_OBJECT_SIZE 5  // 题目限制对象最大大小为5

// 读取请求结构 - 极简设计
struct Request_ {
    int object_id;
    int prev_id;
    bool is_done;
    unsigned char blocks_read;  // 位图，每位表示一个块是否已读取
};

// 对象结构 - 使用固定大小数组替代动态分配内存
struct Object_ {
    int replica[REP_NUM + 1];
    int unit[REP_NUM + 1][MAX_OBJECT_SIZE + 1];  // 使用固定大小数组
    int size;
    int tag;
    int last_request_id;
    bool is_deleted;
};

// 全局变量
Request_ requests[MAX_REQUEST_NUM];
Object_ objects[MAX_OBJECT_NUM];

int T, M, N, V, G;
int disk[MAX_DISK_NUM][MAX_DISK_SIZE];        // 硬盘上存储的对象ID
int disk_head[MAX_DISK_NUM];                  // 每个硬盘的磁头位置
int disk_last_token[MAX_DISK_NUM];            // 每个磁头上一次操作消耗的令牌数
int disk_load[MAX_DISK_NUM];                  // 每个磁盘的负载（存储的块数）
int current_timestamp = 0;

// 区间树结构，跟踪连续空闲空间
// key: 空闲区间的大小
// value: 空闲区间的起始位置
std::multimap<int, int> free_intervals[MAX_DISK_NUM][MAX_TAG_NUM+1];

// 块信息，[disk][pos] = {obj_id, rep_id, block_idx}
int block_info[MAX_DISK_NUM][MAX_DISK_SIZE][3];

// 标签分区信息
int tag_partition_size;                         // 每个标签分区的大小
int tag_start_pos[MAX_TAG_NUM][MAX_DISK_NUM];  // 每个磁盘上每个标签的起始位置
int tag_next_pos[MAX_TAG_NUM][MAX_DISK_NUM];   // 每个磁盘上每个标签的下一个可用位置

// 用于快速检查位置是否被使用
bool is_used[MAX_DISK_NUM][MAX_DISK_SIZE];

// 检查块是否已读
inline bool is_block_read(int req_id, int block_idx) {
    return (requests[req_id].blocks_read & (1 << block_idx)) != 0;
}

// 标记块已读
inline void mark_block_read(int req_id, int block_idx) {
    requests[req_id].blocks_read |= (1 << block_idx);
}

// 检查请求是否完成
inline bool is_request_complete(int req_id) {
    int obj_id = requests[req_id].object_id;
    int size = objects[obj_id].size;
    
    // 检查所有块是否都已读取
    for (int i = 0; i < size; i++) {
        if (!is_block_read(req_id, i)) {
            return false;
        }
    }
    
    return true;
}

// 初始化标签分区 - 每个盘分成M份
void init_tag_partitions() {
    tag_partition_size = V / M;
    
    for (int tag = 1; tag <= M; tag++) {
        for (int disk_id = 1; disk_id <= N; disk_id++) {
            tag_start_pos[tag][disk_id] = (tag - 1) * tag_partition_size + 1;
            tag_next_pos[tag][disk_id] = tag_start_pos[tag][disk_id];
        }
    }
}

// 初始化区间树
void init_interval_trees() {
    for (int disk_id = 1; disk_id <= N; disk_id++) {
        for (int tag = 1; tag <= M; tag++) {
            int start = tag_start_pos[tag][disk_id];
            int end = (tag == M) ? V : (tag * tag_partition_size);
            // 添加整个分区作为一个空闲区间
            free_intervals[disk_id][tag].insert({end - start + 1, start});
        }
    }
}

// 标记存储单元为已使用
inline void mark_used(int disk_id, int position) {
    is_used[disk_id][position] = true;
}

// 标记存储单元为空闲
inline void mark_free(int disk_id, int position) {
    is_used[disk_id][position] = false;
}

// 从区间树中分配连续空间
void allocate_from_interval(int disk_id, int tag, int pos, int size) {
    int start = tag_start_pos[tag][disk_id];
    int end = (tag == M) ? V : (tag * tag_partition_size);
    
    // 从分区对应的所有区间中查找包含pos的区间
    for (auto it = free_intervals[disk_id][tag].begin(); it != free_intervals[disk_id][tag].end(); ++it) {
        int interval_size = it->first;
        int interval_start = it->second;
        int interval_end = interval_start + interval_size - 1;
        
        // 检查区间是否包含pos
        if (pos >= interval_start && pos + size - 1 <= interval_end) {
            // 从区间树中移除此区间
            free_intervals[disk_id][tag].erase(it);
            
            // 如果pos不是区间的开始，添加前面部分
            if (pos > interval_start) {
                free_intervals[disk_id][tag].insert({pos - interval_start, interval_start});
            }
            
            // 如果分配后区间尾部还有剩余，添加后面部分
            if (pos + size < interval_start + interval_size) {
                free_intervals[disk_id][tag].insert({interval_start + interval_size - (pos + size), pos + size});
            }
            
            // 标记分配的空间为已使用
            for (int i = 0; i < size; i++) {
                mark_used(disk_id, pos + i);
            }
            
            return;
        }
    }
}

// 将空闲空间添加回区间树
void add_free_interval(int disk_id, int tag, int pos, int size) {
    int start = tag_start_pos[tag][disk_id];
    int end = (tag == M) ? V : (tag * tag_partition_size);
    
    // 检查位置是否在分区内
    bool in_partition = (pos >= start && pos <= end);
    
    if (in_partition) {
        // 尝试与相邻的空闲区间合并
        bool merged = false;
        
        for (auto it = free_intervals[disk_id][tag].begin(); it != free_intervals[disk_id][tag].end(); ) {
            int interval_size = it->first;
            int interval_start = it->second;
            int interval_end = interval_start + interval_size - 1;
            
            // 检查是否可以向前合并
            if (interval_end + 1 == pos) {
                // 向前合并
                free_intervals[disk_id][tag].erase(it);
                free_intervals[disk_id][tag].insert({interval_size + size, interval_start});
                merged = true;
                break;
            }
            
            // 检查是否可以向后合并
            if (pos + size == interval_start) {
                // 向后合并
                free_intervals[disk_id][tag].erase(it);
                free_intervals[disk_id][tag].insert({interval_size + size, pos});
                merged = true;
                break;
            }
            
            ++it;
        }
        
        // 如果没有合并成功，作为新区间添加
        if (!merged) {
            free_intervals[disk_id][tag].insert({size, pos});
        }
    } else {
        // 不在分区内，直接添加到全局空闲区间
        for (int tag_id = 1; tag_id <= M; tag_id++) {
            if (pos >= tag_start_pos[tag_id][disk_id] && 
                pos <= (tag_id == M ? V : tag_id * tag_partition_size)) {
                free_intervals[disk_id][tag_id].insert({size, pos});
                break;
            }
        }
    }
    
    // 标记为空闲
    for (int i = 0; i < size; i++) {
        mark_free(disk_id, pos + i);
    }
}

// 使用区间树查找连续空间
int find_continuous_space(int disk_id, int tag, int size) {
    auto& intervals = free_intervals[disk_id][tag];
    
    // 优先查找大小最接近的区间
    auto it = intervals.lower_bound(size);
    if (it != intervals.end()) {
        int start_pos = it->second;
        int interval_size = it->first;
        
        // 从区间树中移除此区间
        intervals.erase(it);
        
        // 如果有剩余空间，将其加回树中
        if (interval_size > size) {
            intervals.insert({interval_size - size, start_pos + size});
        }
        
        // 标记分配的空间为已使用
        for (int i = 0; i < size; i++) {
            mark_used(disk_id, start_pos + i);
        }
        
        return start_pos;
    }
    
    // 如果在标签分区内找不到足够大的空间，尝试在其他分区或全盘查找
    for (int other_tag = 1; other_tag <= M; other_tag++) {
        if (other_tag == tag) continue;
        
        auto& other_intervals = free_intervals[disk_id][other_tag];
        auto other_it = other_intervals.lower_bound(size);
        
        if (other_it != other_intervals.end()) {
            int start_pos = other_it->second;
            int interval_size = other_it->first;
            
            // 从区间树中移除此区间
            other_intervals.erase(other_it);
            
            // 如果有剩余空间，将其加回树中
            if (interval_size > size) {
                other_intervals.insert({interval_size - size, start_pos + size});
            }
            
            // 标记分配的空间为已使用
            for (int i = 0; i < size; i++) {
                mark_used(disk_id, start_pos + i);
            }
            
            return start_pos;
        }
    }
    
    return -1; // 找不到足够的连续空间
}

// 查找分散空间
std::vector<int> find_scattered_space(int disk_id, int tag, int size) {
    std::vector<int> positions;
    positions.reserve(size);  // 预分配空间以避免多次内存分配
    
    // 首先在标签分区内找
    int start = tag_start_pos[tag][disk_id];
    int end = (tag == M) ? V : (tag * tag_partition_size);
    
    // 从小区间开始找单个空位
    for (auto it = free_intervals[disk_id][tag].begin(); it != free_intervals[disk_id][tag].end(); ) {
        int interval_start = it->second;
        int interval_size = it->first;
        
        if (interval_size >= 1) {
            // 每个区间只取一个位置
            positions.push_back(interval_start);
            
            // 更新区间
            if (interval_size > 1) {
                // 从区间树中移除此区间
                auto current = it++;
                free_intervals[disk_id][tag].erase(current);
                free_intervals[disk_id][tag].insert({interval_size - 1, interval_start + 1});
            } else {
                // 区间大小为1，直接移除
                auto current = it++;
                free_intervals[disk_id][tag].erase(current);
            }
            
            // 标记为已使用
            mark_used(disk_id, interval_start);
            
            if (positions.size() == size) {
                break;
            }
        } else {
            ++it;
        }
    }
    
    // 如果标签分区内空间不够，在其他分区查找
    if (positions.size() < size) {
        for (int other_tag = 1; other_tag <= M; other_tag++) {
            if (other_tag == tag) continue;
            
            auto& other_intervals = free_intervals[disk_id][other_tag];
            for (auto it = other_intervals.begin(); it != other_intervals.end(); ) {
                int interval_start = it->second;
                int interval_size = it->first;
                
                if (interval_size >= 1) {
                    // 每个区间只取一个位置
                    positions.push_back(interval_start);
                    
                    // 更新区间
                    if (interval_size > 1) {
                        // 从区间树中移除此区间
                        auto current = it++;
                        other_intervals.erase(current);
                        other_intervals.insert({interval_size - 1, interval_start + 1});
                    } else {
                        // 区间大小为1，直接移除
                        auto current = it++;
                        other_intervals.erase(current);
                    }
                    
                    // 标记为已使用
                    mark_used(disk_id, interval_start);
                    
                    if (positions.size() == size) {
                        break;
                    }
                } else {
                    ++it;
                }
            }
            
            if (positions.size() == size) {
                break;
            }
        }
    }
    
    return positions;
}

// 时间片对齐操作
void timestamp_action() {
    int timestamp;
    scanf("%*s%d", &timestamp);
    printf("TIMESTAMP %d\n", timestamp);
    current_timestamp = timestamp;
    fflush(stdout);
}

// 删除对象操作
void delete_action() {
    int n_delete;
    scanf("%d", &n_delete);
    
    if (n_delete == 0) {
        printf("0\n");
        fflush(stdout);
        return;
    }
    
    std::vector<int> obj_ids(n_delete);
    for (int i = 0; i < n_delete; i++) {
        scanf("%d", &obj_ids[i]);
    }
    
    // 计算需要取消的请求数量
    int abort_count = 0;
    std::vector<int> abort_requests;
    
    for (int i = 0; i < n_delete; i++) {
        int obj_id = obj_ids[i];
        int req_id = objects[obj_id].last_request_id;
        
        while (req_id != 0) {
            if (!requests[req_id].is_done) {
                abort_count++;
                abort_requests.push_back(req_id);
            }
            req_id = requests[req_id].prev_id;
        }
    }
    
    // 输出取消的请求
    printf("%d\n", abort_count);
    for (int req_id : abort_requests) {
        printf("%d\n", req_id);
    }
    
    // 释放硬盘空间
    for (int i = 0; i < n_delete; i++) {
        int obj_id = obj_ids[i];
        objects[obj_id].is_deleted = true;
        
        for (int rep = 1; rep <= REP_NUM; rep++) {
            int disk_id = objects[obj_id].replica[rep];
            int tag = objects[obj_id].tag;
            
            // 检查是否有连续的空间
            int first_unit = objects[obj_id].unit[rep][1];
            bool is_continuous = true;
            
            for (int j = 2; j <= objects[obj_id].size; j++) {
                if (objects[obj_id].unit[rep][j] != first_unit + j - 1) {
                    is_continuous = false;
                    break;
                }
            }
            
            if (is_continuous) {
                // 连续空间，直接添加一个区间
                add_free_interval(disk_id, tag, first_unit, objects[obj_id].size);
            } else {
                // 非连续空间，逐个添加
                for (int j = 1; j <= objects[obj_id].size; j++) {
                    int unit_id = objects[obj_id].unit[rep][j];
                    disk[disk_id][unit_id] = 0;
                    
                    // 添加单个空闲单元
                    add_free_interval(disk_id, tag, unit_id, 1);
                    
                    // 清除块信息
                    block_info[disk_id][unit_id][0] = 0;
                    block_info[disk_id][unit_id][1] = 0;
                    block_info[disk_id][unit_id][2] = 0;
                }
            }
            
            // 更新磁盘负载
            disk_load[disk_id] -= objects[obj_id].size;
        }
    }
    
    fflush(stdout);
}

// 写入对象操作 - 按标签分区策略
void write_action() {
    int n_write;
    scanf("%d", &n_write);
    
    if (n_write == 0) {
        return;
    }
    
    for (int i = 0; i < n_write; i++) {
        int id, size, tag;
        scanf("%d%d%d", &id, &size, &tag);
        
        objects[id].size = size;
        objects[id].tag = tag;
        objects[id].last_request_id = 0;
        objects[id].is_deleted = false;
        
        // 为每个副本选择硬盘 - 负载均衡策略
        bool disk_chosen[MAX_DISK_NUM] = {false};
        
        for (int rep = 1; rep <= REP_NUM; rep++) {
            // 选择负载最低的磁盘
            int best_disk = -1;
            int min_load = INT_MAX;
            
            for (int d = 1; d <= N; d++) {
                if (!disk_chosen[d] && disk_load[d] < min_load) {
                    min_load = disk_load[d];
                    best_disk = d;
                }
            }
            
            objects[id].replica[rep] = best_disk;
            disk_chosen[best_disk] = true;
            
            // 尝试连续写入
            int continuous_start = find_continuous_space(best_disk, tag, size);
            
            if (continuous_start != -1) {
                // 连续写入
                for (int j = 0; j < size; j++) {
                    int unit_id = continuous_start + j;
                    objects[id].unit[rep][j+1] = unit_id;
                    disk[best_disk][unit_id] = id;
                    
                    // 更新块信息
                    block_info[best_disk][unit_id][0] = id;
                    block_info[best_disk][unit_id][1] = rep;
                    block_info[best_disk][unit_id][2] = j;
                }
            } else {
                // 分散写入
                std::vector<int> positions = find_scattered_space(best_disk, tag, size);
                
                for (int j = 0; j < size; j++) {
                    int unit_id = positions[j];
                    objects[id].unit[rep][j+1] = unit_id;
                    disk[best_disk][unit_id] = id;
                    
                    // 更新块信息
                    block_info[best_disk][unit_id][0] = id;
                    block_info[best_disk][unit_id][1] = rep;
                    block_info[best_disk][unit_id][2] = j;
                }
            }
            
            // 更新磁盘负载
            disk_load[best_disk] += size;
        }
        
        // 输出写入结果
        printf("%d\n", id);
        for (int rep = 1; rep <= REP_NUM; rep++) {
            printf("%d", objects[id].replica[rep]);
            for (int j = 1; j <= size; j++) {
                printf(" %d", objects[id].unit[rep][j]);
            }
            printf("\n");
        }
    }
    
    fflush(stdout);
}

// 猛读策略 - 简单直接，有请求就读，没请求就pass
void read_action() {
    int n_read;
    scanf("%d", &n_read);
    
    // 处理新的读取请求
    for (int i = 0; i < n_read; i++) {
        int req_id, obj_id;
        scanf("%d%d", &req_id, &obj_id);
        
        // 初始化请求
        requests[req_id].object_id = obj_id;
        requests[req_id].prev_id = objects[obj_id].last_request_id;
        requests[req_id].is_done = false;
        requests[req_id].blocks_read = 0;
        
        // 更新对象的最后请求
        objects[obj_id].last_request_id = req_id;
    }
    
    // 为每个磁盘生成动作 - 纯粹的猛读策略
    std::string actions[MAX_DISK_NUM];
    std::vector<int> completed_requests;
    
    for (int disk_id = 1; disk_id <= N; disk_id++) {
        int position = disk_head[disk_id];
        int tokens_left = G;
        
        // 单次读取消耗的令牌数
        int read_cost = disk_last_token[disk_id];
        bool was_last_read = (read_cost > 0);
        
        std::string action = "";
        
        // 猛读策略：只要有令牌就一直读或pass
        while (tokens_left > 0) {
            // 检查当前位置是否有对象块
            int obj_id = block_info[disk_id][position][0];
            
            if (obj_id > 0 && !objects[obj_id].is_deleted) {
                // 有对象块，检查是否有需要读取这个块的请求
                int block_idx = block_info[disk_id][position][2];
                int req_id = objects[obj_id].last_request_id;
                bool block_needed = false;
                
                // 检查所有相关请求
                while (req_id != 0) {
                    if (!requests[req_id].is_done && !is_block_read(req_id, block_idx)) {
                        block_needed = true;
                        break;
                    }
                    req_id = requests[req_id].prev_id;
                }
                
                if (block_needed) {
                    // 需要读取这个块
                    // 计算读取成本
                    int new_read_cost;
                    if (!was_last_read) {
                        new_read_cost = 64;  // 首次读取成本
                    } else {
                        // 连续读取成本递减
                        new_read_cost = std::max(16, (read_cost * 8 + 9) / 10);  // 向上取整
                    }
                    
                    // 检查是否有足够的令牌
                    if (tokens_left >= new_read_cost) {
                        // 执行读取操作
                        action += "r";
                        tokens_left -= new_read_cost;
                        read_cost = new_read_cost;  // 更新read_cost为本次使用的值
                        was_last_read = true;
                        
                        // 标记所有相关请求的这个块为已读
                        req_id = objects[obj_id].last_request_id;
                        while (req_id != 0) {
                            if (!requests[req_id].is_done && !is_block_read(req_id, block_idx)) {
                                mark_block_read(req_id, block_idx);
                                
                                // 检查请求是否已完成
                                if (is_request_complete(req_id)) {
                                    requests[req_id].is_done = true;
                                    completed_requests.push_back(req_id);
                                }
                            }
                            req_id = requests[req_id].prev_id;
                        }
                    } else {
                        // 令牌不足，结束本次时间片
                        break;
                    }
                } else {
                    // 无需读取这个块，执行pass操作
                    if (tokens_left >= 1) {
                        action += "p";
                        tokens_left -= 1;
                        was_last_read = false;
                    } else {
                        // 令牌不足，结束本次时间片
                        break;
                    }
                }
            } else {
                // 没有对象块，执行pass操作
                if (tokens_left >= 1) {
                    action += "p";
                    tokens_left -= 1;
                    was_last_read = false;
                } else {
                    // 令牌不足，结束本次时间片
                    break;
                }
            }
            
            // 移动到下一个位置
            position = position % V + 1;
        }
        
        // 添加结束标记
        action += "#";
        actions[disk_id] = action;
        
        // 更新磁头位置
        disk_head[disk_id] = position;
        
        // 更新上一次读取成本
        if (was_last_read) {
            disk_last_token[disk_id] = read_cost;
        } else {
            disk_last_token[disk_id] = 0;
        }
    }
    
    // 输出磁头动作
    for (int i = 1; i <= N; i++) {
        printf("%s\n", actions[i].c_str());
    }
    
    // 去重并输出完成的请求
    std::sort(completed_requests.begin(), completed_requests.end());
    completed_requests.erase(
        std::unique(completed_requests.begin(), completed_requests.end()),
        completed_requests.end());
    
    printf("%d\n", (int)completed_requests.size());
    for (int req_id : completed_requests) {
        printf("%d\n", req_id);
    }
    
    fflush(stdout);
}

// 主函数
int main() {
    // 读取输入
    scanf("%d%d%d%d%d", &T, &M, &N, &V, &G);
    
    // 读取但忽略标签信息
    for (int i = 1; i <= 3 * M; i++) {
        for (int j = 1; j <= (T - 1) / FRE_PER_SLICING + 1; j++) {
            int freq;
            scanf("%d", &freq);
        }
    }
    
    // 初始化
    memset(disk, 0, sizeof(disk));
    memset(block_info, 0, sizeof(block_info));
    memset(disk_load, 0, sizeof(disk_load));
    memset(is_used, 0, sizeof(is_used));
    
    for (int i = 1; i <= N; i++) {
        disk_head[i] = 1;
        disk_last_token[i] = 0;
    }
    
    // 初始化标签分区
    init_tag_partitions();
    
    // 初始化区间树
    init_interval_trees();
    
    // 准备就绪
    printf("OK\n");
    fflush(stdout);
    
    // 主交互循环
    for (int t = 1; t <= T + EXTRA_TIME; t++) {
        timestamp_action();
        delete_action();
        write_action();
        read_action();
    }
    
    return 0;
}