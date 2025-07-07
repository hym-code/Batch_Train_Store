#pragma once
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <functional>
#include <queue>

namespace DistributedStorage {

// ====================== 基础数据结构 ======================

/**
 * @brief 节点状态枚举
 * 
 * 表示集群中节点的当前状态
 */
enum class NodeStatus {
    FOLLOWER,    // 跟随者状态
    CANDIDATE,   // 候选者状态
    LEADER,      // 领导者状态
    RECOVERING   // 恢复中状态
};

/**
 * @brief 操作类型枚举
 * 
 * 表示可以在状态机上执行的操作类型
 */
enum class OperationType {
    READ,        // 读操作
    WRITE,       // 写操作
    DELETE,      // 删除操作
    MIGRATE,     // 数据迁移操作
    SNAPSHOT     // 快照操作
};

/**
 * @brief 链路类型枚举
 * 
 * 表示请求处理的链路类型
 */
enum class LinkType {
    SHORT_TERM,  // 短链路 - 快速处理
    LONG_TERM    // 长链路 - 复杂/持久化处理
};

// ====================== Paxos 协议相关结构 ======================

/**
 * @brief Paxos提案ID
 * 
 * 唯一标识一个Paxos提案
 */
struct ProposalID {
    uint64_t number;      // 提案编号
    int node_id;          // 提出提案的节点ID

    bool operator<(const ProposalID& other) const {
        return number < other.number || 
              (number == other.number && node_id < other.node_id);
    }
    
    bool operator==(const ProposalID& other) const {
        return number == other.number && node_id == other.node_id;
    }
};

/**
 * @brief Paxos提案
 * 
 * 包含提案内容和元数据
 */
struct PaxosProposal {
    ProposalID id;                      // 提案ID
    std::vector<uint8_t> operation;     // 序列化的操作数据
    OperationType op_type;              // 操作类型
    LinkType link_type;                 // 链路类型
    uint64_t client_request_id;         // 客户端请求ID
};

/**
 * @brief Paxos承诺
 * 
 * 节点对提案的响应
 */
struct PaxosPromise {
    bool accepted;                      // 是否接受提案
    ProposalID highest_accepted_id;     // 已接受的最高提案ID
    PaxosProposal highest_accepted;     // 已接受的最高提案
};

/**
 * @brief Paxos接受请求
 * 
 * 请求接受特定提案
 */
struct PaxosAcceptRequest {
    PaxosProposal proposal;             // 待接受的提案
    int proposer_id;                    // 提案者ID
};

/**
 * @brief Paxos学习请求
 * 
 * 通知节点已达成共识的提案
 */
struct PaxosLearnRequest {
    PaxosProposal accepted_proposal;    // 已接受的提案
    int learner_id;                     // 学习节点ID
};

// ====================== 存储系统核心结构 ======================

/**
 * @brief 数据对象元数据
 * 
 * 存储系统中数据对象的元信息
 */
struct ObjectMetadata {
    std::string object_id;              // 对象唯一ID
    size_t size;                        // 对象大小
    uint32_t crc;                       // 校验和
    int primary_node;                   // 主节点ID
    std::vector<int> replica_nodes;     // 副本节点ID列表
    int64_t last_access;                // 最后访问时间戳
    int access_count;                   // 访问计数
    bool is_hot;                        // 是否为热数据
};

/**
 * @brief 存储节点信息
 * 
 * 集群中存储节点的状态信息
 */
struct StorageNode {
    int node_id;                        // 节点唯一ID
    NodeStatus status;                  // 节点状态
    uint64_t current_term;              // 当前任期
    int voted_for;                      // 当前任期投票给的节点ID
    size_t available_space;             // 可用空间
    size_t total_space;                 // 总空间
    int load_factor;                    // 负载因子
    std::string endpoint;               // 网络端点
    std::atomic<int> heartbeat_count;   // 心跳计数
};

/**
 * @brief 内存索引条目
 * 
 * 用于快速定位对象位置
 */
struct MemoryIndexEntry {
    std::string object_id;              // 对象ID
    size_t offset;                      // 对象偏移量
    size_t length;                      // 对象长度
    int storage_node;                   // 存储节点ID
    bool is_primary;                    // 是否为主副本
};

/**
 * @brief 短链路上下文
 * 
 * 短链路请求处理上下文
 */
struct ShortLinkContext {
    uint64_t request_id;                // 请求ID
    std::string object_id;              // 对象ID
    int client_id;                      // 客户端ID
    int64_t start_time;                 // 请求开始时间
    std::vector<uint8_t> data;          // 读取的数据
    std::atomic<bool> completed;        // 是否完成
    std::mutex mutex;                   // 互斥锁
    std::condition_variable cv;         // 条件变量
};

/**
 * @brief 长链路上下文
 * 
 * 长链路请求处理上下文
 */
struct LongLinkContext {
    uint64_t request_id;                // 请求ID
    OperationType op_type;              // 操作类型
    std::string object_id;              // 对象ID
    std::vector<uint8_t> payload;       // 操作负载
    int primary_node;                   // 主节点ID
    std::vector<int> involved_nodes;    // 涉及节点列表
    std::atomic<int> completed_steps;   // 已完成步骤
    int total_steps;                    // 总步骤数
    LinkType link_type;                 // 链路类型
    std::function<void(bool)> callback; // 完成回调
};

// ====================== 内存管理结构 ======================

/**
 * @brief 内存块描述符
 * 
 * 描述内存中数据块的分配情况
 */
struct MemoryChunk {
    void* address;                      // 内存地址
    size_t size;                        // 块大小
    bool allocated;                     // 是否已分配
    int access_count;                   // 访问计数
    int64_t last_access;                // 最后访问时间
    int priority;                       // 优先级
    std::vector<uint8_t> hash;          // 数据哈希
};

/**
 * @brief 内存区域
 * 
 * 管理连续的内存区域
 */
struct MemoryRegion {
    uint8_t* base_address;              // 基地址
    size_t total_size;                  // 总大小
    size_t used_size;                   // 已用大小
    std::vector<MemoryChunk> chunks;    // 内存块列表
    std::mutex region_mutex;            // 区域互斥锁
};

// ====================== 缓存结构 ======================

/**
 * @brief 缓存条目
 * 
 * 缓存系统中的数据条目
 */
struct CacheEntry {
    std::string key;                    // 缓存键
    std::vector<uint8_t> value;         // 缓存值
    int64_t expire_time;                // 过期时间
    int access_count;                   // 访问计数
    int size;                           // 条目大小
    bool dirty;                         // 是否为脏数据
    uint32_t crc;                       // 校验和
};

// ====================== 统计监控结构 ======================

/**
 * @brief 运行时统计
 * 
 * 系统运行时统计数据
 */
struct RuntimeStatistics {
    std::atomic<uint64_t> read_ops;             // 读操作计数
    std::atomic<uint64_t> write_ops;            // 写操作计数
    std::atomic<uint64_t> delete_ops;           // 删除操作计数
    std::atomic<uint64_t> short_link_ops;       // 短链路操作计数
    std::atomic<uint64_t> long_link_ops;        // 长链路操作计数
    std::atomic<uint64_t> paxos_proposals;      // Paxos提案计数
    std::atomic<uint64_t> paxos_timeouts;       // Paxos超时计数
    std::atomic<uint64_t> bytes_read;           // 读取字节数
    std::atomic<uint64_t> bytes_written;        // 写入字节数
    std::atomic<uint64_t> cache_hits;           // 缓存命中数
    std::atomic<uint64_t> cache_misses;         // 缓存未命中数
};

// ====================== 核心系统类 ======================

/**
 * @brief Paxos共识模块
 * 
 * 实现Paxos共识算法
 */
class PaxosConsensus {
public:
    PaxosConsensus(int node_id, std::vector<int> cluster_nodes);
    
    /**
     * @brief 准备阶段
     * @param proposal 提案
     * @return 承诺响应集合
     */
    std::map<int, PaxosPromise> preparePhase(const PaxosProposal& proposal);
    
    /**
     * @brief 接受阶段
     * @param proposal 提案
     * @param promises 承诺集合
     * @return 是否被接受
     */
    bool acceptPhase(const PaxosProposal& proposal, 
                    const std::map<int, PaxosPromise>& promises);
    
    /**
     * @brief 学习阶段
     * @param proposal 已接受的提案
     */
    void learnPhase(const PaxosProposal& proposal);
    
    /**
     * @brief 处理准备请求
     * @param proposal 提案
     * @return 承诺响应
     */
    PaxosPromise handlePrepareRequest(const PaxosProposal& proposal);
    
    /**
     * @brief 处理接受请求
     * @param request 接受请求
     * @return 是否接受
     */
    bool handleAcceptRequest(const PaxosAcceptRequest& request);
    
    /**
     * @brief 处理学习请求
     * @param request 学习请求
     */
    void handleLearnRequest(const PaxosLearnRequest& request);
    
private:
    int node_id_;                                // 当前节点ID
    std::vector<int> cluster_nodes_;             // 集群节点列表
    ProposalID highest_proposal_id_;             // 已知的最高提案ID
    ProposalID highest_accepted_id_;             // 已接受的最高提案ID
    PaxosProposal highest_accepted_proposal_;    // 已接受的最高提案
    std::mutex paxos_mutex_;                     // Paxos互斥锁
    std::atomic<uint64_t> current_term_;         // 当前任期
};

/**
 * @brief 状态机模块
 * 
 * 实现状态机逻辑
 */
class StateMachine {
public:
    StateMachine();
    
    /**
     * @brief 应用操作到状态机
     * @param operation 操作数据
     */
    void apply(const std::vector<uint8_t>& operation);
    
    /**
     * @brief 获取当前状态
     * @return 状态快照
     */
    std::vector<uint8_t> getSnapshot() const;
    
    /**
     * @brief 从快照恢复状态
     * @param snapshot 状态快照
     */
    void restoreFromSnapshot(const std::vector<uint8_t>& snapshot);
    
    /**
     * @brief 执行读操作
     * @param object_id 对象ID
     * @return 对象数据
     */
    std::vector<uint8_t> executeRead(const std::string& object_id);
    
    /**
     * @brief 执行写操作
     * @param object_id 对象ID
     * @param data 对象数据
     */
    void executeWrite(const std::string& object_id, const std::vector<uint8_t>& data);
    
    /**
     * @brief 执行删除操作
     * @param object_id 对象ID
     */
    void executeDelete(const std::string& object_id);
    
private:
    std::map<std::string, std::vector<uint8_t>> storage_;  // 存储状态
    std::map<std::string, ObjectMetadata> metadata_;       // 元数据
    std::mutex state_mutex_;                               // 状态互斥锁
    uint64_t last_applied_index_;                          // 最后应用的日志索引
};

/**
 * @brief 链路管理器
 * 
 * 管理长短链路请求
 */
class LinkManager {
public:
    LinkManager(int node_id, PaxosConsensus& paxos, StateMachine& state_machine);
    
    /**
     * @brief 处理短链路请求
     * @param context 请求上下文
     */
    void processShortLink(ShortLinkContext& context);
    
    /**
     * @brief 处理长链路请求
     * @param context 请求上下文
     */
    void processLongLink(LongLinkContext& context);
    
    /**
     * @brief 提交Paxos提案
     * @param proposal 提案
     * @return 是否成功提交
     */
    bool submitPaxosProposal(const PaxosProposal& proposal);
    
    /**
     * @brief 处理超时请求
     */
    void handleTimeouts();
    
private:
    int node_id_;                                // 当前节点ID
    PaxosConsensus& paxos_;                      // Paxos共识模块引用
    StateMachine& state_machine_;                // 状态机引用
    std::queue<ShortLinkContext> short_queue_;   // 短链路队列
    std::queue<LongLinkContext> long_queue_;     // 长链路队列
    std::mutex queue_mutex_;                     // 队列互斥锁
    std::condition_variable queue_cv_;           // 队列条件变量
    std::vector<std::thread> worker_threads_;    // 工作线程
    std::atomic<bool> running_;                  // 运行标志
    
    void workerThreadFunction();                 // 工作线程函数
};

/**
 * @brief 分布式存储系统
 * 
 * 系统主类，整合所有组件
 */
class DistributedStorageSystem {
public:
    DistributedStorageSystem(int node_id, 
                            const std::vector<int>& cluster_nodes,
                            size_t storage_size);
    
    ~DistributedStorageSystem();
    
    /**
     * @brief 启动系统
     */
    void start();
    
    /**
     * @brief 停止系统
     */
    void stop();
    
    /**
     * @brief 处理客户端请求
     * @param request 请求数据
     * @return 响应数据
     */
    std::vector<uint8_t> handleClientRequest(const std::vector<uint8_t>& request);
    
    /**
     * @brief 读取对象
     * @param object_id 对象ID
     * @return 对象数据
     */
    std::vector<uint8_t> readObject(const std::string& object_id);
    
    /**
     * @brief 写入对象
     * @param object_id 对象ID
     * @param data 对象数据
     */
    void writeObject(const std::string& object_id, const std::vector<uint8_t>& data);
    
    /**
     * @brief 删除对象
     * @param object_id 对象ID
     */
    void deleteObject(const std::string& object_id);
    
    /**
     * @brief 获取运行时统计
     * @return 运行时统计
     */
    RuntimeStatistics getRuntimeStats() const;
    
private:
    int node_id_;                                 // 当前节点ID
    PaxosConsensus paxos_;                        // Paxos共识模块
    StateMachine state_machine_;                  // 状态机
    LinkManager link_manager_;                    // 链路管理器
    MemoryRegion memory_region_;                  // 内存区域
    std::map<std::string, CacheEntry> cache_;     // 缓存
    RuntimeStatistics stats_;                     // 运行时统计
    std::thread heartbeat_thread_;                // 心跳线程
    std::atomic<bool> system_running_;            // 系统运行标志
    
    void heartbeatFunction();                     // 心跳线程函数
    void updateNodeStatus(NodeStatus new_status); // 更新节点状态
};

} // namespace DistributedStorage