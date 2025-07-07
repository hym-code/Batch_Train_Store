#pragma once
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <memory>
#include <functional>
#include <algorithm>
#include <queue>
#include <cmath>
#include <random>
#include <iostream>

namespace ExplainableDRL {

// ====================== 基础结构体 ======================

/**
 * @brief 节点类型枚举
 * 
 * 表示思维链中不同类型的节点
 */
enum class NodeType {
    SENSOR_INPUT,       // 传感器输入节点
    FEATURE_EXTRACTION, // 特征提取节点
    SITUATION_ANALYSIS, // 情境分析节点
    DECISION_MAKING,    // 决策节点
    CONTROL_ACTION,     // 控制动作节点
    CAUSAL_RELATION,    // 因果关系节点
    UNCERTAINTY         // 不确定性节点
};

/**
 * @brief 车辆状态结构体
 * 
 * 表示车辆当前的状态信息
 */
struct VehicleState {
    double timestamp;           // 时间戳
    double velocity;            // 速度 (m/s)
    double acceleration;        // 加速度 (m/s²)
    double steering_angle;      // 方向盘角度 (度)
    double position_x;          // X位置 (m)
    double position_y;          // Y位置 (m)
    double heading;             // 航向角 (度)
    
    // 环境信息
    double distance_to_obstacle; // 到最近障碍物的距离 (m)
    double road_curvature;       // 道路曲率
    bool traffic_light_state;    // 交通灯状态 (true=绿灯)
    bool pedestrian_present;     // 是否有行人
    
    // 序列化状态信息
    std::vector<double> serialize() const {
        return {timestamp, velocity, acceleration, steering_angle,
                position_x, position_y, heading, distance_to_obstacle,
                road_curvature, static_cast<double>(traffic_light_state),
                static_cast<double>(pedestrian_present)};
    }
};

// ====================== DAG核心结构 ======================

/**
 * @brief DAG节点基类
 * 
 * 表示思维链中的一个节点
 */
template <typename T>
struct DAGNode {
    std::string id;                 // 节点唯一标识
    NodeType type;                  // 节点类型
    double activation_level;        // 激活水平 (0.0-1.0)
    double importance_weight;       // 重要性权重
    std::vector<T> data;            // 节点数据
    std::vector<std::string> causal_factors; // 因果因素
    
    // 时间信息
    double creation_time;           // 创建时间戳
    double last_update_time;        // 最后更新时间戳
    
    // 构造函数
    DAGNode(std::string id, NodeType type, double timestamp)
        : id(id), type(type), activation_level(0.0), 
          importance_weight(1.0), creation_time(timestamp),
          last_update_time(timestamp) {}
    
    virtual ~DAGNode() = default;
    
    // 更新节点
    virtual void update(double timestamp, const VehicleState& state) {
        last_update_time = timestamp;
    }
    
    // 计算节点影响
    virtual double computeInfluence() const {
        return activation_level * importance_weight;
    }
};

/**
 * @brief 传感器输入节点
 * 
 * 处理原始传感器数据
 */
template <typename T>
struct SensorInputNode : public DAGNode<T> {
    std::string sensor_type;        // 传感器类型
    double reliability;             // 可靠性指标
    
    SensorInputNode(std::string id, double timestamp, std::string sensor_type)
        : DAGNode<T>(id, NodeType::SENSOR_INPUT, timestamp), 
          sensor_type(sensor_type), reliability(0.95) {}
    
    void update(double timestamp, const VehicleState& state) override {
        DAGNode<T>::update(timestamp, state);
        
        // 根据车辆状态更新激活水平
        if (sensor_type == "LIDAR") {
            this->activation_level = 1.0 - std::exp(-state.distance_to_obstacle / 20.0);
        } else if (sensor_type == "CAMERA") {
            this->activation_level = state.pedestrian_present ? 0.9 : 0.3;
        }
    }
};

/**
 * @brief 决策节点
 * 
 * 表示决策点
 */
template <typename T>
struct DecisionNode : public DAGNode<T> {
    std::string decision_type;      // 决策类型
    double confidence;              // 决策置信度
    
    DecisionNode(std::string id, double timestamp, std::string decision_type)
        : DAGNode<T>(id, NodeType::DECISION_MAKING, timestamp), 
          decision_type(decision_type), confidence(0.8) {}
    
    double computeInfluence() const override {
        return DAGNode<T>::computeInfluence() * confidence;
    }
};

/**
 * @brief 控制动作节点
 * 
 * 表示最终控制动作
 */
template <typename T>
struct ControlActionNode : public DAGNode<T> {
    std::string action_type;        // 动作类型
    double intensity;               // 动作强度
    
    ControlActionNode(std::string id, double timestamp, std::string action_type)
        : DAGNode<T>(id, NodeType::CONTROL_ACTION, timestamp), 
          action_type(action_type), intensity(0.0) {}
    
    void update(double timestamp, const VehicleState& state) override {
        DAGNode<T>::update(timestamp, state);
        
        // 根据动作类型设置强度
        if (action_type == "BRAKE") {
            intensity = std::min(1.0, state.acceleration * -0.5);
        } else if (action_type == "ACCELERATE") {
            intensity = std::max(0.0, state.acceleration * 0.3);
        } else if (action_type == "STEER") {
            intensity = std::abs(state.steering_angle) / 30.0;
        }
        
        this->activation_level = intensity;
    }
};

// ====================== 思维链结构 ======================

/**
 * @brief 思维链路
 * 
 * 表示从输入到决策的一条完整思维链
 */
template <typename T>
struct ThoughtChain {
    std::vector<std::shared_ptr<DAGNode<T>>> nodes; // 链路上的节点
    double chain_strength;                          // 链路强度
    double timestamp;                               // 链路时间戳
    double priority;                                // 链路优先级
    
    // 计算链路强度
    void computeStrength() {
        chain_strength = 0.0;
        for (const auto& node : nodes) {
            chain_strength += node->computeInfluence();
        }
        chain_strength /= nodes.size();
    }
    
    // 更新链路优先级
    void updatePriority(double current_time) {
        double recency_factor = 1.0 / (1.0 + std::exp(-(current_time - timestamp)));
        priority = chain_strength * recency_factor;
    }
};

/**
 * @brief 主次链路分类器
 * 
 * 根据规则对思维链进行分类
 */
template <typename T>
struct ChainClassifier {
    double primary_threshold;  // 主链路阈值
    double secondary_threshold; // 次链路阈值
    
    ChainClassifier(double primary = 0.7, double secondary = 0.4)
        : primary_threshold(primary), secondary_threshold(secondary) {}
    
    // 对链路进行分类
    std::string classify(const ThoughtChain<T>& chain) const {
        if (chain.chain_strength >= primary_threshold) {
            return "PRIMARY";
        } else if (chain.chain_strength >= secondary_threshold) {
            return "SECONDARY";
        }
        return "BACKGROUND";
    }
};

// ====================== 潜在变量与视图 ======================

/**
 * @brief 潜在变量
 * 
 * 表示非线性潜在变量
 */
struct LatentVariable {
    std::string id;                 // 变量ID
    std::vector<double> embeddings; // 嵌入向量
    std::vector<std::string> causal_links; // 因果链接
    
    // 计算相似度
    double similarity(const LatentVariable& other) const {
        if (embeddings.empty() || other.embeddings.empty()) return 0.0;
        
        double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
        for (size_t i = 0; i < embeddings.size(); ++i) {
            dot += embeddings[i] * other.embeddings[i];
            norm1 += embeddings[i] * embeddings[i];
            norm2 += other.embeddings[i] * other.embeddings[i];
        }
        return dot / (std::sqrt(norm1) * std::sqrt(norm2));
    }
};

/**
 * @brief 视图
 * 
 * 表示一个视图，包含一组潜在变量
 */
template <typename T>
struct View {
    std::string id;                                 // 视图ID
    std::vector<LatentVariable> latent_variables;   // 潜在变量
    std::vector<std::shared_ptr<DAGNode<T>>> nodes; // 关联的DAG节点
    
    // 从节点生成潜在变量
    void generateLatentVariables() {
        latent_variables.clear();
        for (const auto& node : nodes) {
            LatentVariable lv;
            lv.id = node->id;
            // 简化示例：使用节点数据作为嵌入
            for (const auto& d : node->data) {
                lv.embeddings.push_back(static_cast<double>(d));
            }
            latent_variables.push_back(lv);
        }
    }
    
    // 计算视图相似度
    double similarity(const View<T>& other) const {
        if (latent_variables.empty() || other.latent_variables.empty()) return 0.0;
        
        double total_sim = 0.0;
        int count = 0;
        for (const auto& lv1 : latent_variables) {
            for (const auto& lv2 : other.latent_variables) {
                total_sim += lv1.similarity(lv2);
                count++;
            }
        }
        return count > 0 ? total_sim / count : 0.0;
    }
};

// ====================== 对比学习与编码器 ======================

/**
 * @brief 对比学习模块
 * 
 * 用于学习视图之间的共享信息
 */
template <typename T>
struct ContrastiveLearner {
    double temperature; // 温度参数
    
    ContrastiveLearner(double temp = 0.5) : temperature(temp) {}
    
    // 计算对比损失
    double computeLoss(const View<T>& view1, const View<T>& view2) const {
        double sim = view1.similarity(view2);
        double loss = -std::log(std::exp(sim / temperature));
        return loss;
    }
    
    // 更新视图
    void updateView(View<T>& view, const std::vector<View<T>>& all_views) {
        view.generateLatentVariables();
    }
};

/**
 * @brief 编码器
 * 
 * 用于学习潜在变量之间的平滑双射
 */
template <typename T>
struct Encoder {
    std::map<std::string, std::vector<double>> embeddings; // 节点嵌入
    
    // 编码节点
    std::vector<double> encode(const DAGNode<T>& node) {
        // 简化示例：使用节点数据作为嵌入
        std::vector<double> result;
        for (const auto& d : node.data) {
            result.push_back(static_cast<double>(d));
        }
        embeddings[node.id] = result;
        return result;
    }
    
    // 学习双射映射
    void learnSmoothBijection(const std::vector<DAGNode<T>>& nodes) {
        // 在实际实现中，这里会使用神经网络
        for (const auto& node : nodes) {
            encode(node);
        }
    }
};

// ====================== DAG核心类 ======================

/**
 * @brief 有向无环图 (DAG)
 * 
 * 用于表示智能驾驶决策的思维链
 */
template <typename T>
class ExplanationDAG {
private:
    std::unordered_map<std::string, std::shared_ptr<DAGNode<T>>> nodes;
    std::unordered_map<std::string, std::vector<std::string>> adjacency_list;
    std::vector<ThoughtChain<T>> chains;
    std::vector<View<T>> views;
    
    // 优先级规则约束
    std::function<double(const ThoughtChain<T>&)> priority_rule;
    
    // 私有辅助函数
    void findChainsDFS(const std::string& current, 
                      const std::string& end, 
                      std::vector<std::string>& path,
                      std::set<std::string>& visited,
                      std::vector<std::vector<std::string>>& all_paths) {
        visited.insert(current);
        path.push_back(current);
        
        if (current == end) {
            all_paths.push_back(path);
        } else {
            for (const auto& neighbor : adjacency_list[current]) {
                if (visited.find(neighbor) == visited.end()) {
                    findChainsDFS(neighbor, end, path, visited, all_paths);
                }
            }
        }
        
        path.pop_back();
        visited.erase(current);
    }
    
public:
    ExplanationDAG() {
        // 默认优先级规则：基于链路强度
        priority_rule = [](const ThoughtChain<T>& chain) {
            return chain.chain_strength;
        };
    }
    
    // ====================== DAG操作 ======================
    
    /**
     * @brief 添加节点
     * @param node 要添加的节点
     */
    void addNode(std::shared_ptr<DAGNode<T>> node) {
        nodes[node->id] = node;
        adjacency_list[node->id] = std::vector<std::string>();
    }
    
    /**
     * @brief 添加边
     * @param from 源节点ID
     * @param to 目标节点ID
     */
    void addEdge(const std::string& from, const std::string& to) {
        if (nodes.find(from) != nodes.end() && nodes.find(to) != nodes.end()) {
            adjacency_list[from].push_back(to);
        }
    }
    
    /**
     * @brief 移除节点
     * @param id 节点ID
     */
    void removeNode(const std::string& id) {
        if (nodes.find(id) == nodes.end()) return;
        
        // 移除节点
        nodes.erase(id);
        adjacency_list.erase(id);
        
        // 移除所有相关边
        for (auto& [source, targets] : adjacency_list) {
            targets.erase(std::remove(targets.begin(), targets.end(), id), targets.end());
        }
    }
    
    /**
     * @brief 更新节点
     * @param id 节点ID
     * @param timestamp 时间戳
     * @param state 车辆状态
     */
    void updateNode(const std::string& id, double timestamp, const VehicleState& state) {
        if (nodes.find(id) != nodes.end()) {
            nodes[id]->update(timestamp, state);
        }
    }
    
    /**
     * @brief 查找所有思维链
     * 
     * 从输入节点到控制动作节点的所有路径
     */
    void findThoughtChains() {
        chains.clear();
        
        // 找到所有输入节点和输出节点
        std::vector<std::string> input_nodes;
        std::vector<std::string> output_nodes;
        
        for (const auto& [id, node] : nodes) {
            if (node->type == NodeType::SENSOR_INPUT) {
                input_nodes.push_back(id);
            } else if (node->type == NodeType::CONTROL_ACTION) {
                output_nodes.push_back(id);
            }
        }
        
        // 对于每对输入-输出，查找所有路径
        for (const auto& input : input_nodes) {
            for (const auto& output : output_nodes) {
                std::vector<std::vector<std::string>> all_paths;
                std::vector<std::string> path;
                std::set<std::string> visited;
                
                findChainsDFS(input, output, path, visited, all_paths);
                
                // 将路径转换为思维链
                for (const auto& p : all_paths) {
                    ThoughtChain<T> chain;
                    chain.timestamp = nodes[p[0]]->creation_time;
                    
                    for (const auto& node_id : p) {
                        chain.nodes.push_back(nodes[node_id]);
                    }
                    
                    chain.computeStrength();
                    chains.push_back(chain);
                }
            }
        }
    }
    
    // ====================== 链路优先级处理 ======================
    
    /**
     * @brief 设置优先级规则
     * @param rule 优先级规则函数
     */
    void setPriorityRule(std::function<double(const ThoughtChain<T>&)> rule) {
        priority_rule = rule;
    }
    
    /**
     * @brief 根据优先级排序思维链
     */
    void prioritizeChains(double current_time) {
        // 首先更新所有链路的优先级
        for (auto& chain : chains) {
            chain.updatePriority(current_time);
        }
        
        // 根据优先级排序
        std::sort(chains.begin(), chains.end(), 
            [this](const ThoughtChain<T>& a, const ThoughtChain<T>& b) {
                return priority_rule(a) > priority_rule(b);
            });
    }
    
    /**
     * @brief 获取主导思维链
     * @return 主导思维链
     */
    ThoughtChain<T> getDominantChain() const {
        if (chains.empty()) {
            throw std::runtime_error("No chains available");
        }
        return chains.front();
    }
    
    /**
     * @brief 获取特定时刻的决策链
     * @param timestamp 时间戳
     * @return 决策思维链
     */
    ThoughtChain<T> getDecisionChainAtTime(double timestamp, double tolerance = 0.1) {
        // 查找时间窗口内的所有链
        std::vector<ThoughtChain<T>> candidate_chains;
        for (const auto& chain : chains) {
            if (std::abs(chain.timestamp - timestamp) <= tolerance) {
                candidate_chains.push_back(chain);
            }
        }
        
        if (candidate_chains.empty()) {
            throw std::runtime_error("No chains found at specified time");
        }
        
        // 选择优先级最高的链
        auto it = std::max_element(candidate_chains.begin(), candidate_chains.end(),
            [this](const ThoughtChain<T>& a, const ThoughtChain<T>& b) {
                return priority_rule(a) < priority_rule(b);
            });
        
        return *it;
    }
    
    // ====================== 视图与潜在变量操作 ======================
    
    /**
     * @brief 创建新视图
     * @param id 视图ID
     * @param node_ids 包含的节点ID
     */
    void createView(const std::string& id, const std::vector<std::string>& node_ids) {
        View<T> view;
        view.id = id;
        
        for (const auto& node_id : node_ids) {
            if (nodes.find(node_id) != nodes.end()) {
                view.nodes.push_back(nodes[node_id]);
            }
        }
        
        view.generateLatentVariables();
        views.push_back(view);
    }
    
    /**
     * @brief 更新所有视图
     * @param learner 对比学习模块
     */
    void updateViews(ContrastiveLearner<T>& learner) {
        for (auto& view : views) {
            learner.updateView(view, views);
        }
    }
    
    // ====================== 解释生成 ======================
    
    /**
     * @brief 生成当前决策解释
     * @return 解释文本
     */
    std::string generateExplanation() const {
        if (chains.empty()) {
            return "No decision chains available";
        }
        
        const auto& dominant_chain = chains[0];
        if (dominant_chain.nodes.empty()) {
            return "Empty decision chain";
        }
        
        // 简化示例：基于节点类型生成解释
        std::string explanation;
        for (const auto& node : dominant_chain.nodes) {
            switch (node->type) {
                case NodeType::SENSOR_INPUT:
                    explanation += "Sensor input detected: ";
                    break;
                case NodeType::FEATURE_EXTRACTION:
                    explanation += "Extracted features: ";
                    break;
                case NodeType::SITUATION_ANALYSIS:
                    explanation += "Analyzed situation: ";
                    break;
                case NodeType::DECISION_MAKING:
                    explanation += "Decision made: ";
                    break;
                case NodeType::CONTROL_ACTION:
                    explanation += "Action taken: ";
                    break;
                case NodeType::CAUSAL_RELATION:
                    explanation += "Causal factor: ";
                    break;
                case NodeType::UNCERTAINTY:
                    explanation += "Uncertainty considered: ";
                    break;
            }
            
            // 添加节点特定信息
            if (auto sensor_node = dynamic_cast<SensorInputNode<T>*>(node.get())) {
                explanation += sensor_node->sensor_type + " data";
            } else if (auto decision_node = dynamic_cast<DecisionNode<T>*>(node.get())) {
                explanation += decision_node->decision_type;
            } else if (auto action_node = dynamic_cast<ControlActionNode<T>*>(node.get())) {
                explanation += action_node->action_type;
            }
            
            explanation += " → ";
        }
        
        // 移除最后的箭头
        if (explanation.size() >= 3) {
            explanation = explanation.substr(0, explanation.size() - 3);
        }
        
        return explanation;
    }
    
    // ====================== 获取与设置方法 ======================
    
    const std::vector<ThoughtChain<T>>& getAllChains() const { return chains; }
    const std::vector<View<T>>& getAllViews() const { return views; }
    const std::unordered_map<std::string, std::shared_ptr<DAGNode<T>>>& getAllNodes() const { return nodes; }
};

// ====================== 算子模板函数 ======================

/**
 * @brief 计算节点影响算子
 * @param node 节点
 * @return 影响值
 */
template <typename T>
double computeNodeInfluence(const DAGNode<T>& node) {
    return node.computeInfluence();
}

/**
 * @brief 更新链路优先级算子
 * @param chain 思维链
 * @param current_time 当前时间
 */
template <typename T>
void updateChainPriority(ThoughtChain<T>& chain, double current_time) {
    chain.updatePriority(current_time);
}

/**
 * @brief 分类链路算子
 * @param chain 思维链
 * @param classifier 分类器
 * @return 分类结果
 */
template <typename T>
std::string classifyChain(const ThoughtChain<T>& chain, const ChainClassifier<T>& classifier) {
    return classifier.classify(chain);
}

/**
 * @brief 计算视图相似度算子
 * @param view1 视图1
 * @param view2 视图2
 * @return 相似度
 */
template <typename T>
double computeViewSimilarity(const View<T>& view1, const View<T>& view2) {
    return view1.similarity(view2);
}

/**
 * @brief 编码节点算子
 * @param encoder 编码器
 * @param node 节点
 * @return 嵌入向量
 */
template <typename T>
std::vector<double> encodeNode(Encoder<T>& encoder, const DAGNode<T>& node) {
    return encoder.encode(node);
}

/**
 * @brief 学习双射映射算子
 * @param encoder 编码器
 * @param nodes 节点集合
 */
template <typename T>
void learnBijection(Encoder<T>& encoder, const std::vector<DAGNode<T>>& nodes) {
    encoder.learnSmoothBijection(nodes);
}

/**
 * @brief 计算对比损失算子
 * @param learner 对比学习模块
 * @param view1 视图1
 * @param view2 视图2
 * @return 损失值
 */
template <typename T>
double computeContrastiveLoss(const ContrastiveLearner<T>& learner, 
                             const View<T>& view1, 
                             const View<T>& view2) {
    return learner.computeLoss(view1, view2);
}

/**
 * @brief 更新视图算子
 * @param learner 对比学习模块
 * @param view 视图
 * @param all_views 所有视图
 */
template <typename T>
void updateView(ContrastiveLearner<T>& learner, View<T>& view, 
               const std::vector<View<T>>& all_views) {
    learner.updateView(view, all_views);
}

/**
 * @brief 创建视图算子
 * @param dag DAG
 * @param view_id 视图ID
 * @param node_ids 节点ID集合
 */
template <typename T>
void createView(ExplanationDAG<T>& dag, const std::string& view_id, 
               const std::vector<std::string>& node_ids) {
    dag.createView(view_id, node_ids);
}

/**
 * @brief 查找思维链算子
 * @param dag DAG
 */
template <typename T>
void findThoughtChains(ExplanationDAG<T>& dag) {
    dag.findThoughtChains();
}

} // namespace ExplainableDRL